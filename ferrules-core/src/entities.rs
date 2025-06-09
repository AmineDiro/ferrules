use image::DynamicImage;
use plsfix::fix_text;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::Duration};

use pdfium_render::prelude::{PdfFontWeight, PdfPageTextChar, PdfRect};

use crate::{blocks::Block, layout::model::LayoutBBox};

pub type PageID = usize;
pub type ElementID = usize;

const FERRULES_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct BBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl BBox {
    fn from_pdfrect(
        PdfRect {
            bottom,
            left,
            top,
            right,
        }: PdfRect,
        page_height: f32,
    ) -> Self {
        Self {
            x0: left.value,
            y0: page_height - top.value,
            x1: right.value,
            y1: page_height - bottom.value,
        }
    }

    #[inline(always)]
    pub fn center(&self) -> (f32, f32) {
        (
            self.x0 + self.width() / 2f32,
            self.y0 + self.height() / 2f32,
        )
    }

    #[inline(always)]
    pub fn height(&self) -> f32 {
        self.y1 - self.y0
    }
    #[inline(always)]
    pub fn width(&self) -> f32 {
        self.x1 - self.x0
    }
    #[inline(always)]
    pub fn area(&self) -> f32 {
        self.height() * self.width()
    }

    #[inline(always)]
    pub fn size(&self) -> (f32, f32) {
        (self.width(), self.height())
    }
    #[inline(always)]
    pub(crate) fn merge(&mut self, other: &Self) {
        self.x0 = self.x0.min(other.x0);
        self.y0 = self.y0.min(other.y0);
        self.x1 = self.x1.max(other.x1);
        self.y1 = self.y1.max(other.y1);
    }
    #[inline(always)]
    fn overlap_x(&self, other: &Self) -> f32 {
        f32::max(
            0f32,
            f32::min(self.x1, other.x1) - f32::max(self.x0, other.x0),
        )
    }
    #[inline(always)]
    fn overlap_y(&self, other: &Self) -> f32 {
        f32::max(
            0f32,
            f32::min(self.y1, other.y1) - f32::max(self.y0, other.y0),
        )
    }

    #[inline(always)]
    pub fn contains(&self, other: &Self) -> bool {
        other.x0 >= self.x0 && other.y0 >= self.y0 && other.x1 <= self.x1 && other.y1 <= self.y1
    }

    #[inline(always)]
    pub fn relaxed_iou(&self, other: &Self) -> f32 {
        let a = self.intersection(other);
        let b = self.area().min(other.area());
        a / b
    }

    #[inline(always)]
    pub fn iou(&self, other: &Self) -> f32 {
        self.intersection(other) / self.union(other)
    }

    #[inline(always)]
    pub fn intersection(&self, other: &Self) -> f32 {
        self.overlap_x(other) * self.overlap_y(other)
    }

    #[inline(always)]
    fn union(&self, other: &Self) -> f32 {
        other.area() + self.area() - self.intersection(other)
    }

    #[inline(always)]
    pub(crate) fn distance(&self, other: &Self, x_weight: f32, y_weight: f32) -> f32 {
        let point_a = self.center();
        let point_b = other.center();

        (point_a.0 - point_b.0).powi(2) * x_weight + (point_a.1 - point_b.1).powi(2) * y_weight
    }

    fn _rotate(self) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ElementText {
    pub(crate) text: String,
}

impl ElementText {
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
    pub fn push_first(&mut self, txt: &str) {
        self.text.push_str(txt);
    }
    pub fn append_line(&mut self, txt: &str) {
        self.text.push(' ');
        self.text.push_str(txt);
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "element_type")]
pub enum ElementType {
    Header,
    FootNote,
    Footer,
    Text,
    Title,
    Subtitle,
    ListItem,
    Caption,
    Image,
    Table,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Element {
    pub id: ElementID,
    pub layout_block_id: i32,
    pub text_block: ElementText,
    pub kind: ElementType,
    pub page_id: usize,
    pub bbox: BBox,
}

impl Element {
    pub fn from_layout_block(id: usize, layout_block: &LayoutBBox, page_id: usize) -> Self {
        let kind = match layout_block.label {
            "Caption" => ElementType::Caption,
            "Formula" | "Text" => ElementType::Text,
            "List-item" => ElementType::ListItem,
            "Footnote" => ElementType::FootNote,
            "Page-footer" => ElementType::Footer,
            "Page-header" => ElementType::Header,
            "Title" => ElementType::Title,
            "Section-header" => ElementType::Subtitle,
            "Table" => ElementType::Table,
            "Picture" => ElementType::Image,
            _ => {
                unreachable!("can't have other type of layout bbox")
            }
        };
        Self {
            id,
            kind,
            layout_block_id: layout_block.id,
            page_id,
            text_block: Default::default(),
            bbox: layout_block.bbox.to_owned(),
        }
    }
    pub fn push_line(&mut self, line: &Line) {
        if self.text_block.is_empty() {
            self.text_block.push_first(&line.text);
        } else {
            self.text_block.append_line(&line.text);
        }
    }
}

#[derive(Debug)]
pub struct StructuredPage {
    pub id: PageID,
    pub width: f32,
    pub height: f32,
    // pub rotation: PdfPageRenderRotation,
    pub need_ocr: bool,
    pub image: DynamicImage,
    pub elements: Vec<Element>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Page {
    pub id: PageID,
    pub width: f32,
    pub height: f32,

    #[serde(skip_serializing, skip_deserializing)]
    pub image: DynamicImage,
    // pub rotation: PdfPageRenderRotation,
    pub need_ocr: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentMetadata {
    #[serde(with = "serde_millis")]
    pub parsing_duration: Duration,
    pub ferrules_version: String,
}

impl DocumentMetadata {
    pub fn new(parsing_duration: Duration) -> Self {
        Self {
            parsing_duration,
            ferrules_version: FERRULES_VERSION.to_owned(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ParsedDocument {
    pub doc_name: String,
    pub pages: Vec<Page>,
    pub blocks: Vec<Block>,
    pub debug_path: Option<PathBuf>,
    pub metadata: DocumentMetadata,
}

#[derive(Debug)]
pub struct CharSpan {
    pub bbox: BBox,
    pub text: String,
    pub rotation: f32,
    pub font_name: String,
    pub font_size: f32,
    pub font_weight: Option<PdfFontWeight>,
    pub char_start_idx: usize,
    pub char_end_idx: usize,
}

impl CharSpan {
    pub fn new_from_char(char: &PdfPageTextChar, page_bbox: &BBox) -> Self {
        Self {
            bbox: BBox::from_pdfrect(
                char.tight_bounds()
                    .expect("Error init span tight bound char"),
                page_bbox.height(),
            ),
            text: char.unicode_char().unwrap_or_default().into(),
            font_name: char.font_name(),
            font_weight: char.font_weight(),
            font_size: char.unscaled_font_size().value,
            rotation: char.get_rotation_clockwise_degrees(),
            char_start_idx: char.index(),
            char_end_idx: char.index(),
        }
    }
    pub fn append(&mut self, char: &PdfPageTextChar, page_bbox: &BBox) -> Option<()> {
        let char_rotation = char.get_rotation_clockwise_degrees();
        if char.unscaled_font_size().value != self.font_size
            || char.font_name() != self.font_name
            || char.font_weight() != self.font_weight
            || char_rotation != self.rotation
        {
            None
        } else {
            let char_bbox = BBox::from_pdfrect(
                char.loose_bounds().expect("error tight bound"),
                page_bbox.height(),
            );
            self.text.push(char.unicode_char().unwrap_or_default());
            self.char_end_idx = char.index();
            self.bbox.merge(&char_bbox);
            Some(())
        }
    }
}
#[derive(Default)]
pub struct Line {
    pub text: String,
    pub bbox: BBox,
    pub rotation: f32,
    pub spans: Vec<CharSpan>,
}

impl std::fmt::Debug for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Line {{ text: {}, height: {:.2}, width: {:.2}, °, \nspans: [{}] }}\n",
            self.text.trim(),
            self.bbox.height(),
            self.bbox.width(),
            self.spans
                .iter()
                .map(|span| format!(
                    "\n\rtext: {}, height: {:?}, width: {:.2}",
                    span.text.trim(),
                    span.bbox.height(),
                    span.bbox.width()
                ))
                .collect::<Vec<_>>()
                .join("\r")
        )
    }
}

impl Line {
    pub fn new_from_span(span: CharSpan) -> Self {
        Self {
            bbox: span.bbox.clone(),
            text: span.text.clone(),
            rotation: span.rotation,
            spans: vec![span],
        }
    }
    // TODO: find a better pattern here
    // return Some if we fail to append the span-> not great
    pub fn append(&mut self, span: CharSpan) -> Result<(), CharSpan> {
        if span.rotation != self.rotation
        // NOTE: sometimes pdfium doesn't inject a linebreak, so we check the span positions
        || span.bbox.y0 > self.bbox.y1
        || span.text.ends_with("\n") || span.text.ends_with("\x02")
        {
            self.text = fix_text(&self.text, None);
            Err(span)
        } else {
            if self.bbox.height() == 0f32 || self.bbox.width() == 0f32 {
                // The previous span in line is a linebreak
                self.bbox = span.bbox.clone();
            } else {
                self.bbox.merge(&span.bbox);
            }
            self.text.push_str(&span.text);
            self.spans.push(span);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersection() {
        let bbox1 = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let bbox2 = BBox {
            x0: 1.0,
            y0: 1.0,
            x1: 3.0,
            y1: 3.0,
        };
        let bbox3 = BBox {
            x0: 2.0,
            y0: 2.0,
            x1: 4.0,
            y1: 4.0,
        };
        let bbox4 = BBox {
            x0: 3.0,
            y0: 3.0,
            x1: 5.0,
            y1: 5.0,
        }; // No overlap
        let bbox5 = BBox {
            x0: -1.0,
            y0: -1.0,
            x1: 1.0,
            y1: 1.0,
        }; // Negative coordinates
        let bbox6 = BBox {
            x0: 0.5,
            y0: 0.5,
            x1: 1.5,
            y1: 1.5,
        }; // Inside bbox1

        // Edge Cases
        assert_eq!(bbox1.intersection(&bbox3), 0.0);
        assert_eq!(bbox1.intersection(&bbox4), 0.0); // Adjacent
        assert_eq!(bbox5.intersection(&bbox1), 1.0); // Overlaps partially with bbox1

        // Overlaps
        assert_eq!(bbox1.intersection(&bbox2), 1.0);
        assert_eq!(bbox1.intersection(&bbox6), bbox6.area()); // bbox6 is inside bbox1

        // Sanity Checks
        assert_eq!(bbox1.intersection(&bbox1), bbox1.area());
    }

    #[test]
    fn test_union() {
        let bbox1 = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let bbox2 = BBox {
            x0: 1.0,
            y0: 1.0,
            x1: 3.0,
            y1: 3.0,
        };
        let bbox3 = BBox {
            x0: 2.0,
            y0: 2.0,
            x1: 4.0,
            y1: 4.0,
        };
        let bbox4 = BBox {
            x0: 3.0,
            y0: 3.0,
            x1: 5.0,
            y1: 5.0,
        }; // No overlap
        let bbox5 = BBox {
            x0: -1.0,
            y0: -1.0,
            x1: 1.0,
            y1: 1.0,
        }; // Negative coordinates

        // Edge Cases
        assert_eq!(bbox1.union(&bbox3), 8.0);
        assert_eq!(bbox1.union(&bbox4), 8.0); // Completely non-overlapping
        assert_eq!(bbox5.union(&bbox1), 7.0); // Negative coordinate case

        // Overlapping
        assert_eq!(bbox1.union(&bbox2), 7.0);

        // Sanity Checks
        assert_eq!(bbox1.union(&bbox1), bbox1.area());
    }

    #[test]
    fn test_iou() {
        let bbox1 = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let bbox2 = BBox {
            x0: 1.0,
            y0: 1.0,
            x1: 3.0,
            y1: 3.0,
        };
        let bbox3 = BBox {
            x0: 2.0,
            y0: 2.0,
            x1: 4.0,
            y1: 4.0,
        };
        let bbox4 = BBox {
            x0: 3.0,
            y0: 3.0,
            x1: 5.0,
            y1: 5.0,
        }; // No overlap
        let bbox6 = BBox {
            x0: 0.5,
            y0: 0.5,
            x1: 1.5,
            y1: 1.5,
        }; // Inside bbox1

        // Sanity Checks
        assert_eq!(bbox1.iou(&bbox1), 1.0);
        // Completely non-overlapping
        assert_eq!(bbox1.iou(&bbox4), 0.0);

        // Edge Cases
        assert_eq!(bbox1.iou(&bbox3), 0.0);

        // Overlapping
        assert_eq!(bbox1.iou(&bbox2), 1.0 / 7.0);
        assert_eq!(bbox1.iou(&bbox6), bbox6.area() / bbox1.area()); // bbox6 is inside bbox1
    }
    #[test]
    fn test_distance() {
        let bbox1 = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let bbox2 = BBox {
            x0: 3.0,
            y0: 3.0,
            x1: 5.0,
            y1: 5.0,
        };
        let bbox3 = BBox {
            x0: 0.0,
            y0: 2.0,
            x1: 2.0,
            y1: 4.0,
        };

        let x_weight = 1.0;
        let y_weight = 1.0;

        // Standard Case
        let distance = bbox1.distance(&bbox2, x_weight, y_weight);
        assert_eq!(distance, 18.0); // ((4 - 1)^2 + (4 - 1)^2)

        // Boxes with Overlapping Edges
        let distance = bbox1.distance(&bbox3, x_weight, y_weight);
        assert_eq!(distance, 4.0); // ((1 - 1)^2 + (3 - 1)^2)

        // // Identical Boxes
        let distance = bbox1.distance(&bbox1, x_weight, y_weight);
        assert_eq!(distance, 0.0);

        // // Test with different weights
        let x_weight = 2.0;
        let y_weight = 3.0;
        let distance = bbox1.distance(&bbox2, x_weight, y_weight);
        assert_eq!(distance, 45.0); // (3-1)^2 * 2 + (4-1)^2 * 3
    }
}
