use std::collections::HashMap;

use tracing::instrument;

use crate::{
    blocks::{Block, BlockType, ImageBlock, List, TextBlock, Title, TitleLevel},
    entities::{Element, ElementID, ElementType, Line, PageID},
    error::FerrulesError,
    layout::model::LayoutBBox,
};

/// This constant defines the minimum required intersection ratio between the bounding box of an
/// OCR-detected text line and a text block detected through layout analysis.
/// This approach ensures that only text lines significantly overlapping with a layout block are
/// paired, thus improving the accuracy of OCR-text and layout alignment.
const MIN_INTERSECTION_LAYOUT: f32 = 0.5;

/// Weights used for calculating distances between bounding boxes in layout analysis
/// X_WEIGHT is weighted higher (5.0) to prioritize horizontal alignment
/// Y_WEIGHT is weighted lower (1.0) to be more lenient with vertical spacing
const LAYOUT_DISTANCE_X_WEIGHT: f32 = 5.0;
const LAYOUT_DISTANCE_Y_WEIGHT: f32 = 1.0;

/// Maximum allowable distance between a text line and a layout block for assignment.
/// If the weighted distance (using X_WEIGHT and Y_WEIGHT) between a text line and
/// the nearest layout block exceeds this threshold, the line will not be assigned to any block.
/// This helps prevent incorrect assignments of text lines that are too far from layout blocks.
const MAXIMUM_ASSIGNMENT_DISTANCE: f32 = 20.0;

fn merge_or_create_elements(
    elements: &mut Vec<Element>,
    line: &Line,
    line_layout_block: &LayoutBBox,
    page_id: PageID,
) {
    if elements.is_empty() {
        let mut el = Element::from_layout_block(0, line_layout_block, page_id);
        el.push_line(line);
        elements.push(el);
        return;
    }

    // let last_el = elements.last_mut().unwrap();
    let matched_element = elements
        .iter_mut()
        .find(|e| e.layout_block_id == line_layout_block.id);

    match matched_element {
        Some(el) => {
            el.push_line(line);
        }
        None => {
            let mut element =
                Element::from_layout_block(elements.len() + 1, line_layout_block, page_id);
            element.push_line(line);
            elements.push(element);
        }
    }
}
/// Merges lines into blocks based on their layout, maintaining the order of lines.
///
/// This function takes a list of text boxes representing layout bounding boxes that contain text,
/// and a list of lines (which could be obtained from OCR or  PDF library pdfium2,
/// and merges these lines into blocks. The merging is done based on the intersection
/// of each line with the layout bounding boxes.
///
/// NOTE: The function iterates through lines to maintain global layout order as both OCR and pdfium return lines
/// in correct order
pub(crate) fn merge_lines_layout(
    layout_boxes: &[LayoutBBox],
    lines: &[Line],
    page_id: usize,
) -> Result<Vec<Element>, FerrulesError> {
    let line_block_iterator = lines.iter().map(|line| {
        // TODO: the max here is sometimes very far away from the line.
        // ex: megatrends.pdf, header is categorized as text-block but the intersection  happens
        //
        // Get max intersection block for the line
        let max_intersection_bbox = layout_boxes.iter().max_by(|a, b| {
            let a_intersection = a.bbox.intersection(&line.bbox);
            let b_intersection = b.bbox.intersection(&line.bbox);

            a_intersection.partial_cmp(&b_intersection).unwrap()
        });
        // Get min distance block for the line
        let min_distance_block = layout_boxes.iter().min_by(|a, b| {
            let a_intersection = a.bbox.distance(
                &line.bbox,
                LAYOUT_DISTANCE_X_WEIGHT,
                LAYOUT_DISTANCE_Y_WEIGHT,
            );
            let b_intersection = b.bbox.distance(
                &line.bbox,
                LAYOUT_DISTANCE_X_WEIGHT,
                LAYOUT_DISTANCE_Y_WEIGHT,
            );
            a_intersection.partial_cmp(&b_intersection).unwrap()
        });
        let max_intersection_bbox = max_intersection_bbox.and_then(|b| {
            if line.bbox.intersection(&b.bbox) / line.bbox.area() > MIN_INTERSECTION_LAYOUT {
                Some(b)
            } else {
                None
            }
        });
        // Compare based on distance
        let matched_block = if max_intersection_bbox.is_none() {
            min_distance_block.and_then(|b| {
                if b.bbox.distance(
                    &line.bbox,
                    LAYOUT_DISTANCE_X_WEIGHT,
                    LAYOUT_DISTANCE_Y_WEIGHT,
                ) < MAXIMUM_ASSIGNMENT_DISTANCE
                {
                    Some(b)
                } else {
                    None
                }
            })
        } else {
            max_intersection_bbox
        };
        (line, matched_block)
    });

    let mut headers = Vec::new();
    let mut elements = Vec::new();
    let mut footers = Vec::new();
    for (line, layout_block) in line_block_iterator {
        match &layout_block.as_ref() {
            Some(&line_layout_block) => match line_layout_block.label {
                "Page-header" => {
                    merge_or_create_elements(&mut headers, line, line_layout_block, page_id);
                }
                "Page-footer" => {
                    merge_or_create_elements(&mut footers, line, line_layout_block, page_id);
                }
                _ => {
                    merge_or_create_elements(&mut elements, line, line_layout_block, page_id);
                }
            },
            // Line is detected but isn't assignable to some layout element, for now skip
            None => {
                // TODO:
                // Check distance between line and the last element

                // let el = Element {
                //     id: 0,
                //     layout_block_id: -1,
                //     text_block: ElementText {
                //         text: line.text.to_owned(),
                //     },
                //     kind: ElementType::Text,
                //     page_id,
                //     bbox: line.bbox.clone(),
                // };
                // elements.push(el);
            }
        }
    }
    elements.append(&mut footers);

    headers.append(&mut elements);
    Ok(headers)
}

pub(crate) fn merge_remaining(
    elements: &mut Vec<Element>,
    remaining: &[&LayoutBBox],
    page_id: PageID,
) {
    for layout_box in remaining {
        let closest_block = elements
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let a_intersection = a.bbox.distance(
                    &layout_box.bbox,
                    LAYOUT_DISTANCE_X_WEIGHT,
                    LAYOUT_DISTANCE_Y_WEIGHT,
                );
                let b_intersection = b.bbox.distance(
                    &layout_box.bbox,
                    LAYOUT_DISTANCE_X_WEIGHT,
                    LAYOUT_DISTANCE_Y_WEIGHT,
                );
                a_intersection.partial_cmp(&b_intersection).unwrap()
            })
            .map(|(index, _)| index)
            .unwrap_or(elements.len());

        elements.insert(
            closest_block,
            Element::from_layout_block(elements.len(), layout_box, page_id),
        );
    }
}

#[instrument(skip_all)]
pub(crate) fn merge_elements_into_blocks(
    elements: Vec<Element>,
    title_level: HashMap<(PageID, ElementID), TitleLevel>,
) -> Result<Vec<Block>, FerrulesError> {
    let mut element_it = elements.into_iter().peekable();

    let mut blocks = Vec::new();
    let mut block_id = 0;
    let mut image_id = 0;
    while let Some(mut curr_el) = element_it.next() {
        match &mut curr_el.kind {
            ElementType::Text => {
                let text_block = Block {
                    id: block_id,
                    kind: crate::blocks::BlockType::TextBlock(TextBlock {
                        text: curr_el.text_block.text,
                    }),
                    pages_id: vec![curr_el.page_id],
                    bbox: curr_el.bbox,
                };
                // TODO : Change this to use some minimum gap
                // Check to see if we have another text block that is close
                // while let Some(next_el) = element_it.peek() {
                //     if matches!(next_el.kind, crate::entities::ElementType::Text(_))
                //         && (curr_el.bbox.distance(&next_el.bbox, 1.0, 1.0)
                //             < MAXIMUM_ASSIGNMENT_DISTANCE)
                //     {
                //         text_block.merge(next_el)?;
                //         element_it.next();
                //     } else {
                //         break;
                //     }
                // }
                block_id += 1;
                blocks.push(text_block);
            }
            ElementType::ListItem => {
                let mut list_block = Block {
                    id: block_id,
                    kind: BlockType::ListBlock(List {
                        items: vec![curr_el.text_block.text],
                    }),
                    pages_id: vec![curr_el.page_id],
                    bbox: curr_el.bbox,
                };

                while let Some(next_el) = element_it.peek() {
                    // TODO: add constraint on gap between bounding boxes on all dimensions (l,r,b,t)
                    if matches!(next_el.kind, crate::entities::ElementType::ListItem) {
                        let next_el = element_it.next().unwrap();
                        list_block.merge(next_el)?;
                    } else {
                        break;
                    }
                }
                block_id += 1;
                blocks.push(list_block);
            }
            ElementType::FootNote | ElementType::Caption => {
                // We find the closest image and create and image block
                loop {
                    match element_it.peek() {
                        None => {
                            // last element -> transform to txt block and break
                            let text_block = Block {
                                id: block_id,
                                kind: crate::blocks::BlockType::TextBlock(TextBlock {
                                    text: curr_el.text_block.text,
                                }),
                                pages_id: vec![curr_el.page_id],
                                bbox: curr_el.bbox,
                            };
                            element_it.next();
                            block_id += 1;
                            blocks.push(text_block);
                            break;
                        }
                        Some(next_el) => {
                            match &next_el.kind {
                                crate::entities::ElementType::FootNote
                                | crate::entities::ElementType::Caption => {
                                    // Merge this with a the caption
                                    curr_el.text_block.append_line(&next_el.text_block.text);
                                    element_it.next();
                                }
                                crate::entities::ElementType::Image => {
                                    curr_el.bbox.merge(&next_el.bbox);
                                    let img_block = Block {
                                        id: block_id,
                                        kind: BlockType::Image(ImageBlock {
                                            id: image_id,
                                            caption: Some(curr_el.text_block.text),
                                        }),
                                        pages_id: vec![next_el.page_id],
                                        bbox: curr_el.bbox,
                                    };
                                    image_id += 1;
                                    block_id += 1;
                                    blocks.push(img_block);
                                    element_it.next();
                                    break;
                                }
                                _ => {
                                    // This caption isn't associated with Image/Table, transform to textblock
                                    let text_block = Block {
                                        id: block_id,
                                        kind: crate::blocks::BlockType::TextBlock(TextBlock {
                                            text: curr_el.text_block.text,
                                        }),
                                        pages_id: vec![curr_el.page_id],
                                        bbox: curr_el.bbox,
                                    };
                                    block_id += 1;
                                    blocks.push(text_block);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            ElementType::Image => {
                match element_it.peek() {
                    None => {
                        let block = Block {
                            id: block_id,
                            kind: crate::blocks::BlockType::Image(ImageBlock {
                                id: image_id,
                                caption: None,
                            }),
                            pages_id: vec![curr_el.page_id],
                            bbox: curr_el.bbox,
                        };
                        element_it.next();
                        image_id += 1;
                        block_id += 1;
                        blocks.push(block);
                    }
                    Some(next_el) => {
                        match &next_el.kind {
                            crate::entities::ElementType::FootNote
                            | crate::entities::ElementType::Caption => {
                                // TODO: check if there is a case where there is multiple caption associated with the same image
                                let next_el = element_it.next().unwrap();
                                curr_el.bbox.merge(&next_el.bbox);
                                let block = Block {
                                    id: block_id,
                                    kind: crate::blocks::BlockType::Image(ImageBlock {
                                        id: image_id,
                                        caption: Some(next_el.text_block.text),
                                    }),
                                    pages_id: vec![curr_el.page_id],
                                    bbox: curr_el.bbox,
                                };
                                image_id += 1;
                                block_id += 1;
                                blocks.push(block);
                            }
                            _ => {
                                let block = Block {
                                    id: block_id,
                                    kind: crate::blocks::BlockType::Image(ImageBlock {
                                        id: image_id,
                                        caption: None,
                                    }),
                                    pages_id: vec![curr_el.page_id],
                                    bbox: curr_el.bbox,
                                };
                                image_id += 1;
                                block_id += 1;
                                blocks.push(block);
                            }
                        }
                    }
                }
            }
            ElementType::Header => {
                let mut header_block = Block {
                    id: block_id,
                    kind: BlockType::Header(TextBlock {
                        text: curr_el.text_block.text,
                    }),
                    pages_id: vec![curr_el.page_id],
                    bbox: curr_el.bbox,
                };

                while let Some(next_el) = element_it.peek() {
                    if matches!(next_el.kind, crate::entities::ElementType::Header) {
                        let next_el = element_it.next().unwrap();
                        header_block.merge(next_el)?;
                    } else {
                        break;
                    }
                }
                block_id += 1;
                blocks.push(header_block);
            }
            ElementType::Footer => {
                let mut footer_block = Block {
                    id: block_id,
                    kind: BlockType::Footer(TextBlock {
                        text: curr_el.text_block.text,
                    }),
                    pages_id: vec![curr_el.page_id],
                    bbox: curr_el.bbox,
                };

                while let Some(next_el) = element_it.peek() {
                    if matches!(next_el.kind, ElementType::Footer) {
                        let next_el = element_it.next().unwrap();
                        footer_block.merge(next_el)?;
                    } else {
                        break;
                    }
                }
                block_id += 1;
                blocks.push(footer_block);
            }
            ElementType::Title | ElementType::Subtitle => {
                let lvl = title_level
                    .get(&(curr_el.page_id, curr_el.id))
                    .unwrap_or(&0u8);
                let title = Block {
                    id: block_id,
                    kind: BlockType::Title(Title {
                        level: *lvl,
                        text: curr_el.text_block.text,
                    }),
                    pages_id: vec![curr_el.page_id],
                    bbox: curr_el.bbox,
                };
                block_id += 1;
                blocks.push(title);
            }
            _ => {
                continue;
            }
        }
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::entities::BBox;
    use crate::entities::ElementText;

    fn create_text_element(id: usize, page_id: usize, text: &str, bbox: BBox) -> Element {
        Element {
            id,
            layout_block_id: 0,
            kind: ElementType::Text,
            text_block: ElementText {
                text: text.to_owned(),
            },
            page_id,
            bbox,
        }
    }

    fn create_list_element(id: usize, page_id: usize, text: &str, bbox: BBox) -> Element {
        Element {
            id,
            layout_block_id: 0,
            kind: ElementType::ListItem,
            text_block: ElementText {
                text: text.to_string(),
            },
            page_id,
            bbox,
        }
    }

    fn create_caption_element(id: usize, page_id: usize, text: &str, bbox: BBox) -> Element {
        Element {
            id,
            layout_block_id: 0,
            kind: ElementType::Caption,
            text_block: ElementText {
                text: text.to_string(),
            },
            page_id,
            bbox,
        }
    }

    fn create_footnote_element(id: usize, page_id: usize, text: &str, bbox: BBox) -> Element {
        Element {
            id,
            layout_block_id: 0,
            kind: ElementType::FootNote,
            text_block: ElementText {
                text: text.to_string(),
            },
            page_id,
            bbox,
        }
    }
    fn create_image_element(id: usize, page_id: usize, bbox: BBox) -> Element {
        Element {
            id,
            layout_block_id: 0,
            kind: ElementType::Image,
            text_block: ElementText::default(),
            page_id,
            bbox,
        }
    }

    #[test]
    fn test_merge_adjacent_text_blocks() -> anyhow::Result<()> {
        let bbox1 = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let bbox2 = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_text_element(0, 1, "First paragraph", bbox1),
            create_text_element(1, 1, "Second paragraph", bbox2),
        ];

        let blocks = merge_elements_into_blocks(elements, HashMap::new())?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::TextBlock(text) = &blocks[0].kind {
            assert!(text.text.contains("First paragraph"));
            assert!(text.text.contains("Second paragraph"));
        } else {
            panic!("Expected TextBlock");
        }
        Ok(())
    }

    #[test]
    fn test_merge_list_items() -> anyhow::Result<()> {
        let bbox1 = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let bbox2 = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_list_element(0, 1, "First item", bbox1),
            create_list_element(1, 1, "Second item", bbox2.clone()),
            create_text_element(2, 1, "Random text", bbox2),
        ];

        let blocks = merge_elements_into_blocks(elements, HashMap::new())?;

        assert_eq!(blocks.len(), 2);
        if let BlockType::ListBlock(list) = &blocks[0].kind {
            assert_eq!(list.items.len(), 2);
            assert_eq!(list.items[0], "First item");
            assert_eq!(list.items[1], "Second item");
        } else {
            panic!("Expected ListItem");
        }
        Ok(())
    }

    #[test]
    fn test_merge_caption_with_image() -> anyhow::Result<()> {
        let caption_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let image_bbox = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_caption_element(0, 1, "Image caption", caption_bbox),
            create_image_element(1, 1, image_bbox),
        ];

        let blocks = merge_elements_into_blocks(elements, HashMap::new())?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::Image(image) = &blocks[0].kind {
            assert_eq!(image.caption, Some("Image caption".to_string()));
        } else {
            panic!("Expected Image");
        }
        Ok(())
    }

    #[test]
    fn test_merge_orphan_caption_becomes_text() -> anyhow::Result<()> {
        let caption_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };

        let elements = vec![create_caption_element(0, 1, "Orphan caption", caption_bbox)];

        let blocks = merge_elements_into_blocks(elements, HashMap::new())?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::TextBlock(text) = &blocks[0].kind {
            assert_eq!(text.text, "Orphan caption");
        } else {
            panic!("Expected TextBlock");
        }
        Ok(())
    }

    #[test]
    fn test_merge_distant_text_blocks_not_merged() -> anyhow::Result<()> {
        let bbox1 = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let bbox2 = BBox {
            x0: 0.0,
            y0: 20.0, // Far away
            x1: 2.0,
            y1: 22.0,
        };

        let elements = vec![
            create_text_element(0, 1, "First paragraph", bbox1),
            create_text_element(1, 1, "Distant paragraph", bbox2),
        ];

        let blocks = merge_elements_into_blocks(elements, HashMap::new())?;

        assert_eq!(blocks.len(), 2);
        Ok(())
    }

    #[test]
    fn test_merge_image_last_element() -> anyhow::Result<()> {
        let image_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };

        let elements = vec![create_image_element(0, 1, image_bbox)];

        let blocks = merge_elements_into_blocks(elements, HashMap::new())?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::Image(image) = &blocks[0].kind {
            assert_eq!(image.caption, None);
        } else {
            panic!("Expected Image block");
        }
        Ok(())
    }

    #[test]
    fn test_merge_image_with_following_caption() -> anyhow::Result<()> {
        let image_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let caption_bbox = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_image_element(0, 1, image_bbox),
            create_caption_element(1, 1, "Image Description", caption_bbox),
        ];

        let blocks = merge_elements_into_blocks(elements, HashMap::new())?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::Image(image) = &blocks[0].kind {
            assert_eq!(image.caption, Some("Image Description".to_string()));
        } else {
            panic!("Expected Image block with caption");
        }
        Ok(())
    }

    #[test]
    fn test_merge_image_with_following_non_caption() -> anyhow::Result<()> {
        let image_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let text_bbox = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_image_element(0, 1, image_bbox),
            create_text_element(1, 1, "Regular text", text_bbox),
        ];

        let blocks = merge_elements_into_blocks(elements, HashMap::new())?;

        assert_eq!(blocks.len(), 2);
        if let BlockType::Image(image) = &blocks[0].kind {
            assert_eq!(image.caption, None);
        } else {
            panic!("Expected Image block without caption");
        }

        if let BlockType::TextBlock(text) = &blocks[1].kind {
            assert_eq!(text.text, "Regular text");
        } else {
            panic!("Expected Text block");
        }
        Ok(())
    }

    #[test]
    fn test_merge_image_with_footnote() -> anyhow::Result<()> {
        let image_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let footnote_bbox = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_image_element(0, 1, image_bbox),
            create_footnote_element(1, 1, "Image Footnote", footnote_bbox),
        ];

        let blocks = merge_elements_into_blocks(elements, HashMap::new())?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::Image(image) = &blocks[0].kind {
            assert_eq!(image.caption, Some("Image Footnote".to_string()));
        } else {
            panic!("Expected Image block with footnote as caption");
        }
        Ok(())
    }
}
