use image::{DynamicImage, ImageBuffer, Rgba};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

use crate::blocks::Block;
use crate::entities::Line;
use crate::error::FerrulesError;
use crate::layout::model::LayoutBBox;
use crate::ocr::OCRLines;

use ab_glyph::FontArc;

static FONT_BYTES: &[u8] = include_bytes!("../../font/Arial.ttf");

const BLOCK_COLOR: [u8; 4] = [209, 139, 0, 255];
const LAYOUT_COLOR: [u8; 4] = [0, 0, 255, 255];
const LINE_OCR_COLOR: [u8; 4] = [17, 138, 1, 255];
const LINE_PDFIRUM_COLOR: [u8; 4] = [255, 0, 0, 255];
const PATH_COLOR: [u8; 4] = [0, 255, 0, 255]; // Green for paths
const TABLE_ROW_COLOR: [u8; 4] = [0, 0, 255, 255]; // Blue
const TABLE_CELL_COLOR: [u8; 4] = [255, 0, 0, 255]; // Red
const VISION_COLOR: [u8; 4] = [138, 43, 226, 96]; // Violet/Purple with medium alpha

fn load_font() -> FontArc {
    FontArc::try_from_slice(FONT_BYTES).unwrap()
}

pub(crate) fn draw_text_lines(
    lines: &[Line],
    page_img: &DynamicImage,
    is_ocr: bool,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, FerrulesError> {
    // Convert the dynamic image to RGBA for in-place drawing.
    let mut out_img = page_img.to_rgba8();

    let color = if is_ocr {
        Rgba(LINE_PDFIRUM_COLOR)
    } else {
        Rgba(LINE_OCR_COLOR)
    };
    // Iterate over all bounding boxes and draw them.
    for line in lines {
        let x0 = (line.bbox.x0) as i32;
        let y0 = (line.bbox.y0) as i32;
        let x1 = (line.bbox.x1) as i32;
        let y1 = (line.bbox.y1) as i32;

        let width = (x1 - x0).max(1) as u32;
        let height = (y1 - y0).max(1) as u32;

        let rect = Rect::at(x0, y0).of_size(width, height);
        draw_hollow_rect_mut(&mut out_img, rect, color);
    }

    Ok(out_img)
}

pub(crate) fn draw_layout_bboxes(
    bboxes: &[LayoutBBox],
    page_img: &DynamicImage,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, FerrulesError> {
    // Convert the dynamic image to RGBA for in-place drawing.
    let mut out_img = page_img.to_rgba8();

    let font: FontArc = load_font();
    for layout_box in bboxes {
        let x0 = layout_box.bbox.x0 as i32;
        let y0 = layout_box.bbox.y0 as i32;
        let x1 = layout_box.bbox.x1 as i32;
        let y1 = layout_box.bbox.y1 as i32;

        let width = (x1 - x0).max(0) as u32;
        let height = (y1 - y0).max(0) as u32;

        let rect = Rect::at(x0, y0).of_size(width, height);

        draw_hollow_rect_mut(&mut out_img, rect, Rgba(LAYOUT_COLOR));
        let legend = format!("{} {:.2}", layout_box.label, layout_box.proba);
        let scale = 50;
        let legend_size = page_img.width().max(page_img.height()) / scale;
        imageproc::drawing::draw_text_mut(
            &mut out_img,
            image::Rgba(LAYOUT_COLOR),
            layout_box.bbox.x0 as i32,
            (layout_box.bbox.y0 - legend_size as f32) as i32,
            legend_size as f32,
            &font,
            &legend,
        );
    }

    Ok(out_img)
}

#[allow(dead_code)]
pub(crate) fn draw_ocr_bboxes(
    bboxes: &[OCRLines],
    page_img: &DynamicImage,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, FerrulesError> {
    // Convert the dynamic image to RGBA for in-place drawing.
    let mut out_img = page_img.to_rgba8();

    let font: FontArc = load_font();
    for ocr_box in bboxes {
        let x0 = ocr_box.bbox.x0 as i32;
        let y0 = ocr_box.bbox.y0 as i32;
        let x1 = ocr_box.bbox.x1 as i32;
        let y1 = ocr_box.bbox.y1 as i32;

        let width = (x1 - x0).max(1) as u32;
        let height = (y1 - y0).max(1) as u32;

        let rect = Rect::at(x0, y0).of_size(width, height);

        draw_hollow_rect_mut(&mut out_img, rect, Rgba(LINE_OCR_COLOR));
        let legend = format!("{} ({:.2})", ocr_box.text, ocr_box.confidence);
        let scale = 70;
        let legend_size = page_img.width().max(page_img.height()) / scale;
        imageproc::drawing::draw_text_mut(
            &mut out_img,
            image::Rgba(LINE_OCR_COLOR),
            ocr_box.bbox.x0 as i32,
            (ocr_box.bbox.y0 - legend_size as f32) as i32,
            legend_size as f32,
            &font,
            &legend,
        );
    }

    Ok(out_img)
}

pub(crate) fn draw_paths(
    paths: &[crate::entities::PDFPath],
    page_img: &DynamicImage,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, FerrulesError> {
    let mut out_img = page_img.to_rgba8();

    for path in paths {
        for segment in &path.segments {
            match segment {
                crate::entities::Segment::Line { start, end } => {
                    let start = (start.0 as f32, start.1 as f32);
                    let end = (end.0 as f32, end.1 as f32);
                    imageproc::drawing::draw_line_segment_mut(
                        &mut out_img,
                        start,
                        end,
                        Rgba(PATH_COLOR),
                    );
                }
                crate::entities::Segment::Rect { bbox } => {
                    let x0 = bbox.x0 as i32;
                    let y0 = bbox.y0 as i32;
                    let width = (bbox.width() as u32).max(1);
                    let height = (bbox.height() as u32).max(1);
                    let rect = Rect::at(x0, y0).of_size(width, height);
                    draw_hollow_rect_mut(&mut out_img, rect, Rgba(PATH_COLOR));
                }
            }
        }
    }

    Ok(out_img)
}

pub(crate) fn draw_blocks(
    bboxes: &[Block],
    page_img: &DynamicImage,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, FerrulesError> {
    // Convert the dynamic image to RGBA for in-place drawing.
    let mut out_img = page_img.to_rgba8();

    let font: FontArc = load_font();
    for block in bboxes {
        match &block.kind {
            crate::blocks::BlockType::Table(table) => {
                draw_table_structure(table, &mut out_img);
            }
            _ => {
                let x0 = block.bbox.x0 as i32;
                let y0 = block.bbox.y0 as i32;
                let x1 = block.bbox.x1 as i32;
                let y1 = block.bbox.y1 as i32;

                let width = (x1 - x0).max(1) as u32;
                let height = (y1 - y0).max(1) as u32;

                let rect = Rect::at(x0, y0).of_size(width, height);

                draw_hollow_rect_mut(&mut out_img, rect, Rgba(BLOCK_COLOR));
                let scale = 70;
                let legend_size = page_img.width().max(page_img.height()) / scale;
                imageproc::drawing::draw_text_mut(
                    &mut out_img,
                    image::Rgba(BLOCK_COLOR),
                    block.bbox.x0 as i32,
                    (block.bbox.y0 - legend_size as f32) as i32,
                    legend_size as f32,
                    &font,
                    block.label(),
                );
            }
        }
    }

    Ok(out_img)
}

fn draw_filled_rect_alpha(img: &mut image::RgbaImage, rect: Rect, color: Rgba<u8>) {
    let alpha = color[3] as f32 / 255.0;
    let (w, h) = img.dimensions();

    let left = rect.left().max(0);
    let top = rect.top().max(0);
    let right = rect.right().min(w as i32);
    let bottom = rect.bottom().min(h as i32);

    for y in top..bottom {
        for x in left..right {
            let px = img.get_pixel_mut(x as u32, y as u32);
            px[0] = ((1.0 - alpha) * px[0] as f32 + alpha * color[0] as f32) as u8;
            px[1] = ((1.0 - alpha) * px[1] as f32 + alpha * color[1] as f32) as u8;
            px[2] = ((1.0 - alpha) * px[2] as f32 + alpha * color[2] as f32) as u8;
        }
    }
}

fn draw_table_structure(
    table_block: &crate::blocks::TableBlock,
    out_img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
) {
    // 1. Draw Vision hints (bottom layer)
    if let crate::blocks::TableAlgorithm::Vision = &table_block.algorithm {
        for row in &table_block.rows {
            // Row detection
            let row_rect = Rect::at(row.bbox.x0 as i32, row.bbox.y0 as i32).of_size(
                row.bbox.width().max(1.0) as u32,
                row.bbox.height().max(1.0) as u32,
            );
            draw_filled_rect_alpha(out_img, row_rect, Rgba(VISION_COLOR));
        }

        // Column detections based on first row cells
        if let Some(first_row) = table_block.rows.first() {
            let y_start = first_row.bbox.y0;
            let y_end = table_block
                .rows
                .last()
                .map(|r| r.bbox.y1)
                .unwrap_or(y_start);
            for cell in &first_row.cells {
                let col_rect = Rect::at(cell.bbox.x0 as i32, y_start as i32).of_size(
                    cell.bbox.width().max(1.0) as u32,
                    (y_end - y_start).max(1.0) as u32,
                );
                draw_filled_rect_alpha(out_img, col_rect, Rgba(VISION_COLOR));
            }
        }
    }

    // 2. Draw Table BBox
    // We can assume the table block itself is drawn by the caller if needed,
    // but here we focus on internal structure.

    // Draw Rows
    for row in &table_block.rows {
        let x0 = row.bbox.x0 as i32;
        let y0 = row.bbox.y0 as i32;
        let x1 = row.bbox.x1 as i32;
        let y1 = row.bbox.y1 as i32;
        let width = (x1 - x0).max(1) as u32;
        let height = (y1 - y0).max(1) as u32;
        let rect = Rect::at(x0, y0).of_size(width, height);
        draw_hollow_rect_mut(out_img, rect, Rgba(TABLE_ROW_COLOR));

        // Draw Cells
        for cell in &row.cells {
            let x0 = cell.bbox.x0 as i32;
            let y0 = cell.bbox.y0 as i32;
            let x1 = cell.bbox.x1 as i32;
            let y1 = cell.bbox.y1 as i32;
            let width = (x1 - x0).max(1) as u32;
            let height = (y1 - y0).max(1) as u32;
            let rect = Rect::at(x0, y0).of_size(width, height);
            draw_hollow_rect_mut(out_img, rect, Rgba(TABLE_CELL_COLOR));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::{TableBlock, TableCell, TableRow};
    use crate::entities::BBox;
    use image::RgbaImage;

    #[test]
    fn test_draw_table_structure() {
        let page_img = DynamicImage::ImageRgba8(RgbaImage::new(100, 100));
        let mut out_img = page_img.to_rgba8();

        // Create dummy table
        let table_block = TableBlock {
            id: 1,
            caption: None,
            has_borders: true,
            rows: vec![
                TableRow {
                    is_header: true,
                    bbox: BBox {
                        x0: 10.0,
                        y0: 10.0,
                        x1: 90.0,
                        y1: 30.0,
                    },
                    cells: vec![
                        TableCell {
                            text: "Header".to_string(),
                            bbox: BBox {
                                x0: 10.0,
                                y0: 10.0,
                                x1: 50.0,
                                y1: 30.0,
                            },
                            row_span: 1,
                            col_span: 1,
                            content_ids: vec![],
                        },
                        TableCell {
                            text: "Header 2".to_string(),
                            bbox: BBox {
                                x0: 50.0,
                                y0: 10.0,
                                x1: 90.0,
                                y1: 30.0,
                            },
                            row_span: 1,
                            col_span: 1,
                            content_ids: vec![],
                        },
                    ],
                },
                TableRow {
                    is_header: false,
                    bbox: BBox {
                        x0: 10.0,
                        y0: 30.0,
                        x1: 90.0,
                        y1: 50.0,
                    },
                    cells: vec![
                        TableCell {
                            text: "Cell 1".to_string(),
                            bbox: BBox {
                                x0: 10.0,
                                y0: 30.0,
                                x1: 50.0,
                                y1: 50.0,
                            },
                            row_span: 1,
                            col_span: 1,
                            content_ids: vec![],
                        },
                        TableCell {
                            text: "Cell 2".to_string(),
                            bbox: BBox {
                                x0: 50.0,
                                y0: 30.0,
                                x1: 90.0,
                                y1: 50.0,
                            },
                            row_span: 1,
                            col_span: 1,
                            content_ids: vec![],
                        },
                    ],
                },
            ],
            algorithm: crate::blocks::TableAlgorithm::Unknown,
        };

        // Directly call the internal function to test it
        draw_table_structure(&table_block, &mut out_img);

        // Also test via draw_blocks
        let block = crate::blocks::Block {
            id: 1,
            kind: crate::blocks::BlockType::Table(table_block),
            pages_id: vec![0],
            bbox: BBox {
                x0: 10.0,
                y0: 10.0,
                x1: 90.0,
                y1: 50.0,
            },
        };

        let result = draw_blocks(&[block], &page_img);
        assert!(result.is_ok());
    }
}
