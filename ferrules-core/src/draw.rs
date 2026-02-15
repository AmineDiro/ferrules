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
) -> anyhow::Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
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
) -> anyhow::Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
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

pub(crate) fn draw_blocks(
    bboxes: &[Block],
    page_img: &DynamicImage,
) -> anyhow::Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    // Convert the dynamic image to RGBA for in-place drawing.
    let mut out_img = page_img.to_rgba8();

    let font: FontArc = load_font();
    for block in bboxes {
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

    Ok(out_img)
}
