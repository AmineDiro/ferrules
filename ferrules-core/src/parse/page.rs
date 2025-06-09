use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use image::DynamicImage;
use tracing::instrument;

use crate::{
    draw::{draw_blocks, draw_layout_bboxes, draw_text_lines},
    entities::{Element, Line, PageID, StructuredPage},
    error::FerrulesError,
    layout::{
        model::LayoutBBox, Metadata, ParseLayoutQueue, ParseLayoutRequest, ParseLayoutResponse,
    },
    ocr::parse_image_ocr,
};

use super::{
    merge::{merge_elements_into_blocks, merge_lines_layout, merge_remaining},
    native::ParseNativePageResult,
};

/// This constant defines the minimum ratio between the area of text lines identified
/// by the pdfium2 and the area of text regions detected through layout analysis.
/// If this ratio falls below the threshold of 0.5 (or 50%), it indicates that the page
/// may not have enough __native__ lines, and therefore should
/// be considered for OCR to ensure accurate text extraction.
const MIN_LAYOUT_COVERAGE_THRESHOLD: f32 = 0.5;

fn page_needs_ocr(text_boxes: &[&LayoutBBox], text_lines: &[Line]) -> bool {
    let line_area = text_lines.iter().map(|l| l.bbox.area()).sum::<f32>();
    let text_layoutbbox_area = text_boxes.iter().map(|l| l.bbox.area()).sum::<f32>();

    if text_layoutbbox_area > 0f32 {
        line_area / text_layoutbbox_area < MIN_LAYOUT_COVERAGE_THRESHOLD
    } else {
        true
    }
}

#[instrument(skip_all)]
fn build_page_elements(
    page_layout: &[LayoutBBox],
    text_lines: &[Line],
    page_idx: PageID,
) -> Result<Vec<Element>, FerrulesError> {
    let mut elements = merge_lines_layout(page_layout, text_lines, page_idx)?;
    let merged_layout_blocks_ids = elements
        .iter()
        .map(|e| e.layout_block_id)
        .collect::<Vec<_>>();
    let unmerged_layout_boxes: Vec<&LayoutBBox> = page_layout
        .iter()
        .filter(|&b| !merged_layout_blocks_ids.contains(&b.id))
        .collect();

    merge_remaining(&mut elements, &unmerged_layout_boxes, page_idx);
    Ok(elements)
}

#[instrument(skip_all)]
fn parse_page_text(
    native_text_lines: Vec<Line>,
    page_layout: &[LayoutBBox],
    page_image: &DynamicImage,
    downscale_factor: f32,
) -> Result<(Vec<Line>, bool), FerrulesError> {
    let text_layout_box: Vec<&LayoutBBox> =
        page_layout.iter().filter(|b| b.is_text_block()).collect();
    let need_ocr = page_needs_ocr(&text_layout_box, &native_text_lines);

    let ocr_result = if need_ocr {
        parse_image_ocr(page_image, downscale_factor).ok()
    } else {
        None
    };

    let lines = if need_ocr && ocr_result.is_some() {
        let lines = ocr_result
            .as_ref()
            .unwrap()
            .iter()
            .map(|ocr_line| ocr_line.to_line())
            .collect::<Vec<_>>();
        lines
    } else {
        native_text_lines
    };
    Ok((lines, need_ocr))
}

#[instrument(
    skip_all,
    fields(
        layout_queue_time_ms,
        layout_parse_duration_ms,
        parse_native_duration_ms,
    )
)]
pub async fn parse_page_full(
    parse_native_result: ParseNativePageResult,
    debug_dir: Option<PathBuf>,
    layout_queue: ParseLayoutQueue,
) -> Result<StructuredPage, FerrulesError> {
    let span = tracing::Span::current();
    let ParseNativePageResult {
        page_id,
        text_lines,
        page_bbox,
        page_image,
        page_image_scale1,
        downscale_factor,
        metadata: parse_native_metadata,
    } = parse_native_result;
    let (layout_tx, layout_rx) = tokio::sync::oneshot::channel();

    let layout_req = ParseLayoutRequest {
        page_id,
        page_image: Arc::clone(&page_image),
        downscale_factor,
        metadata: Metadata {
            response_tx: layout_tx,
            queue_time: Instant::now(),
        },
    };
    layout_queue.push(layout_req).await?;

    let ParseLayoutResponse {
        page_id,
        layout_bbox: page_layout,
        layout_parse_duration_ms,
        layout_queue_time_ms,
    } = layout_rx
        .await
        // TODO: better unwrapping
        .map_err(|_| FerrulesError::LayoutParsingError)?
        .map_err(|_| FerrulesError::LayoutParsingError)?;

    let (text_lines, need_ocr) =
        parse_page_text(text_lines, &page_layout, &page_image, downscale_factor)?;

    // Merging elements with layout
    let elements = build_page_elements(&page_layout, &text_lines, page_id)?;
    if let Some(tmp_dir) = debug_dir {
        debug_page(
            &tmp_dir,
            page_id,
            &page_image_scale1,
            &text_lines,
            need_ocr,
            &page_layout,
            &elements,
        )?
    };

    let structured_page = StructuredPage {
        id: page_id,
        width: page_bbox.width(),
        height: page_bbox.height(),
        image: page_image_scale1,
        elements,
        need_ocr,
    };

    span.record(
        "layout_queue_time_ms",
        format!("{:?}", layout_queue_time_ms),
    );
    span.record(
        "layout_parse_duration_ms",
        format!("{:?}", layout_parse_duration_ms),
    );

    span.record(
        "parse_native_duration_ms",
        format!("{:?}", parse_native_metadata.parse_native_duration_ms),
    );
    // TODO: add OCR timings

    Ok(structured_page)
}

fn debug_page(
    tmp_dir: &Path,
    page_idx: PageID,
    page_image: &DynamicImage,
    text_lines: &[Line],
    need_ocr: bool,
    page_layout: &[LayoutBBox],
    elements: &[Element],
) -> Result<(), FerrulesError> {
    let output_file = tmp_dir.join(format!("page_{}.png", page_idx));
    let final_output_file = tmp_dir.join(format!("page_blocks_{}.png", page_idx));
    let out_img = draw_text_lines(text_lines, page_image, need_ocr).map_err(|_| {
        FerrulesError::DebugPageError {
            tmp_dir: tmp_dir.to_path_buf(),
            page_idx,
        }
    })?;
    let out_img = draw_layout_bboxes(page_layout, &out_img.into()).map_err(|_| {
        FerrulesError::DebugPageError {
            tmp_dir: tmp_dir.to_path_buf(),
            page_idx,
        }
    })?;
    // Draw the final prediction -
    // TODO: Implement titles hashmap for titles in the page
    let blocks = merge_elements_into_blocks(elements.to_vec(), HashMap::new())?;
    let final_img =
        draw_blocks(&blocks, page_image).map_err(|_| FerrulesError::DebugPageError {
            tmp_dir: tmp_dir.to_path_buf(),
            page_idx,
        })?;
    out_img
        .save(output_file)
        .map_err(|_| FerrulesError::DebugPageError {
            tmp_dir: tmp_dir.to_path_buf(),
            page_idx,
        })?;

    final_img
        .save(final_output_file)
        .map_err(|_| FerrulesError::DebugPageError {
            tmp_dir: tmp_dir.to_path_buf(),
            page_idx,
        })
}
