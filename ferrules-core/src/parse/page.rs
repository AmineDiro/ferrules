use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tokio::task::JoinSet;

use image::DynamicImage;
use tracing::instrument;

use crate::{
    draw::{draw_blocks, draw_layout_bboxes, draw_text_lines},
    entities::{Element, ElementType, Line, PDFPath, PageID, StructuredPage},
    error::FerrulesError,
    layout::{
        model::LayoutBBox, Metadata, ParseLayoutQueue, ParseLayoutRequest, ParseLayoutResponse,
    },
    metrics::{OCRMetrics, PageMetrics, StepMetrics},
    ocr::parse_image_ocr,
    parse::table::ParseTableQueue,
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
async fn parse_page_text(
    native_text_lines: Vec<Line>,
    page_layout: &[LayoutBBox],
    page_image: Arc<DynamicImage>,
    downscale_factor: f32,
) -> Result<(Vec<Line>, Option<StepMetrics>, bool), FerrulesError> {
    let text_layout_box: Vec<&LayoutBBox> =
        page_layout.iter().filter(|b| b.is_text_block()).collect();
    let need_ocr = page_needs_ocr(&text_layout_box, &native_text_lines);

    let (ocr_result, ocr_metrics) = if need_ocr {
        let page_image_clone = Arc::clone(&page_image);
        let start_wait = Instant::now();
        let _permit = crate::ocr::OCR_SEMAPHORE.acquire().await.unwrap();
        let wait_duration = start_wait.elapsed();

        let start_ocr = Instant::now();
        // We pass None for debug_dir for now
        let res = parse_image_ocr(&page_image_clone, None, downscale_factor).await;

        let ocr_duration = start_ocr.elapsed();
        drop(_permit);
        tracing::debug!(
            "OCR semaphore wait: {}ms, OCR execution: {}ms",
            wait_duration.as_millis(),
            ocr_duration.as_millis()
        );
        match res {
            Ok((lines, mut metrics)) => {
                metrics.idle_time_ms = wait_duration.as_millis();
                (Some(lines), Some(metrics))
            }
            Err(_) => (None, None),
        }
    } else {
        (None, None)
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
    Ok((lines, ocr_metrics, need_ocr))
}

#[instrument(
    skip_all,
    fields(
        layout_queue_time_ms,
        layout_parse_duration_ms,
        layout_parse_duration_ms,
        parse_native_duration_ms,
        table_queue_time_ms,
        table_parse_duration_ms,
    )
)]
pub async fn parse_page_full(
    parse_native_result: ParseNativePageResult,
    debug_dir: Option<PathBuf>,
    layout_queue: ParseLayoutQueue,
    table_queue: ParseTableQueue,
) -> Result<StructuredPage, FerrulesError> {
    let start_time = Instant::now();
    let span = tracing::Span::current();
    let ParseNativePageResult {
        page_id,
        text_lines,
        paths,
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
        page_id: _,
        layout_bbox: page_layout,
        step_metrics: layout_step_metrics,
    } = layout_rx
        .await
        // TODO: better unwrapping
        .map_err(|_| FerrulesError::LayoutParsingError)?
        .map_err(|_| FerrulesError::LayoutParsingError)?;

    let native_lines_captured = text_lines.clone();
    let (text_lines_processed, ocr_step_metrics_inner, need_ocr) = parse_page_text(
        text_lines,
        &page_layout,
        Arc::clone(&page_image),
        downscale_factor,
    )
    .await?;

    let ocr_step_metrics = ocr_step_metrics_inner.map(|m| OCRMetrics {
        step_metrics: m,
        lines_count: text_lines_processed.len(), // Approximate lines count from OCR result
    });

    // Merging elements with layout
    let mut elements = build_page_elements(&page_layout, &text_lines_processed, page_id)?;
    let text_lines_arc = Arc::new(text_lines_processed.clone());
    let paths_arc = Arc::new(paths);

    // Table parsing
    let mut set = JoinSet::new();
    let mut total_table_parse_duration = 0;
    let mut total_table_queue_time = 0;
    for (idx, element) in elements.iter().enumerate() {
        if matches!(element.kind, ElementType::Table(_)) {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let req = crate::parse::table::ParseTableRequest {
                page_id,
                page_image: Arc::clone(&page_image),
                lines: Arc::clone(&text_lines_arc),
                paths: Arc::clone(&paths_arc),
                table_bbox: element.bbox.clone(),
                downscale_factor,
                metadata: crate::parse::table::TableMetadata {
                    response_tx: tx,
                    queue_time: Instant::now(),
                },
            };
            table_queue.push(req).await?;
            set.spawn(async move { (idx, rx.await) });
        }
    }

    while let Some(res) = set.join_next().await {
        if let Ok((idx, Ok(Ok(resp)))) = res {
            if let ElementType::Table(ref mut table_opt) = elements[idx].kind {
                *table_opt = Some(resp.table_block);
                total_table_parse_duration += resp.step_metrics.execution_time_ms;
                total_table_queue_time += resp.step_metrics.queue_time_ms;
            }
        }
    }

    let table_step_metrics = StepMetrics {
        queue_time_ms: total_table_queue_time,
        execution_time_ms: total_table_parse_duration,
        idle_time_ms: 0, // Should also sum idle time if available, but let's stick to what we have or update loop
    };
    if let Some(tmp_dir) = debug_dir {
        debug_page(
            &tmp_dir,
            page_id,
            &page_image_scale1,
            &text_lines_processed,
            need_ocr,
            &page_layout,
            &elements,
            &paths_arc,
        )?
    };

    let native_step = StepMetrics::new(parse_native_metadata.parse_native_duration_ms);

    let page_metrics = PageMetrics {
        page_id,
        total_duration_ms: start_time.elapsed().as_millis(),
        native_step,
        layout_step: layout_step_metrics,
        table_step: table_step_metrics,
        ocr_step: ocr_step_metrics,
    };

    page_metrics.record();
    page_metrics.record_span(&span);

    let structured_page = StructuredPage {
        id: page_id,
        width: page_bbox.width(),
        height: page_bbox.height(),
        image: page_image_scale1,
        elements,
        paths: paths_arc.as_ref().clone(),
        need_ocr,
        native_lines: native_lines_captured,
        layout: page_layout,
        ocr_lines: if need_ocr {
            text_lines_processed.clone()
        } else {
            vec![]
        },
        metrics: page_metrics,
    };

    Ok(structured_page)
}

#[allow(clippy::too_many_arguments, clippy::result_large_err)]
fn debug_page(
    tmp_dir: &Path,
    page_idx: PageID,
    page_image: &DynamicImage,
    text_lines: &[Line],
    need_ocr: bool,
    page_layout: &[LayoutBBox],
    elements: &[Element],
    paths: &[PDFPath],
) -> Result<(), FerrulesError> {
    let images_dir = tmp_dir.join("images");
    let blocks_dir = tmp_dir.join("blocks");
    let _ = std::fs::create_dir_all(&images_dir);
    let _ = std::fs::create_dir_all(&blocks_dir);

    let output_file = images_dir.join(format!("page_{}.png", page_idx));
    let final_output_file = blocks_dir.join(format!("page_blocks_{}.png", page_idx));
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
    let final_img_buffer =
        draw_blocks(&blocks, page_image).map_err(|_| FerrulesError::DebugPageError {
            tmp_dir: tmp_dir.to_path_buf(),
            page_idx,
        })?;

    // Draw paths on final image for debugging
    let dynamic_final_img = image::DynamicImage::ImageRgba8(final_img_buffer);
    let final_img_with_paths =
        crate::draw::draw_paths(paths, &dynamic_final_img).map_err(|_| {
            FerrulesError::DebugPageError {
                tmp_dir: tmp_dir.to_path_buf(),
                page_idx,
            }
        })?;

    out_img
        .save(output_file)
        .map_err(|_| FerrulesError::DebugPageError {
            tmp_dir: tmp_dir.to_path_buf(),
            page_idx,
        })?;

    final_img_with_paths
        .save(final_output_file)
        .map_err(|_| FerrulesError::DebugPageError {
            tmp_dir: tmp_dir.to_path_buf(),
            page_idx,
        })
}
