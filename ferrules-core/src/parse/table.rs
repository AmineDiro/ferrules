use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array4, Axis};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, TensorRTExecutionProvider,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{Instrument, Span};

use crate::blocks::{TableAlgorithm, TableBlock};
use crate::entities::{BBox, PDFPath, PageID};
use crate::error::FerrulesError;
use crate::layout::model::LayoutBBox;
use anyhow::Result;

#[derive(Debug)]
pub struct TableMetadata {
    pub(crate) response_tx: oneshot::Sender<Result<ParseTableResponse>>,
    pub(crate) queue_time: Instant,
}

#[derive(Debug)]
pub(crate) struct ParseTableRequest {
    pub(crate) page_id: PageID,
    pub(crate) page_image: Arc<DynamicImage>,
    pub(crate) lines: Arc<Vec<crate::entities::Line>>,
    pub(crate) paths: Arc<Vec<crate::entities::PDFPath>>,
    pub(crate) table_bbox: BBox,
    pub(crate) downscale_factor: f32,
    pub(crate) metadata: TableMetadata,
}

#[derive(Debug)]
pub(crate) struct ParseTableResponse {
    pub(crate) page_id: PageID,
    pub(crate) table_block: TableBlock,
    pub(crate) table_parse_duration_ms: u128,
}

#[derive(Debug, Clone)]
pub struct ParseTableQueue {
    queue: mpsc::Sender<(ParseTableRequest, Span)>,
}

impl ParseTableQueue {
    pub fn new(table_parser: Arc<TableParser>) -> Self {
        let (queue_sender, queue_receiver) = mpsc::channel(16); // Buffer size

        tokio::task::spawn(start_table_parser(table_parser, queue_receiver));
        Self {
            queue: queue_sender,
        }
    }

    pub(crate) async fn push(&self, req: ParseTableRequest) -> Result<(), FerrulesError> {
        let span = Span::current();
        self.queue
            .send((req, span))
            .await
            .map_err(|_| FerrulesError::LayoutParsingError) // Reusing error or add specific
    }
}

async fn start_table_parser(
    table_parser: Arc<TableParser>,
    mut input_rx: mpsc::Receiver<(ParseTableRequest, Span)>,
) {
    let s = Arc::new(Semaphore::new(4)); // Limit concurrent table workers
    while let Some((req, span)) = input_rx.recv().await {
        let _guard = span.enter();
        tokio::spawn(handle_table_request(s.clone(), table_parser.clone(), req).in_current_span());
    }
}

async fn handle_table_request(
    s: Arc<Semaphore>,
    _parser: Arc<TableParser>,
    req: ParseTableRequest,
) {
    let _permit = s.acquire().await.unwrap();

    let ParseTableRequest {
        page_id,
        page_image,
        lines,
        paths,
        table_bbox,
        downscale_factor,
        metadata,
    } = req;

    let parser = _parser.clone();
    let lines = lines.clone();
    let paths = paths.clone();
    let page_image = page_image.clone();
    let table_bbox = table_bbox.clone();

    let start = Instant::now();
    let table_result = tokio::task::spawn_blocking(move || {
        parser.parse(
            page_id,
            &lines,
            &paths,
            &table_bbox,
            &page_image,
            downscale_factor,
        )
    })
    .await;

    // Handle JoinError from spawn_blocking
    let table_result = match table_result {
        Ok(res) => res,
        Err(_e) => return, // Task panicked or cancelled
    };

    let duration = start.elapsed().as_millis();
    drop(_permit);

    let response = table_result.map(|t| ParseTableResponse {
        page_id,
        table_block: t,
        table_parse_duration_ms: duration,
    });

    let _ = metadata.response_tx.send(response);
}

pub const TABLE_MODEL_BYTES: &[u8] =
    include_bytes!("../../../models/table-transformer-structure-recognition_fp16.onnx");

pub struct TableParser {
    transformer: Option<TableTransformer>,
    table_id_counter: AtomicUsize,
}

pub struct TableTransformer {
    session: Session,
    _output_names: Vec<String>,
}

impl TableTransformer {
    const INPUT_SIZE: [usize; 4] = [1, 3, 1000, 1000];
    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    pub fn new(config: &crate::layout::model::ORTConfig) -> Result<Self> {
        let mut execution_providers = Vec::new();

        // Providers
        for provider in &config.execution_providers {
            match provider {
                crate::layout::model::OrtExecutionProvider::Trt(device_id) => {
                    execution_providers.push(
                        TensorRTExecutionProvider::default()
                            .with_device_id(*device_id)
                            .build(),
                    );
                }
                crate::layout::model::OrtExecutionProvider::CUDA(device_id) => {
                    execution_providers.push(
                        CUDAExecutionProvider::default()
                            .with_device_id(*device_id)
                            .build(),
                    );
                }
                crate::layout::model::OrtExecutionProvider::CoreML { ane_only } => {
                    let provider = CoreMLExecutionProvider::default();
                    let provider = if *ane_only {
                        provider.with_ane_only().build()
                    } else {
                        provider.build()
                    };
                    execution_providers.push(provider)
                }
                crate::layout::model::OrtExecutionProvider::CPU => {
                    execution_providers.push(CPUExecutionProvider::default().build());
                }
            }
        }

        let opt_lvl = match config.opt_level {
            Some(crate::layout::model::ORTGraphOptimizationLevel::Level1) => {
                GraphOptimizationLevel::Level1
            }
            Some(crate::layout::model::ORTGraphOptimizationLevel::Level2) => {
                GraphOptimizationLevel::Level2
            }
            Some(crate::layout::model::ORTGraphOptimizationLevel::Level3) => {
                GraphOptimizationLevel::Level3
            }
            None => GraphOptimizationLevel::Disable,
        };

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(opt_lvl)?
            .with_intra_threads(config.intra_threads)?
            .with_inter_threads(config.inter_threads)?
            .commit_from_memory(TABLE_MODEL_BYTES)?;

        let output_names = session.outputs.iter().map(|o| o.name.clone()).collect();

        Ok(Self {
            session,
            _output_names: output_names,
        })
    }

    pub fn preprocess(&self, img: &DynamicImage) -> Array4<f32> {
        let resized = img.resize_exact(
            Self::INPUT_SIZE[2] as u32,
            Self::INPUT_SIZE[3] as u32,
            FilterType::Triangle,
        );
        let mut input = Array4::zeros(Self::INPUT_SIZE);

        for (x, y, pixel) in resized.pixels() {
            let [r, g, b, _] = pixel.0;
            input[[0, 0, y as usize, x as usize]] =
                (r as f32 / 255.0 - Self::IMAGENET_MEAN[0]) / Self::IMAGENET_STD[0];
            input[[0, 1, y as usize, x as usize]] =
                (g as f32 / 255.0 - Self::IMAGENET_MEAN[1]) / Self::IMAGENET_STD[1];
            input[[0, 2, y as usize, x as usize]] =
                (b as f32 / 255.0 - Self::IMAGENET_MEAN[2]) / Self::IMAGENET_STD[2];
        }

        input
    }

    pub fn run(&self, input: Array4<f32>) -> Result<ort::session::SessionOutputs<'_, '_>> {
        let outputs = self.session.run(ort::inputs![input]?)?;
        Ok(outputs)
    }

    /// Decode the DETR-style output from the Table Transformer.
    /// Boxes are [center_x, center_y, width, height] normalized.
    pub fn postprocess(
        &self,
        outputs: &ort::session::SessionOutputs,
        orig_width: u32,
        orig_height: u32,
    ) -> Result<Vec<LayoutBBox>> {
        let logits = outputs["logits"].try_extract_tensor::<f32>()?;
        let boxes = outputs["pred_boxes"].try_extract_tensor::<f32>()?;

        // logits: [1, 125, 7] (Structure Recognition has 6 classes + 1 Background)
        // boxes: [1, 125, 4]

        let mut results = Vec::new();

        // Structure Recognition Labels:
        // 0: table, 1: column, 2: row, 3: column header, 4: projected row header, 5: spanning cell
        let labels = [
            "table",
            "column",
            "row",
            "column_header",
            "projected_row_header",
            "spanning_cell",
        ];

        let logits = logits.index_axis(Axis(0), 0);
        let boxes = boxes.index_axis(Axis(0), 0);

        for i in 0..125 {
            let logit = logits.index_axis(Axis(0), i);
            let box_coords = boxes.index_axis(Axis(0), i);

            // Find best class
            let (max_idx, &max_val) = logit
                .iter()
                .enumerate()
                .take(6) // Only first 6 are valid classes
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            // Background class is 6
            let bg_val = logit[6];
            if max_val < bg_val || max_val < 0.5 {
                continue;
            }

            let cx = box_coords[0] * orig_width as f32;
            let cy = box_coords[1] * orig_height as f32;
            let w = box_coords[2] * orig_width as f32;
            let h = box_coords[3] * orig_height as f32;

            results.push(LayoutBBox {
                id: i as i32,
                bbox: BBox {
                    x0: cx - w / 2.0,
                    y0: cy - h / 2.0,
                    x1: cx + w / 2.0,
                    y1: cy + h / 2.0,
                },
                label: labels[max_idx],
                proba: 1.0 / (1.0 + (-max_val).exp()),
            });
        }

        Ok(results)
    }
}

impl TableParser {
    const ROW_OVERLAP_THRESHOLD: f32 = 0.5;

    pub fn new(transformer: Option<TableTransformer>) -> Self {
        Self {
            transformer,
            table_id_counter: AtomicUsize::new(0),
        }
    }

    /// Main entry point to parse a table within a given bounding box on a page.
    pub fn parse(
        &self,
        _page_id: PageID,
        lines: &[crate::entities::Line],
        paths: &[PDFPath],
        table_bbox: &BBox,
        page_image: &DynamicImage,
        downscale_factor: f32,
    ) -> Result<TableBlock> {
        // Decide between Lattice and Stream
        // For now, let's try Lattice if we have paths
        if !paths.is_empty() {
            tracing::debug!(
                "Page {} - BBox {:?} - {} paths. Trying Lattice...",
                _page_id,
                table_bbox,
                paths.len()
            );
            if let Some(mut table) = self.parse_lattice(lines, paths, table_bbox) {
                table.algorithm = TableAlgorithm::Lattice;
                tracing::debug!("Page {} - Lattice successful.", _page_id);
                return Ok(table);
            }
            tracing::debug!("Page {} - Lattice failed.", _page_id);
        } else {
            tracing::debug!("Page {} has no paths. Skipping Lattice.", _page_id);
        }

        let mut table = self.parse_stream(lines, table_bbox)?;
        table.algorithm = TableAlgorithm::Stream;

        let cell_count: usize = table.rows.iter().map(|r| r.cells.len()).sum();
        let row_count = table.rows.len();
        let avg_cells_per_row = if row_count > 0 {
            cell_count as f32 / row_count as f32
        } else {
            0.0
        };

        // Heuristics to trigger Vision fallback:
        // 1. Stream found very few cells in a large area.
        // 2. Stream found no rows.
        // 3. For borderless tables, if the area is large, evaluate Vision as it might be more structured.
        let is_suspicious =
            (cell_count <= 2 && table_bbox.width() > 50.0 && table_bbox.height() > 30.0)
                || (row_count <= 1 && table_bbox.height() > 80.0)
                || (avg_cells_per_row < 1.5 && table_bbox.width() > 150.0);

        if table.rows.is_empty() || is_suspicious || table_bbox.area() > 5000.0 {
            tracing::debug!(
                "Page {} - Stream found {} cells ({} rows). Trying Vision comparison (Area {:?})...",
                _page_id,
                cell_count,
                row_count,
                table_bbox.area()
            );

            if let Ok(vision_table) =
                self.parse_vision(page_image, lines, table_bbox, downscale_factor)
            {
                let vision_cell_count: usize =
                    vision_table.rows.iter().map(|r| r.cells.len()).sum();

                // Pick vision if it found more cells OR if stream was highly suspicious
                if vision_cell_count > cell_count
                    || (cell_count == 0 && !vision_table.rows.is_empty())
                {
                    tracing::debug!(
                        "Page {} - Vision ({} cells) preferred over Stream ({} cells).",
                        _page_id,
                        vision_cell_count,
                        cell_count
                    );
                    return Ok(vision_table);
                }
            }
        }

        if table.rows.is_empty() {
            // Keep as Stream (Unknown) if both failed
            return Ok(table);
        }

        tracing::debug!(
            "Page {} - Stream successful ({} cells).",
            _page_id,
            cell_count
        );
        Ok(table)
    }

    fn parse_lattice(
        &self,
        lines: &[crate::entities::Line],
        paths: &[crate::entities::PDFPath],
        table_bbox: &BBox,
    ) -> Option<TableBlock> {
        let padding = 5.0;
        let mut h_lines = Vec::new();
        let mut v_lines = Vec::new();

        for path in paths {
            for segment in &path.segments {
                match segment {
                    crate::entities::Segment::Line { start, end } => {
                        let (x1, y1) = *start;
                        let (x2, y2) = *end;

                        // Horizontal line
                        if (y1 - y2).abs() < 1.0 {
                            let y = (y1 + y2) / 2.0;
                            if y >= table_bbox.y0 - padding && y <= table_bbox.y1 + padding {
                                let x_min = x1.min(x2);
                                let x_max = x1.max(x2);
                                if x_min < table_bbox.x1 + padding
                                    && x_max > table_bbox.x0 - padding
                                {
                                    h_lines.push((y, x_min, x_max));
                                }
                            }
                        }
                        // Vertical line
                        else if (x1 - x2).abs() < 1.0 {
                            let x = (x1 + x2) / 2.0;
                            if x >= table_bbox.x0 - padding && x <= table_bbox.x1 + padding {
                                let y_min = y1.min(y2);
                                let y_max = y1.max(y2);
                                if y_min < table_bbox.y1 + padding
                                    && y_max > table_bbox.y0 - padding
                                {
                                    v_lines.push((x, y_min, y_max));
                                }
                            }
                        }
                    }
                    crate::entities::Segment::Rect { bbox } => {
                        if table_bbox.intersection(bbox) > 0.0 {
                            h_lines.push((bbox.y0, bbox.x0, bbox.x1));
                            h_lines.push((bbox.y1, bbox.x0, bbox.x1));
                            v_lines.push((bbox.x0, bbox.y0, bbox.y1));
                            v_lines.push((bbox.x1, bbox.y0, bbox.y1));
                        }
                    }
                }
            }
        }

        tracing::debug!(
            "Lattice - BBox {:?} - segments: H={}, V={}",
            table_bbox,
            h_lines.len(),
            v_lines.len()
        );

        if h_lines.is_empty() || v_lines.is_empty() {
            return None;
        }

        // Simple clustering
        h_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        v_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut unique_h = Vec::new();
        if !h_lines.is_empty() {
            let mut curr = h_lines[0];
            for next in h_lines.iter().skip(1) {
                if (next.0 - curr.0).abs() < 2.0 {
                    curr.1 = curr.1.min(next.1);
                    curr.2 = curr.2.max(next.2);
                } else {
                    unique_h.push(curr);
                    curr = *next;
                }
            }
            unique_h.push(curr);
        }

        let mut unique_v = Vec::new();
        if !v_lines.is_empty() {
            let mut curr = v_lines[0];
            for next in v_lines.iter().skip(1) {
                if (next.0 - curr.0).abs() < 2.0 {
                    curr.1 = curr.1.min(next.1);
                    curr.2 = curr.2.max(next.2);
                } else {
                    unique_v.push(curr);
                    curr = *next;
                }
            }
            unique_v.push(curr);
        }

        let h_coords: Vec<f32> = unique_h.iter().map(|l| l.0).collect();
        let v_coords: Vec<f32> = unique_v.iter().map(|l| l.0).collect();

        if h_coords.len() < 2 || v_coords.len() < 2 {
            tracing::debug!(
                "Lattice failed: insufficient grid lines. H={:?}, V={:?}",
                h_coords,
                v_coords
            );
            return None;
        }
        tracing::debug!("Lattice Grid: H={:?}, V={:?}", h_coords, v_coords);

        // Pre-filter lines that intersect the table
        let table_lines: Vec<_> = lines
            .iter()
            .filter(|l| table_bbox.intersection(&l.bbox) > 0.0)
            .collect();

        let mut rows = Vec::new();

        // Matrix to track visited grid cells (for spanning)
        let num_rows = h_coords.len() - 1;
        let num_cols = v_coords.len() - 1;
        let mut visited = vec![vec![false; num_cols]; num_rows];

        for i in 0..num_rows {
            let y0 = h_coords[i];
            let y1 = h_coords[i + 1];
            let mut row_cells = Vec::new();

            for j in 0..num_cols {
                if visited[i][j] {
                    continue;
                }

                // Determine horizontal span
                let mut col_span = 1;
                while j + col_span < num_cols {
                    let next_x = v_coords[j + col_span];
                    // Is there a vertical line at next_x covering this row?
                    let has_boundary = v_lines.iter().any(|(sx, ymin, ymax)| {
                        if (sx - next_x).abs() > 1.5 {
                            return false;
                        }
                        let overlap_start = y0.max(*ymin);
                        let overlap_end = y1.min(*ymax);
                        let overlap = (overlap_end - overlap_start).max(0.0);
                        overlap > (y1 - y0) * 0.1
                    });

                    if has_boundary {
                        break;
                    }
                    col_span += 1;
                }

                let row_span = 1;
                let x0 = v_coords[j];
                let x1 = v_coords[j + col_span];
                let cell_bbox = BBox { x0, y0, x1, y1 };

                // Mark visited
                for r in i..i + row_span {
                    for c in j..j + col_span {
                        visited[r][c] = true;
                    }
                }

                // Collect spans that belong to this cell
                let mut cell_spans = Vec::new();
                for line in &table_lines {
                    for span in &line.spans {
                        let (cx, cy) = span.bbox.center();
                        if cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1 {
                            cell_spans.push(span);
                        }
                    }
                }

                // Sort spans by Y then X to maintain reading order
                cell_spans.sort_by(|a, b| {
                    a.bbox
                        .y0
                        .partial_cmp(&b.bbox.y0)
                        .unwrap()
                        .then(a.bbox.x0.partial_cmp(&b.bbox.x0).unwrap())
                });

                let mut cell_text = String::new();
                let mut last_bbox: Option<BBox> = None;

                for span in cell_spans {
                    // Clean text: replace newlines with space
                    let cleaned = span.text.replace('\n', " ");
                    let cleaned_trimmed = cleaned.trim();
                    if cleaned_trimmed.is_empty() {
                        continue;
                    }

                    if let Some(last) = last_bbox {
                        let is_new_line = (span.bbox.y0 - last.y0).abs() > last.height() * 0.3;
                        let gap_x = span.bbox.x0 - last.x1;

                        // Add space if there is a significant vertical or horizontal gap
                        if is_new_line || gap_x > 2.0 {
                            if !cell_text.ends_with(' ') {
                                cell_text.push(' ');
                            }
                        }
                    }

                    cell_text.push_str(cleaned_trimmed);
                    last_bbox = Some(span.bbox.clone());
                }

                row_cells.push(crate::blocks::TableCell {
                    content: vec![],
                    text: cell_text,
                    row_span: row_span as u8,
                    col_span: col_span as u8,
                    bbox: cell_bbox,
                });
            }

            if !row_cells.is_empty() {
                let mut row_bbox = row_cells[0].bbox.clone();
                for cell in &row_cells[1..] {
                    row_bbox.merge(&cell.bbox);
                }
                rows.push(crate::blocks::TableRow {
                    cells: row_cells,
                    is_header: false,
                    bbox: row_bbox,
                });
            }
        }

        if rows.is_empty() {
            return None;
        }

        let table_id = self.table_id_counter.fetch_add(1, Ordering::SeqCst);
        Some(TableBlock {
            id: table_id,
            caption: None,
            rows,
            has_borders: true,
            algorithm: TableAlgorithm::Lattice,
        })
    }

    fn parse_vision(
        &self,
        image: &DynamicImage,
        lines: &[crate::entities::Line],
        table_bbox: &BBox,
        downscale_factor: f32,
    ) -> Result<TableBlock> {
        let transformer = match &self.transformer {
            Some(t) => t,
            None => return Ok(TableBlock::default()),
        };

        // 1. Crop image to table_bbox (in image coordinates)
        let scale = 1.0 / downscale_factor;
        let x0 = (table_bbox.x0 * scale) as u32;
        let y0 = (table_bbox.y0 * scale) as u32;
        // Ensure we don't go out of bounds
        let x0 = x0.min(image.width());
        let y0 = y0.min(image.height());

        // Calculate width/height in image coordinates
        let w_img = ((table_bbox.width() * scale) as u32).max(1);
        let h_img = ((table_bbox.height() * scale) as u32).max(1);

        let w = w_img.min(image.width() - x0).max(1);
        let h = h_img.min(image.height() - y0).max(1);

        let crop = image.crop_imm(x0, y0, w, h);

        // 2. Preprocess
        let input = transformer.preprocess(&crop);

        // 3. Run Inference
        let outputs = transformer.run(input)?;

        // 4. Postprocess
        let detections = transformer.postprocess(&outputs, w, h)?;

        // 5. Map detections to Table structure
        // Simple mapping: find all 'row' and 'column' labels
        let mut rows = detections
            .iter()
            .filter(|d| d.label == "row")
            .collect::<Vec<_>>();
        let mut cols = detections
            .iter()
            .filter(|d| d.label == "column")
            .collect::<Vec<_>>();

        rows.sort_by(|a, b| a.bbox.y0.partial_cmp(&b.bbox.y0).unwrap());
        cols.sort_by(|a, b| a.bbox.x0.partial_cmp(&b.bbox.x0).unwrap());

        let mut table_rows = Vec::new();
        for row_det in &rows {
            let mut cells = Vec::new();
            for col_det in &cols {
                let col_x0_pdf = (col_det.bbox.x0 + x0 as f32) * downscale_factor;
                let col_x1_pdf = (col_det.bbox.x1 + x0 as f32) * downscale_factor;
                let row_y0_pdf = (row_det.bbox.y0 + y0 as f32) * downscale_factor;
                let row_y1_pdf = (row_det.bbox.y1 + y0 as f32) * downscale_factor;

                let cell_bbox = BBox {
                    x0: col_x0_pdf.max(table_bbox.x0),
                    y0: row_y0_pdf.max(table_bbox.y0),
                    x1: col_x1_pdf.min(table_bbox.x1),
                    y1: row_y1_pdf.min(table_bbox.y1),
                };

                // Filter lines that fall into this cell
                let cell_text = lines
                    .iter()
                    .filter(|l| cell_bbox.contains(&l.bbox))
                    .map(|l| l.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                cells.push(crate::blocks::TableCell {
                    text: cell_text,
                    bbox: cell_bbox,
                    col_span: 1,
                    row_span: 1,
                    content: Vec::new(),
                });
            }
            table_rows.push(crate::blocks::TableRow {
                cells,
                bbox: BBox {
                    x0: (row_det.bbox.x0 + x0 as f32) * downscale_factor,
                    y0: (row_det.bbox.y0 + y0 as f32) * downscale_factor,
                    x1: (row_det.bbox.x1 + x0 as f32) * downscale_factor,
                    y1: (row_det.bbox.y1 + y0 as f32) * downscale_factor,
                },
                is_header: false,
            });
        }

        let table_id = self.table_id_counter.fetch_add(1, Ordering::SeqCst);
        Ok(TableBlock {
            id: table_id,
            caption: None,
            rows: table_rows,
            has_borders: true,
            algorithm: TableAlgorithm::Vision,
        })
    }

    fn parse_stream(
        &self,
        lines: &[crate::entities::Line],
        table_bbox: &BBox,
    ) -> Result<TableBlock> {
        // 1. Filter lines within table_bbox
        let mut table_lines: Vec<_> = lines
            .iter()
            .filter(|l| table_bbox.contains(&l.bbox))
            .collect();

        // 2. Sort lines by Y (vertical)
        table_lines.sort_by(|a, b| a.bbox.y0.partial_cmp(&b.bbox.y0).unwrap());

        // 3. Group lines into rows based on Y overlap
        let mut rows = Vec::new();
        if table_lines.is_empty() {
            let table_id = self.table_id_counter.fetch_add(1, Ordering::SeqCst);
            return Ok(TableBlock {
                id: table_id,
                caption: None,
                rows: vec![],
                has_borders: false,
                algorithm: TableAlgorithm::Unknown,
            });
        }

        let mut current_row_lines = vec![table_lines[0]];
        for line in table_lines.iter().skip(1) {
            let last_line = current_row_lines.last().unwrap();
            // NOTE: If the next line significantly overlaps vertically or is very close, it's the same row
            if line.bbox.y0
                < last_line.bbox.y1 - last_line.bbox.height() * Self::ROW_OVERLAP_THRESHOLD
            {
                current_row_lines.push(line);
            } else {
                rows.push(self.process_row_lines(&current_row_lines));
                current_row_lines = vec![line];
            }
        }
        rows.push(self.process_row_lines(&current_row_lines));

        let table_id = self.table_id_counter.fetch_add(1, Ordering::SeqCst);
        Ok(TableBlock {
            id: table_id,
            caption: None,
            rows,
            has_borders: false,
            algorithm: TableAlgorithm::Stream,
        })
    }

    fn process_row_lines(&self, row_lines: &[&crate::entities::Line]) -> crate::blocks::TableRow {
        if row_lines.is_empty() {
            return crate::blocks::TableRow::default();
        }

        let mut sorted_lines = row_lines.to_vec();
        sorted_lines.sort_by(|a, b| a.bbox.x0.partial_cmp(&b.bbox.x0).unwrap());

        let mut cells = Vec::new();
        let mut current_cell_text = sorted_lines[0].text.clone();
        let mut current_cell_bbox = sorted_lines[0].bbox.clone();

        for line in sorted_lines.iter().skip(1) {
            // Threshold for horizontal gap between words/cells in a table
            // Usually tables have larger gaps than normal text
            let horizontal_gap = line.bbox.x0 - current_cell_bbox.x1;
            if horizontal_gap < 10.0 {
                current_cell_text.push(' ');
                current_cell_text.push_str(&line.text);
                current_cell_bbox.merge(&line.bbox);
            } else {
                cells.push(crate::blocks::TableCell {
                    content: vec![],
                    text: current_cell_text.trim().to_string(),
                    bbox: current_cell_bbox,
                    col_span: 1,
                    row_span: 1,
                });
                current_cell_text = line.text.clone();
                current_cell_bbox = line.bbox.clone();
            }
        }

        cells.push(crate::blocks::TableCell {
            content: vec![],
            text: current_cell_text.trim().to_string(),
            bbox: current_cell_bbox,
            col_span: 1,
            row_span: 1,
        });

        let mut row_bbox = cells[0].bbox.clone();
        for cell in &cells[1..] {
            row_bbox.merge(&cell.bbox);
        }

        crate::blocks::TableRow {
            cells,
            bbox: row_bbox,
            is_header: false,
        }
    }
}
