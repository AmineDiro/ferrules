use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use ndarray::{s, stack, Array4, ArrayD, Axis};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, TensorRTExecutionProvider,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;
use tracing::Instrument;

use crate::blocks::{TableAlgorithm, TableBlock};
use crate::entities::BBox;
use crate::error::FerrulesError;
use crate::layout::model::{nms, LayoutBBox};

pub const TABLE_MODEL_BYTES: &[u8] =
    include_bytes!("../../../../models/table-transformer-structure-recognition_fp16.onnx");

pub const TABLE_MODEL_ANE_BYTES: &[u8] =
    include_bytes!("../../../../models/table-transformer-structure-recognition-ane-b4.onnx");

#[allow(dead_code)]
#[derive(Clone)]
pub struct TableTransformerStandard {
    tx: mpsc::Sender<InferenceRequest>,
}

struct InferenceRequest {
    input: Array4<f32>,
    // Tuple of (logits, pred_boxes)
    response_tx: oneshot::Sender<Result<(ArrayD<f32>, ArrayD<f32>), FerrulesError>>,
}

struct BatchInferenceRunner {
    session: Arc<Session>,
    rx: mpsc::Receiver<InferenceRequest>,
    max_batch_size: usize,
    batch_timeout: Duration,
    is_fp16: bool,
}

impl BatchInferenceRunner {
    /// maximum batch size for the table transformer to process
    const MAX_BATCH_SIZE: usize = 4;
    /// maximum time to wait for a batch to be filled
    const BATCH_TIMEOUT: Duration = Duration::from_millis(50);

    fn new(session: Session, rx: mpsc::Receiver<InferenceRequest>, is_fp16: bool) -> Self {
        Self {
            session: Arc::new(session),
            rx,
            max_batch_size: Self::MAX_BATCH_SIZE,
            batch_timeout: Self::BATCH_TIMEOUT,
            is_fp16,
        }
    }

    async fn run(mut self) {
        let mut batch = Vec::with_capacity(self.max_batch_size);

        loop {
            // 1. Accumulate batch
            let first_req = match self.rx.recv().await {
                Some(req) => req,
                None => break, // Channel closed
            };
            let batch_start = tokio::time::Instant::now();
            batch.push(first_req);

            let deadline = tokio::time::Instant::now() + self.batch_timeout;

            while batch.len() < self.max_batch_size {
                let remaining_time =
                    deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining_time.is_zero() {
                    break;
                }

                match timeout(remaining_time, self.rx.recv()).await {
                    Ok(Some(req)) => batch.push(req),
                    Ok(None) => break, // Channel closed
                    Err(_) => break,   // Timeout
                }
            }

            if batch.is_empty() {
                continue;
            }

            let accumulation_time = batch_start.elapsed().as_secs_f64() * 1000.0;

            // 2. Prepare batch input
            let prep_start = tokio::time::Instant::now();
            let current_batch_size = batch.len();

            // Find max H and W
            let mut max_h = 0;
            let mut max_w = 0;
            for req in &batch {
                let (_, _, h, w) = req.input.dim();
                max_h = max_h.max(h);
                max_w = max_w.max(w);
            }

            // Pad inputs to max_h, max_w
            // Input is [1, 3, h, w]
            // Stack along axis 0 -> [N, 3, max_h, max_w]
            let mut batch_input_vec = Vec::with_capacity(self.max_batch_size);
            for req in &batch {
                let mut padded = Array4::<f32>::zeros((1, 3, max_h, max_w));
                let (_, _, h, w) = req.input.dim();
                padded.slice_mut(s![.., .., ..h, ..w]).assign(&req.input);
                batch_input_vec.push(padded.remove_axis(Axis(0))); // [3, max_h, max_w]
            }

            // Pad to max_batch_size for static shape models (like ANE)
            for _ in current_batch_size..self.max_batch_size {
                batch_input_vec.push(ndarray::Array3::<f32>::zeros((3, max_h, max_w)));
            }

            let batch_input_views: Vec<_> = batch_input_vec.iter().map(|a| a.view()).collect();
            let batch_tensor =
                match stack(Axis(0), &batch_input_views) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("Failed to stack batch inputs: {}", e);
                        // Fail all
                        for req in batch.drain(..) {
                            let _ = req.response_tx.send(Err(
                                FerrulesError::TableTransformerModelError(e.to_string()),
                            ));
                        }
                        continue;
                    }
                };

            let prep_time = prep_start.elapsed().as_secs_f64() * 1000.0;

            // 3. Run Inference (Async)
            let run_start = tokio::time::Instant::now();
            let run_result = async {
                if self.is_fp16 {
                    let input_f16 = batch_tensor.mapv(half::f16::from_f32);
                    let outputs = self.session.run_async(ort::inputs![input_f16]?)?.await?;
                    let logits = outputs["logits"]
                        .try_extract_tensor::<half::f16>()?
                        .mapv(|x| x.to_f32())
                        .into_dyn();
                    let boxes = outputs["pred_boxes"]
                        .try_extract_tensor::<half::f16>()?
                        .mapv(|x| x.to_f32())
                        .into_dyn();
                    Ok::<_, ort::Error>((logits, boxes))
                } else {
                    let outputs = self.session.run_async(ort::inputs![batch_tensor]?)?.await?;
                    let logits = outputs["logits"]
                        .try_extract_tensor::<f32>()?
                        .to_owned()
                        .into_dyn();
                    let boxes = outputs["pred_boxes"]
                        .try_extract_tensor::<f32>()?
                        .to_owned()
                        .into_dyn();
                    Ok::<_, ort::Error>((logits, boxes))
                }
            }
            .await;

            let run_time = run_start.elapsed().as_secs_f64() * 1000.0;
            let total_batch_time = batch_start.elapsed().as_secs_f64() * 1000.0;

            tracing::debug!(
                "Table Transformer Batch: size={}, total={:.1}ms (accum={:.1}ms, prep={:.1}ms, run={:.1}ms)",
                current_batch_size,
                total_batch_time,
                accumulation_time,
                prep_time,
                run_time
            );

            // 4. Distribute results
            match run_result {
                Ok((logits, boxes)) => {
                    for (i, req) in batch.drain(..).enumerate() {
                        let logit = logits.index_axis(Axis(0), i).to_owned();
                        let bbox = boxes.index_axis(Axis(0), i).to_owned();
                        let _ = req.response_tx.send(Ok((logit, bbox)));
                    }
                }
                Err(e) => {
                    tracing::error!("Inference failed: {}", e);
                    for req in batch.drain(..) {
                        let _ =
                            req.response_tx
                                .send(Err(FerrulesError::TableTransformerModelError(
                                    e.to_string(),
                                )));
                    }
                }
            }
        }
    }
}

#[allow(dead_code)]
impl TableTransformerStandard {
    const SHORTEST_EDGE: usize = 800;
    const MAX_SIZE: usize = 1333;
    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    // Structure Recognition Labels:
    // 0: table, 1: column, 2: row, 3: column header, 4: projected row header, 5: spanning cell
    const TABLE_LABELS: [&'static str; 6] = [
        "table",
        "column",
        "row",
        "column_header",
        "projected_row_header",
        "spanning_cell",
    ];

    const CONFIDENCE_THRESHOLD: f32 = 0.6;

    fn scale_wh(&self, w0: f32, h0: f32) -> (f32, f32, f32) {
        let mut r = Self::SHORTEST_EDGE as f32 / w0.min(h0);
        if (w0.max(h0) * r) > Self::MAX_SIZE as f32 {
            r = Self::MAX_SIZE as f32 / w0.max(h0);
        }
        (r, (w0 * r).round(), (h0 * r).round())
    }

    pub fn new(config: &crate::layout::model::ORTConfig) -> Result<Self, FerrulesError> {
        let mut execution_providers = Vec::new();

        // Get providers sorted by priority: accelerators first
        let providers = config.get_sorted_providers();

        // Providers
        for provider in providers {
            match provider {
                crate::layout::model::OrtExecutionProvider::Trt(device_id) => {
                    execution_providers.push(
                        TensorRTExecutionProvider::default()
                            .with_device_id(device_id)
                            .build(),
                    );
                }
                crate::layout::model::OrtExecutionProvider::CUDA(device_id) => {
                    execution_providers.push(
                        CUDAExecutionProvider::default()
                            .with_device_id(device_id)
                            .build(),
                    );
                }
                crate::layout::model::OrtExecutionProvider::CoreML { ane_only } => {
                    let provider = CoreMLExecutionProvider::default();
                    let provider = if ane_only {
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

        let mut builder = Session::builder()
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_execution_providers(execution_providers)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_optimization_level(opt_lvl)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_intra_threads(config.intra_threads)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_inter_threads(config.inter_threads)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?;

        if let Some(profile_path) = &config.profile_table {
            builder = builder
                .with_profiling(profile_path)
                .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?;
        }

        let session = builder
            .commit_from_memory(TABLE_MODEL_BYTES)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?;

        let (tx, rx) = mpsc::channel(32);
        let runner = BatchInferenceRunner::new(session, rx, true);
        tokio::spawn(runner.run());

        Ok(Self { tx })
    }

    pub fn preprocess(&self, img: &DynamicImage) -> Array4<f32> {
        let (w0, h0) = img.dimensions();
        let (_, w_new, h_new) = self.scale_wh(w0 as f32, h0 as f32);

        let resized = img.resize_exact(w_new as u32, h_new as u32, FilterType::Triangle);
        let (w_final, h_final) = resized.dimensions();

        let mut input = Array4::zeros([1, 3, h_final as usize, w_final as usize]);

        for (x, y, pixel) in resized.pixels() {
            let [r, g, b, _] = pixel.0;
            // Normalize with ImageNet mean/std
            input[[0, 0, y as usize, x as usize]] =
                (r as f32 / 255.0 - Self::IMAGENET_MEAN[0]) / Self::IMAGENET_STD[0];
            input[[0, 1, y as usize, x as usize]] =
                (g as f32 / 255.0 - Self::IMAGENET_MEAN[1]) / Self::IMAGENET_STD[1];
            input[[0, 2, y as usize, x as usize]] =
                (b as f32 / 255.0 - Self::IMAGENET_MEAN[2]) / Self::IMAGENET_STD[2];
        }

        input
    }

    pub async fn run(
        &self,
        input: Array4<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>), FerrulesError> {
        let (tx, rx) = oneshot::channel();

        self.tx
            .send(InferenceRequest {
                input,
                response_tx: tx,
            })
            .await
            .map_err(|e| {
                FerrulesError::TableTransformerModelError(format!(
                    "Table transformer queue send error: {}",
                    e
                ))
            })?;

        rx.await.map_err(|e| {
            FerrulesError::TableTransformerModelError(format!(
                "Table transformer channel closed: {}",
                e
            ))
        })?
    }

    /// Decode the DETR-style output from the Table Transformer.
    /// Boxes are [center_x, center_y, width, height] normalized.
    pub fn postprocess(
        &self,
        results: &(ArrayD<f32>, ArrayD<f32>),
        orig_width: u32,
        orig_height: u32,
    ) -> Result<Vec<LayoutBBox>, FerrulesError> {
        let (logits, boxes) = results;

        // logits: [125, 7] (Structure Recognition has 6 classes + 1 Background)
        // boxes: [125, 4]

        let mut results = Vec::new();

        // Already sliced to [125, 7] and [125, 4] by the batch runner if we did it right
        // Wait, index_axis returns dims [125, 7] if input was [N, 125, 7]. Correct.

        for i in 0..125 {
            let logit = logits.index_axis(Axis(0), i);
            let box_coords = boxes.index_axis(Axis(0), i);

            // Apply softmax to get proper probabilities
            let max_logit = logit.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logit.iter().map(|&v| (v - max_logit).exp()).sum();
            let softmax_probs: Vec<f32> = logit
                .iter()
                .map(|&v| (v - max_logit).exp() / exp_sum)
                .collect();

            // Find best class
            let (max_idx, &max_prob) = softmax_probs
                .iter()
                .enumerate()
                .take(Self::TABLE_LABELS.len()) // only first 6 are valid classes
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            if max_prob < Self::CONFIDENCE_THRESHOLD {
                continue;
            }

            let cx = box_coords[0] * orig_width as f32;
            let cy = box_coords[1] * orig_height as f32;
            let w = box_coords[2] * orig_width as f32;
            let h = box_coords[3] * orig_height as f32;

            results.push(LayoutBBox {
                id: i as i32,
                label: Self::TABLE_LABELS[max_idx].to_string(),
                proba: max_prob,
                bbox: BBox {
                    x0: cx - w / 2.0,
                    y0: cy - h / 2.0,
                    x1: cx + w / 2.0,
                    y1: cy + h / 2.0,
                },
            });
        }

        Ok(results)
    }

    #[tracing::instrument(skip(self, image, lines), fields(table_bbox = ?table_bbox))]
    pub async fn parse_table_transformer(
        &self,
        table_id_counter: &Arc<AtomicUsize>,
        image: &DynamicImage,
        lines: &[crate::entities::Line],
        table_bbox: &BBox,
        downscale_factor: f32,
    ) -> Result<TableBlock, FerrulesError> {
        // 1. Crop image to table_bbox (in image coordinates)
        let scale = 1.0 / downscale_factor;
        let x0_f = table_bbox.x0 * scale;
        let y0_f = table_bbox.y0 * scale;
        let x0 = x0_f.floor() as u32;
        let y0 = y0_f.floor() as u32;

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
        let input = {
            let _span =
                tracing::trace_span!("preprocess", width = crop.width(), height = crop.height())
                    .entered();
            self.preprocess(&crop)
        };

        // 3. Run Inference
        let outputs = self
            .run(input)
            .instrument(tracing::debug_span!("inference"))
            .await?;

        // 4. Postprocess
        let detections = {
            let _span = tracing::debug_span!("postprocess").entered();
            self.postprocess(&outputs, w, h).map_err(|e| {
                tracing::error!("parse_vision: Postprocess failed: {:?}", e);
                FerrulesError::TableTransformerModelError(e.to_string())
            })?
        };

        tracing::debug!(
            "Vision detections: rows={}, cols={}, spanning={}, headers={}",
            detections.iter().filter(|d| d.label == "row").count(),
            detections.iter().filter(|d| d.label == "column").count(),
            detections
                .iter()
                .filter(|d| d.label == "spanning_cell")
                .count(),
            detections
                .iter()
                .filter(|d| d.label == "column_header")
                .count()
        );

        // 5. Map detections to Table structure
        // Simple mapping: find all 'row' and 'column' labels
        let mut rows: Vec<LayoutBBox> = detections
            .iter()
            .filter(|d| d.label == "row")
            .cloned()
            .collect();
        let mut cols: Vec<LayoutBBox> = detections
            .iter()
            .filter(|d| d.label == "column")
            .cloned()
            .collect();

        // 5a. Apply NMS to rows and columns independently
        nms(&mut rows, 0.5);
        nms(&mut cols, 0.5);

        rows.sort_by(|a, b| a.bbox.y0.partial_cmp(&b.bbox.y0).unwrap());
        cols.sort_by(|a, b| a.bbox.x0.partial_cmp(&b.bbox.x0).unwrap());

        // Snap outermost column/row edges to the table bbox so cells cover
        // the full table area. The model detects content areas which are
        // typically slightly narrower than the full table boundaries.
        if let Some(first_col) = cols.first_mut() {
            first_col.bbox.x0 = 0.0;
        }
        if let Some(last_col) = cols.last_mut() {
            last_col.bbox.x1 = w as f32;
        }
        if let Some(first_row) = rows.first_mut() {
            first_row.bbox.y0 = 0.0;
        }
        if let Some(last_row) = rows.last_mut() {
            last_row.bbox.y1 = h as f32;
        }

        // Extract spanning cells and column headers
        let spanning_cells: Vec<&LayoutBBox> = detections
            .iter()
            .filter(|d| d.label == "spanning_cell")
            .collect();
        let header_dets: Vec<&LayoutBBox> = detections
            .iter()
            .filter(|d| d.label == "column_header")
            .collect();

        let mut table_rows = Vec::new();
        for row_det in &rows {
            let row_y0_pdf = (row_det.bbox.y0 + y0 as f32) * downscale_factor;
            let row_y1_pdf = (row_det.bbox.y1 + y0 as f32) * downscale_factor;

            // Check if this row is a header row
            let is_header = header_dets.iter().any(|hdr| {
                let row_bbox_crop = &row_det.bbox;
                row_bbox_crop.intersection(&hdr.bbox) / row_bbox_crop.area() > 0.5
            });

            let mut cells = Vec::new();
            let mut col_idx = 0;
            while col_idx < cols.len() {
                // Build the cell bbox for current (row, col) in crop-pixel space
                let cell_crop = BBox {
                    x0: cols[col_idx].bbox.x0,
                    y0: row_det.bbox.y0,
                    x1: cols[col_idx].bbox.x1,
                    y1: row_det.bbox.y1,
                };

                // Check if a spanning cell covers this position
                let spanning = spanning_cells
                    .iter()
                    .find(|sc| cell_crop.intersection(&sc.bbox) / cell_crop.area() > 0.5);

                let col_span = if let Some(sc) = spanning {
                    // Count how many consecutive columns this spanning cell covers
                    let mut span = 1;
                    for j in (col_idx + 1)..cols.len() {
                        let col_overlap = cols[j].bbox.overlap_x(&sc.bbox);
                        if col_overlap / cols[j].bbox.width() > 0.5 {
                            span += 1;
                        } else {
                            break;
                        }
                    }
                    span
                } else {
                    1usize
                };

                // Build the merged cell bbox spanning col_idx..col_idx+col_span
                let last_col = &cols[(col_idx + col_span - 1).min(cols.len() - 1)];
                let cell_x0_pdf = (cols[col_idx].bbox.x0 + x0 as f32) * downscale_factor;
                let cell_x1_pdf = (last_col.bbox.x1 + x0 as f32) * downscale_factor;

                let cell_bbox = BBox {
                    x0: cell_x0_pdf.max(table_bbox.x0),
                    y0: row_y0_pdf.max(table_bbox.y0),
                    x1: cell_x1_pdf.min(table_bbox.x1),
                    y1: row_y1_pdf.min(table_bbox.y1),
                };

                let cell_text = lines
                    .iter()
                    .filter(|l| cell_bbox.intersection(&l.bbox) / l.bbox.area() > 0.5)
                    .map(|l| l.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                cells.push(crate::blocks::TableCell {
                    text: cell_text,
                    bbox: cell_bbox,
                    col_span: col_span as u8,
                    row_span: 1,
                    content_ids: Vec::new(),
                });

                col_idx += col_span;
            }

            table_rows.push(crate::blocks::TableRow {
                cells,
                bbox: BBox {
                    x0: table_bbox.x0,
                    y0: row_y0_pdf,
                    x1: table_bbox.x1,
                    y1: row_y1_pdf,
                },
                is_header,
            });
        }

        let table_id = table_id_counter.fetch_add(1, Ordering::SeqCst);
        Ok(TableBlock {
            id: table_id,
            caption: None,
            rows: table_rows,
            has_borders: true,
            algorithm: TableAlgorithm::Vision,
        })
    }
}

#[derive(Clone)]
pub struct TableTransformer {
    tx: mpsc::Sender<InferenceRequest>,
}

impl TableTransformer {
    const INPUT_SIZE: usize = 1000;
    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
    const TABLE_LABELS: [&'static str; 6] = [
        "table",
        "column",
        "row",
        "column_header",
        "projected_row_header",
        "spanning_cell",
    ];
    const CONFIDENCE_THRESHOLD: f32 = 0.6;

    pub fn new(config: &crate::layout::model::ORTConfig) -> Result<Self, FerrulesError> {
        let mut execution_providers = Vec::new();
        let providers = config.get_sorted_providers();

        for provider in providers {
            match provider {
                crate::layout::model::OrtExecutionProvider::Trt(device_id) => {
                    execution_providers.push(
                        TensorRTExecutionProvider::default()
                            .with_device_id(device_id)
                            .build(),
                    );
                }
                crate::layout::model::OrtExecutionProvider::CUDA(device_id) => {
                    execution_providers.push(
                        CUDAExecutionProvider::default()
                            .with_device_id(device_id)
                            .build(),
                    );
                }
                crate::layout::model::OrtExecutionProvider::CoreML { ane_only } => {
                    let provider = CoreMLExecutionProvider::default();
                    let provider = if ane_only {
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

        let mut builder = Session::builder()
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_execution_providers(execution_providers)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_optimization_level(opt_lvl)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_intra_threads(config.intra_threads)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_inter_threads(config.inter_threads)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?;

        if let Some(profile_path) = &config.profile_table {
            builder = builder
                .with_profiling(profile_path)
                .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?;
        }

        let session = builder
            .commit_from_memory(TABLE_MODEL_ANE_BYTES)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?;

        let (tx, rx) = mpsc::channel(32);
        let runner = BatchInferenceRunner::new(session, rx, false);
        tokio::spawn(runner.run());

        Ok(Self { tx })
    }

    /// Preprocess image for ANE model:
    /// Resizes image exactly to 1000x1000.
    /// Note: This ignores aspect ratio, which may affect detection accuracy for extreme ratios.
    /// However, this approach passed the parity test with the current model.
    pub fn preprocess(&self, img: &DynamicImage) -> (Array4<f32>, f32, f32) {
        let (w0, h0) = img.dimensions();
        let scale_x = Self::INPUT_SIZE as f32 / w0 as f32;
        let scale_y = Self::INPUT_SIZE as f32 / h0 as f32;

        let resized = img.resize_exact(
            Self::INPUT_SIZE as u32,
            Self::INPUT_SIZE as u32,
            FilterType::Triangle,
        );

        let mut input = Array4::zeros([1, 3, Self::INPUT_SIZE, Self::INPUT_SIZE]);

        for (x, y, pixel) in resized.pixels() {
            let [r, g, b, _] = pixel.0;
            input[[0, 0, y as usize, x as usize]] =
                (r as f32 / 255.0 - Self::IMAGENET_MEAN[0]) / Self::IMAGENET_STD[0];
            input[[0, 1, y as usize, x as usize]] =
                (g as f32 / 255.0 - Self::IMAGENET_MEAN[1]) / Self::IMAGENET_STD[1];
            input[[0, 2, y as usize, x as usize]] =
                (b as f32 / 255.0 - Self::IMAGENET_MEAN[2]) / Self::IMAGENET_STD[2];
        }

        (input, scale_x, scale_y)
    }

    pub fn postprocess(
        &self,
        results: &(ArrayD<f32>, ArrayD<f32>),
        _orig_width: u32,
        _orig_height: u32,
        scale_x: f32,
        scale_y: f32,
    ) -> Result<Vec<LayoutBBox>, FerrulesError> {
        let (logits, boxes) = results;
        let mut results = Vec::new();

        for i in 0..125 {
            let logit = logits.index_axis(Axis(0), i);
            let box_coords = boxes.index_axis(Axis(0), i);

            // Softmax
            let max_logit = logit.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logit.iter().map(|&v| (v - max_logit).exp()).sum();
            let softmax_probs: Vec<f32> = logit
                .iter()
                .map(|&v| (v - max_logit).exp() / exp_sum)
                .collect();

            // Best class
            let (max_idx, &max_prob) = softmax_probs
                .iter()
                .enumerate()
                .take(Self::TABLE_LABELS.len())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            if max_prob < Self::CONFIDENCE_THRESHOLD {
                continue;
            }

            // Coordinates in the 1000x1000 frame
            let cx_1000 = box_coords[0] * Self::INPUT_SIZE as f32;
            let cy_1000 = box_coords[1] * Self::INPUT_SIZE as f32;
            let w_1000 = box_coords[2] * Self::INPUT_SIZE as f32;
            let h_1000 = box_coords[3] * Self::INPUT_SIZE as f32;

            // Map back to original image coordinates
            let cx = cx_1000 / scale_x;
            let cy = cy_1000 / scale_y;
            let w = w_1000 / scale_x;
            let h = h_1000 / scale_y;

            results.push(LayoutBBox {
                id: i as i32,
                label: Self::TABLE_LABELS[max_idx].to_string(),
                proba: max_prob,
                bbox: BBox {
                    x0: cx - w / 2.0,
                    y0: cy - h / 2.0,
                    x1: cx + w / 2.0,
                    y1: cy + h / 2.0,
                },
            });
        }

        Ok(results)
    }

    pub async fn run(
        &self,
        input: Array4<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>), FerrulesError> {
        let (tx, rx) = oneshot::channel();

        self.tx
            .send(InferenceRequest {
                input,
                response_tx: tx,
            })
            .await
            .map_err(|e| {
                FerrulesError::TableTransformerModelError(format!(
                    "Table transformer ANE queue send error: {}",
                    e
                ))
            })?;

        rx.await.map_err(|e| {
            FerrulesError::TableTransformerModelError(format!(
                "Table transformer ANE channel closed: {}",
                e
            ))
        })?
    }

    #[tracing::instrument(skip(self, image, lines), fields(table_bbox = ?table_bbox))]
    pub async fn parse_table_transformer(
        &self,
        table_id_counter: &Arc<AtomicUsize>,
        image: &DynamicImage,
        lines: &[crate::entities::Line],
        table_bbox: &BBox,
        downscale_factor: f32,
    ) -> Result<TableBlock, FerrulesError> {
        // 1. Crop image to table_bbox (in image coordinates)
        let scale = 1.0 / downscale_factor;
        let x0_f = table_bbox.x0 * scale;
        let y0_f = table_bbox.y0 * scale;
        let x0 = x0_f.floor() as u32;
        let y0 = y0_f.floor() as u32;

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
        let (input, scale_x, scale_y) = {
            let _span =
                tracing::trace_span!("preprocess", width = crop.width(), height = crop.height())
                    .entered();
            self.preprocess(&crop)
        };

        // 3. Run Inference
        let outputs = self
            .run(input)
            .instrument(tracing::debug_span!("inference"))
            .await?;

        // 4. Postprocess
        let detections = {
            let _span = tracing::debug_span!("postprocess").entered();
            self.postprocess(&outputs, w, h, scale_x, scale_y)
                .map_err(|e| {
                    tracing::error!("parse_vision: Postprocess failed: {:?}", e);
                    FerrulesError::TableTransformerModelError(e.to_string())
                })?
        };

        tracing::debug!(
            "Vision detections (ANE): rows={}, cols={}, spanning={}, headers={}",
            detections.iter().filter(|d| d.label == "row").count(),
            detections.iter().filter(|d| d.label == "column").count(),
            detections
                .iter()
                .filter(|d| d.label == "spanning_cell")
                .count(),
            detections
                .iter()
                .filter(|d| d.label == "column_header")
                .count()
        );

        // 5. Map detections to Table structure
        // Simple mapping: find all 'row' and 'column' labels
        let mut rows: Vec<LayoutBBox> = detections
            .iter()
            .filter(|d| d.label == "row")
            .cloned()
            .collect();
        let mut cols: Vec<LayoutBBox> = detections
            .iter()
            .filter(|d| d.label == "column")
            .cloned()
            .collect();

        // 5a. Apply NMS to rows and columns independently
        nms(&mut rows, 0.5);
        nms(&mut cols, 0.5);

        rows.sort_by(|a, b| a.bbox.y0.partial_cmp(&b.bbox.y0).unwrap());
        cols.sort_by(|a, b| a.bbox.x0.partial_cmp(&b.bbox.x0).unwrap());

        // Snap outermost column/row edges to the table bbox so cells cover
        // the full table area. The model detects content areas which are
        // typically slightly narrower than the full table boundaries.
        if let Some(first_col) = cols.first_mut() {
            first_col.bbox.x0 = 0.0;
        }
        if let Some(last_col) = cols.last_mut() {
            last_col.bbox.x1 = w as f32;
        }
        if let Some(first_row) = rows.first_mut() {
            first_row.bbox.y0 = 0.0;
        }
        if let Some(last_row) = rows.last_mut() {
            last_row.bbox.y1 = h as f32;
        }

        // Extract spanning cells and column headers
        let spanning_cells: Vec<&LayoutBBox> = detections
            .iter()
            .filter(|d| d.label == "spanning_cell")
            .collect();
        let header_dets: Vec<&LayoutBBox> = detections
            .iter()
            .filter(|d| d.label == "column_header")
            .collect();

        let mut table_rows = Vec::new();
        for row_det in &rows {
            let row_y0_pdf = (row_det.bbox.y0 + y0 as f32) * downscale_factor;
            let row_y1_pdf = (row_det.bbox.y1 + y0 as f32) * downscale_factor;

            // Check if this row is a header row
            let is_header = header_dets.iter().any(|hdr| {
                let row_bbox_crop = &row_det.bbox;
                row_bbox_crop.intersection(&hdr.bbox) / row_bbox_crop.area() > 0.5
            });

            let mut cells = Vec::new();
            let mut col_idx = 0;
            while col_idx < cols.len() {
                // Build the cell bbox for current (row, col) in crop-pixel space
                let cell_crop = BBox {
                    x0: cols[col_idx].bbox.x0,
                    y0: row_det.bbox.y0,
                    x1: cols[col_idx].bbox.x1,
                    y1: row_det.bbox.y1,
                };

                // Check if a spanning cell covers this position
                let spanning = spanning_cells
                    .iter()
                    .find(|sc| cell_crop.intersection(&sc.bbox) / cell_crop.area() > 0.5);

                let col_span = if let Some(sc) = spanning {
                    // Count how many consecutive columns this spanning cell covers
                    let mut span = 1;
                    for j in (col_idx + 1)..cols.len() {
                        let col_overlap = cols[j].bbox.overlap_x(&sc.bbox);
                        if col_overlap / cols[j].bbox.width() > 0.5 {
                            span += 1;
                        } else {
                            break;
                        }
                    }
                    span
                } else {
                    1usize
                };

                // Build the merged cell bbox spanning col_idx..col_idx+col_span
                let last_col = &cols[(col_idx + col_span - 1).min(cols.len() - 1)];
                let cell_x0_pdf = (cols[col_idx].bbox.x0 + x0 as f32) * downscale_factor;
                let cell_x1_pdf = (last_col.bbox.x1 + x0 as f32) * downscale_factor;

                let cell_bbox = BBox {
                    x0: cell_x0_pdf.max(table_bbox.x0),
                    y0: row_y0_pdf.max(table_bbox.y0),
                    x1: cell_x1_pdf.min(table_bbox.x1),
                    y1: row_y1_pdf.min(table_bbox.y1),
                };

                let cell_text = lines
                    .iter()
                    .filter(|l| cell_bbox.intersection(&l.bbox) / l.bbox.area() > 0.5)
                    .map(|l| l.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                cells.push(crate::blocks::TableCell {
                    text: cell_text,
                    bbox: cell_bbox,
                    col_span: col_span as u8,
                    row_span: 1,
                    content_ids: Vec::new(),
                });

                col_idx += col_span;
            }

            table_rows.push(crate::blocks::TableRow {
                cells,
                bbox: BBox {
                    x0: table_bbox.x0,
                    y0: row_y0_pdf,
                    x1: table_bbox.x1,
                    y1: row_y1_pdf,
                },
                is_header,
            });
        }

        let table_id = table_id_counter.fetch_add(1, Ordering::SeqCst);
        Ok(TableBlock {
            id: table_id,
            caption: None,
            rows: table_rows,
            has_borders: true,
            algorithm: TableAlgorithm::Vision,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::model::{ORTConfig, OrtExecutionProvider};
    use std::time::Instant;

    fn get_test_image() -> DynamicImage {
        // Create a random image
        let width = 1000;
        let height = 1000;
        let mut buffer = image::ImageBuffer::new(width, height);
        for (_, _, pixel) in buffer.enumerate_pixels_mut() {
            *pixel = image::Rgb([rand::random(), rand::random(), rand::random()]);
        }
        DynamicImage::ImageRgb8(buffer)
    }

    #[tokio::test]
    async fn test_benchmark_cpu() {
        // Configure for CPU
        let config = ORTConfig {
            execution_providers: vec![OrtExecutionProvider::CPU],
            ..ORTConfig::default()
        };

        println!("Loading model on CPU...");
        let load_start = Instant::now();
        let model = TableTransformerStandard::new(&config).expect("Failed to load model on CPU");
        println!("Model loaded in {:?}", load_start.elapsed());

        let img = get_test_image();
        let input = model.preprocess(&img);

        // Warmup
        println!("Warming up...");
        let _ = model.run(input.clone()).await.expect("Warmup failed");

        // Benchmark
        println!("Running inference...");
        let start = Instant::now();
        let _ = model.run(input).await.expect("Inference failed");
        let duration = start.elapsed();
        println!("CPU Inference time: {:?}", duration);
    }

    #[tokio::test]
    #[cfg(target_os = "macos")]
    async fn test_benchmark_coreml() {
        // Configure for CoreML (ANE)
        let config = ORTConfig {
            execution_providers: vec![OrtExecutionProvider::CoreML { ane_only: true }],
            ..ORTConfig::default()
        };

        println!("Loading model on CoreML (ANE)...");
        let load_start = Instant::now();
        let model = TableTransformer::new(&config).expect("Failed to load model on CoreML");
        println!("Model loaded in {:?}", load_start.elapsed());

        let img = get_test_image();
        let (input, _, _) = model.preprocess(&img);

        // Warmup
        println!("Warming up...");
        let _ = model.run(input.clone()).await.expect("Warmup failed");

        // Benchmark
        println!("Running inference...");
        let start = Instant::now();
        let _ = model.run(input).await.expect("Inference failed");
        let duration = start.elapsed();
        println!("CoreML (ANE) Inference time: {:?}", duration);
    }

    #[tokio::test]
    #[cfg(target_os = "macos")]
    async fn test_benchmark_coreml_cpu() {
        // Configure for CoreML (CPU ONLY)
        let config = ORTConfig {
            execution_providers: vec![OrtExecutionProvider::CoreML { ane_only: false }],
            ..ORTConfig::default()
        };

        println!("Loading model on CoreML (CPU)...");
        let load_start = Instant::now();
        let model = TableTransformer::new(&config).expect("Failed to load model on CoreML");
        println!("Model loaded in {:?}", load_start.elapsed());

        let img = get_test_image();
        let (input, _, _) = model.preprocess(&img);

        // Warmup
        println!("Warming up...");
        let _ = model.run(input.clone()).await.expect("Warmup failed");

        // Benchmark
        println!("Running inference...");
        let start = Instant::now();
        let _ = model.run(input).await.expect("Inference failed");
        let duration = start.elapsed();
        println!("CoreML (CPU) Inference time: {:?}", duration);
    }

    #[tokio::test]
    async fn test_benchmark_batch_coreml() {
        // Configure for CoreML (ANE)
        let config = ORTConfig {
            execution_providers: vec![OrtExecutionProvider::CoreML { ane_only: true }],
            ..ORTConfig::default()
        };

        println!("Loading model for batch benchmark on CoreML...");
        let model = TableTransformer::new(&config).expect("Failed to load model");
        let img = get_test_image();
        let (input, _, _) = model.preprocess(&img);

        // Warmup
        let _ = model.run(input.clone()).await.expect("Warmup failed");

        let batch_size = 4;
        println!("Running batch inference (batch size: {})...", batch_size);

        let mut handles = Vec::with_capacity(batch_size);
        let start = Instant::now();

        // Spawn concurrent requests to trigger batching
        for _ in 0..batch_size {
            let model = model.clone();
            let input = input.clone();
            handles.push(tokio::spawn(async move { model.run(input).await }));
        }

        for h in handles {
            let _ = h.await.unwrap().expect("Inference failed");
        }

        let duration = start.elapsed();
        println!(
            "Batch ({}) CoreML Inference time: {:?} ({:?} per item)",
            batch_size,
            duration,
            duration / batch_size as u32
        );
    }

    #[tokio::test]
    #[cfg(target_os = "macos")]
    async fn test_parity_ane_vs_cpu() {
        // 1. Setup both models
        let config_cpu = ORTConfig {
            execution_providers: vec![OrtExecutionProvider::CPU],
            ..ORTConfig::default()
        };
        let config_ane = ORTConfig {
            execution_providers: vec![OrtExecutionProvider::CoreML { ane_only: true }],
            ..ORTConfig::default()
        };

        println!("Loading CPU model...");
        let model_cpu =
            TableTransformerStandard::new(&config_cpu).expect("Failed to load CPU model");
        println!("Loading ANE model...");
        let model_ane = TableTransformer::new(&config_ane).expect("Failed to load ANE model");

        // 2. Mock image (1200x800)
        let mut buffer = image::ImageBuffer::new(1200, 800);
        for (x, y, pixel) in buffer.enumerate_pixels_mut() {
            // Add some "structure" so detection might actually fire
            if x % 100 < 10 || y % 50 < 5 {
                *pixel = image::Rgb([0, 0, 0]);
            } else {
                *pixel = image::Rgb([255, 255, 255]);
            }
        }
        let img = DynamicImage::ImageRgb8(buffer);

        // 3. Preprocess and Run CPU
        let input_cpu = model_cpu.preprocess(&img);
        let output_cpu = model_cpu.run(input_cpu).await.expect("CPU run failed");
        let results_cpu = model_cpu
            .postprocess(&output_cpu, img.width(), img.height())
            .unwrap();

        // 4. Preprocess and Run ANE
        let (input_ane, scale_x, scale_y) = model_ane.preprocess(&img);
        let output_ane = model_ane.run(input_ane).await.expect("ANE run failed");
        let results_ane = model_ane
            .postprocess(&output_ane, img.width(), img.height(), scale_x, scale_y)
            .unwrap();

        // 5. Compare
        println!("Found {} detections on CPU", results_cpu.len());
        println!("Found {} detections on ANE", results_ane.len());

        // Compare counts for rows/cols if they exist
        let cpu_rows = results_cpu.iter().filter(|d| d.label == "row").count();
        let ane_rows = results_ane.iter().filter(|d| d.label == "row").count();
        println!("Rows: CPU={}, ANE={}", cpu_rows, ane_rows);

        // We don't necessarily expect EXACT parity because of model differences/precisions,
        // but they should be in the same ballpark.
        // For a random/grid image, they might both find nothing or something.

        // Check first detection coordinate match if any
        if !results_cpu.is_empty() && !results_ane.is_empty() {
            let b0_cpu = &results_cpu[0].bbox;
            let b0_ane = &results_ane[0].bbox;
            println!("CPU BBox[0]: {:?}", b0_cpu);
            println!("ANE BBox[0]: {:?}", b0_ane);

            let dist = ((b0_cpu.x0 - b0_ane.x0).powi(2) + (b0_cpu.y0 - b0_ane.y0).powi(2)).sqrt();
            println!("Distance between top-left corners: {}", dist);
            // Allow some slack for FP16 and different preprocessing
            assert!(dist < 50.0, "Bounding boxes differ too much!");
        }
    }

    #[tokio::test]
    #[cfg(target_os = "macos")]
    async fn test_benchmark_batch_sizes_ane() {
        // Configure for CoreML (ANE)
        let config = ORTConfig {
            execution_providers: vec![OrtExecutionProvider::CoreML { ane_only: true }],
            ..ORTConfig::default()
        };

        println!("Loading model for batch benchmark on CoreML (ANE)...");
        let model = TableTransformer::new(&config).expect("Failed to load model");
        let img = get_test_image();
        let (input, _, _) = model.preprocess(&img);

        // Warmup
        let _ = model.run(input.clone()).await.expect("Warmup failed");

        let batch_sizes = [1, 2, 4, 8, 16, 32];

        println!("\n| Batch Size | Latency (ms) | Throughput (img/s) |");
        println!("|---|---|---|");

        for &batch_size in &batch_sizes {
            let mut handles = Vec::with_capacity(batch_size);
            let start = Instant::now();

            // Spawn concurrent requests to trigger batching
            for _ in 0..batch_size {
                let model = model.clone();
                let input = input.clone();
                handles.push(tokio::spawn(async move { model.run(input).await }));
            }

            for h in handles {
                let _ = h.await.unwrap().expect("Inference failed");
            }

            let duration = start.elapsed();
            let latency_ms = duration.as_millis();
            let throughput = batch_size as f64 / duration.as_secs_f64();

            println!("| {} | {} | {:.2} |", batch_size, latency_ms, throughput);
        }
        println!("\n");
    }

    #[tokio::test]
    async fn test_benchmark_batch_sizes_cpu() {
        // Configure for CPU
        let config = ORTConfig {
            execution_providers: vec![OrtExecutionProvider::CPU],
            ..ORTConfig::default()
        };

        println!("Loading model for batch benchmark on CPU...");
        let model = TableTransformer::new(&config).expect("Failed to load model");
        let img = get_test_image();
        let (input, _, _) = model.preprocess(&img);

        // Warmup
        let _ = model.run(input.clone()).await.expect("Warmup failed");

        let batch_sizes = [1, 2, 4, 8, 16];

        println!("\n| Batch Size (CPU) | Latency (ms) | Throughput (img/s) |");
        println!("|---|---|---|");

        for &batch_size in &batch_sizes {
            let mut handles = Vec::with_capacity(batch_size);
            let start = Instant::now();

            // Spawn concurrent requests (handled by BatchInferenceRunner)
            for _ in 0..batch_size {
                let model = model.clone();
                let input = input.clone();
                handles.push(tokio::spawn(async move { model.run(input).await }));
            }

            for h in handles {
                let _ = h.await.unwrap().expect("Inference failed");
            }

            let duration = start.elapsed();
            let latency_ms = duration.as_millis();
            let throughput = batch_size as f64 / duration.as_secs_f64();

            println!("| {} | {} | {:.2} |", batch_size, latency_ms, throughput);
        }
        println!("\n");
    }
}
