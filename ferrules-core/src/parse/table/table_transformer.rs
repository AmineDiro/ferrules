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

#[derive(Clone)]
pub struct TableTransformer {
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
}

impl BatchInferenceRunner {
    /// maximum batch size for the table transformer to process
    const MAX_BATCH_SIZE: usize = 8;
    /// maximum time to wait for a batch to be filled
    const BATCH_TIMEOUT: Duration = Duration::from_millis(10);

    fn new(session: Session, rx: mpsc::Receiver<InferenceRequest>) -> Self {
        Self {
            session: Arc::new(session),
            rx,
            max_batch_size: Self::MAX_BATCH_SIZE,
            batch_timeout: Self::BATCH_TIMEOUT,
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

            // 2. Prepare batch input
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
            let mut batch_input_vec = Vec::with_capacity(current_batch_size);
            for req in &batch {
                let mut padded = Array4::<f32>::zeros((1, 3, max_h, max_w));
                let (_, _, h, w) = req.input.dim();
                padded.slice_mut(s![.., .., ..h, ..w]).assign(&req.input);
                batch_input_vec.push(padded.remove_axis(Axis(0))); // [3, max_h, max_w]
            }

            // Stack along axis 0 -> [N, 3, max_h, max_w]
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

            // 3. Run Inference (Async)
            let input_f16 =
                match tokio::task::spawn_blocking(move || batch_tensor.mapv(half::f16::from_f32))
                    .await
                {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("Failed to spawn blocking for f16 conversion: {}", e);
                        for req in batch.drain(..) {
                            let _ = req.response_tx.send(Err(FerrulesError::LayoutParsingError));
                        }
                        continue;
                    }
                };

            let run_result = async {
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
            }
            .await;

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

impl TableTransformer {
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

        let session = Session::builder()
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_execution_providers(execution_providers)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_optimization_level(opt_lvl)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_intra_threads(config.intra_threads)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .with_inter_threads(config.inter_threads)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?
            .commit_from_memory(TABLE_MODEL_BYTES)
            .map_err(|e| FerrulesError::TableTransformerModelError(e.to_string()))?;

        let (tx, rx) = mpsc::channel(32);
        let runner = BatchInferenceRunner::new(session, rx);
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
            .map_err(|_| FerrulesError::LayoutParsingError)?;

        rx.await.map_err(|_| FerrulesError::LayoutParsingError)?
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
