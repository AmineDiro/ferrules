use image::DynamicImage;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::{oneshot, Semaphore};
use tracing::{Instrument, Span};

use crate::entities::{BBox, Line, PageID};
use crate::error::FerrulesError;
use crate::metrics::StepMetrics;

const CONCURRENT_OCR_REQUESTS: usize = 32;
const MAX_OCR_BATCH_SIZE: usize = 8;
const OCR_BATCH_TIMEOUT_MS: u64 = 50;

#[derive(Debug)]
pub struct OCRMetadata {
    pub(crate) response_tx: oneshot::Sender<Result<ParseOCRResponse, FerrulesError>>,
    pub(crate) queue_time: Instant,
}

#[derive(Debug)]
pub(crate) struct ParseOCRRequest {
    pub(crate) page_id: PageID,
    pub(crate) page_image: Arc<DynamicImage>,
    pub(crate) rescale_factor: f32,
    pub(crate) metadata: OCRMetadata,
}

#[derive(Debug)]
pub(crate) struct ParseOCRResponse {
    pub(crate) ocr_lines: Vec<OCRLines>,
    pub(crate) step_metrics: StepMetrics,
}

#[derive(Debug, Clone)]
pub struct OCRQueue {
    queue: Sender<(ParseOCRRequest, Span)>,
}

impl OCRQueue {
    pub fn new(ocr_parser: Arc<OCRParser>) -> Self {
        let (queue_sender, queue_receiver) = mpsc::channel(128); // Larger buffer for OCR requests

        tokio::task::spawn(start_ocr_parser(ocr_parser, queue_receiver));
        Self {
            queue: queue_sender,
        }
    }

    pub(crate) async fn push(&self, req: ParseOCRRequest) -> Result<(), FerrulesError> {
        let span = Span::current();
        self.queue
            .send((req, span))
            .await
            .map_err(|_| FerrulesError::LayoutParsingError) // TODO: Specific OCR error
    }
}

async fn start_ocr_parser(
    ocr_parser: Arc<OCRParser>,
    mut input_rx: Receiver<(ParseOCRRequest, Span)>,
) {
    let s = Arc::new(Semaphore::new(CONCURRENT_OCR_REQUESTS));
    while let Some((req, span)) = input_rx.recv().await {
        let queue_time = req.metadata.queue_time.elapsed().as_secs_f64() * 1000.0;
        let page_id = req.page_id;
        tracing::debug!("ocr request queue time for page {page_id} took: {queue_time}ms");
        tokio::spawn(
            handle_ocr_request(s.clone(), ocr_parser.clone(), req, queue_time).instrument(span),
        );
    }
}

async fn handle_ocr_request(
    s: Arc<Semaphore>,
    parser: Arc<OCRParser>,
    req: ParseOCRRequest,
    ocr_queue_time_ms: f64,
) {
    let start_wait = Instant::now();
    let _permit = s.acquire().await.unwrap();
    let idle_time_ms = start_wait.elapsed().as_secs_f64() * 1000.0;

    let ParseOCRRequest {
        page_id,
        page_image,
        rescale_factor,
        metadata,
    } = req;

    let start = Instant::now();
    let (tx, rx) = oneshot::channel();
    let _ = parser
        .inference_tx
        .send(OCRInferenceRequest {
            image: page_image,
            rescale_factor,
            response_tx: tx,
        })
        .await;

    let ocr_result = rx.await.unwrap_or(Err(FerrulesError::LayoutParsingError));
    let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    drop(_permit);

    tracing::debug!("ocr inference time for page {page_id} took: {execution_time_ms}ms");

    let response = ocr_result.map(|ocr_lines| ParseOCRResponse {
        ocr_lines,
        step_metrics: StepMetrics {
            queue_time_ms: ocr_queue_time_ms,
            execution_time_ms,
            idle_time_ms,
        },
    });

    let _ = metadata.response_tx.send(response);
}

struct OCRInferenceRequest {
    image: Arc<DynamicImage>,
    rescale_factor: f32,
    response_tx: oneshot::Sender<Result<Vec<OCRLines>, FerrulesError>>,
}

struct BatchOCRRunner {
    rx: Receiver<OCRInferenceRequest>,
}

impl BatchOCRRunner {
    async fn run(mut self) {
        let mut batch = Vec::with_capacity(MAX_OCR_BATCH_SIZE);

        loop {
            let first_req = match self.rx.recv().await {
                Some(req) => req,
                None => break,
            };
            batch.push(first_req);

            let deadline = tokio::time::Instant::now()
                + std::time::Duration::from_millis(OCR_BATCH_TIMEOUT_MS);

            while batch.len() < MAX_OCR_BATCH_SIZE {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match tokio::time::timeout(remaining, self.rx.recv()).await {
                    Ok(Some(req)) => batch.push(req),
                    Ok(None) => break,
                    Err(_) => break,
                }
            }

            if batch.is_empty() {
                continue;
            }

            let batch_size = batch.len();
            tracing::debug!("Processing OCR batch of size {}", batch_size);

            let mut images = Vec::with_capacity(batch_size);
            let mut restxs = Vec::with_capacity(batch_size);

            for req in batch.drain(..) {
                images.push((req.image, req.rescale_factor));
                restxs.push(req.response_tx);
            }

            let results = tokio::task::spawn_blocking(move || parse_images_ocr_batch(images))
                .await
                .unwrap_or_else(|_| {
                    let mut errs = Vec::with_capacity(batch_size);
                    for _ in 0..batch_size {
                        errs.push(Err(anyhow::anyhow!("OCR Batch Panic")));
                    }
                    errs
                });

            for (tx, res) in restxs.into_iter().zip(results) {
                let _ = tx.send(res.map_err(|_| FerrulesError::LayoutParsingError));
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct OCRParser {
    inference_tx: Sender<OCRInferenceRequest>,
}

impl OCRParser {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(256);
        let runner = BatchOCRRunner { rx };
        tokio::spawn(runner.run());
        Self { inference_tx: tx }
    }

    pub async fn parse(
        &self,
        image: &DynamicImage,
        rescale_factor: f32,
    ) -> Result<Vec<OCRLines>, FerrulesError> {
        let (tx, rx) = oneshot::channel();
        let _ = self
            .inference_tx
            .send(OCRInferenceRequest {
                image: Arc::new(image.clone()),
                rescale_factor,
                response_tx: tx,
            })
            .await;
        rx.await.unwrap_or(Err(FerrulesError::LayoutParsingError))
    }
}

#[cfg(target_os = "linux")]
use ocr_linux::{parse_images_ocr_batch, parse_single_image_ocr};

#[cfg(target_os = "macos")]
use ocr_mac::{parse_images_ocr_batch, parse_single_image_ocr};

#[derive(Debug, Clone)]
pub struct OCRLines {
    pub text: String,
    pub confidence: f32,
    pub bbox: BBox,
}

impl OCRLines {
    pub(crate) fn to_line(&self) -> Line {
        Line {
            text: self.text.to_string(),
            bbox: self.bbox.clone(),
            rotation: 0f32,
            spans: vec![],
        }
    }
}

pub async fn parse_image_ocr(
    image: &DynamicImage,
    _debug_dir: Option<PathBuf>,
    rescale_factor: f32,
) -> Result<(Vec<OCRLines>, StepMetrics), FerrulesError> {
    let start = Instant::now();
    let ocr_result = parse_single_image_ocr(image, rescale_factor)
        .map_err(|_| FerrulesError::LayoutParsingError)?;
    let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let step_metrics = StepMetrics {
        queue_time_ms: 0.0,
        execution_time_ms,
        idle_time_ms: 0.0,
    };

    Ok((ocr_result, step_metrics))
}

#[cfg(target_os = "macos")]
mod ocr_mac {
    use super::*;
    use objc2::ClassType;
    use objc2_foundation::{CGRect, NSArray, NSData, NSDictionary};
    use objc2_vision::{VNImageRequestHandler, VNRecognizeTextRequest, VNRequest};
    const CONFIDENCE_THRESHOLD: f32 = 0f32;

    /// Convert vision coordinates to Bbox absolute coordinates
    #[inline]
    fn cgrect_to_bbox(
        bbox: &CGRect,
        img_width: u32,
        img_height: u32,
        downscale_factor: f32,
    ) -> BBox {
        // Change to (upper-left, lower-right)
        let bx0 = bbox.origin.x as f32;
        let by0 = bbox.origin.y as f32;
        let bw = bbox.size.width as f32;
        let bh = bbox.size.height as f32;

        let x0 = bx0 * img_width as f32;
        let y1 = (1f32 - by0) * (img_height as f32);
        let x1 = x0 + bw * (img_width as f32);
        let y0 = y1 - bh * (img_height as f32);

        assert!(x0 < x1);
        assert!(y0 < y1);
        assert!(x1 < img_width as f32);
        assert!(y1 < img_height as f32);

        BBox {
            x0: x0 * downscale_factor,
            y0: y0 * downscale_factor,
            x1: x1 * downscale_factor,
            y1: y1 * downscale_factor,
        }
    }

    pub(super) fn parse_images_ocr_batch(
        inputs: Vec<(Arc<DynamicImage>, f32)>,
    ) -> Vec<anyhow::Result<Vec<OCRLines>>> {
        if inputs.is_empty() {
            return vec![];
        }

        if inputs.len() == 1 {
            let (image, rescale_factor) = inputs.into_iter().next().unwrap();
            return vec![parse_single_image_ocr(&image, rescale_factor)];
        }

        // Stitching logic for true batching
        let mut total_height = 0u32;
        let mut max_width = 0u32;
        for (img, _) in &inputs {
            total_height += img.height();
            max_width = max_width.max(img.width());
        }

        let mut combined_image = image::ImageBuffer::new(max_width, total_height);
        let mut offsets = Vec::with_capacity(inputs.len());
        let mut current_y = 0u32;

        for (img, _) in &inputs {
            offsets.push(current_y);
            image::imageops::overlay(&mut combined_image, img.as_ref(), 0, current_y as i64);
            current_y += img.height();
        }

        let combined_image = DynamicImage::ImageRgba8(combined_image);
        let mut buffer = std::io::Cursor::new(Vec::new());
        if let Err(e) = combined_image.write_to(&mut buffer, image::ImageFormat::Tiff) {
            let mut errs = Vec::with_capacity(inputs.len());
            for _ in 0..inputs.len() {
                errs.push(Err(anyhow::anyhow!(e.to_string())));
            }
            return errs;
        }
        let raw_data = buffer.into_inner();

        let mut final_results = vec![Vec::new(); inputs.len()];

        unsafe {
            let mut requests = Vec::with_capacity(inputs.len());
            for (i, (_, _)) in inputs.iter().enumerate() {
                let request = VNRecognizeTextRequest::new();
                request.setRecognitionLevel(objc2_vision::VNRequestTextRecognitionLevel::Accurate);
                request.setUsesLanguageCorrection(true);

                // Set Region Of Interest for this specific image in the strip
                let y0 = offsets[i] as f64 / total_height as f64;
                let h = inputs[i].0.height() as f64 / total_height as f64;
                // Vision ROI is [x, y, w, h] in normalized coords (0,0 is bottom-left)
                // Since we stitched top-to-bottom, we need to flip Y
                let roi_y = 1.0 - y0 - h;
                request.setRegionOfInterest(objc2_foundation::CGRect {
                    origin: objc2_foundation::CGPoint { x: 0.0, y: roi_y },
                    size: objc2_foundation::CGSize {
                        width: 1.0,
                        height: h,
                    },
                });

                requests.push(request);
            }

            let handler = VNImageRequestHandler::initWithData_options(
                VNImageRequestHandler::alloc(),
                &NSData::with_bytes(&raw_data),
                &NSDictionary::new(),
            );

            let v_requests: Vec<&VNRequest> =
                requests.iter().map(|r| r.as_ref() as &VNRequest).collect();
            let ns_requests = NSArray::from_slice(&v_requests);
            if let Err(e) = handler.performRequests_error(&ns_requests) {
                let mut errs = Vec::with_capacity(inputs.len());
                for _ in 0..inputs.len() {
                    errs.push(Err(anyhow::anyhow!(e.to_string())));
                }
                return errs;
            }

            for (i, request) in requests.iter().enumerate() {
                let rescale_factor = inputs[i].1;
                let img_width = inputs[i].0.width();
                let img_height = inputs[i].0.height();

                if let Some(result) = request.results() {
                    for recognized_text_region in result.to_vec() {
                        if (*recognized_text_region).confidence() > CONFIDENCE_THRESHOLD {
                            if let Some(rec_text) = recognized_text_region.topCandidates(1).first()
                            {
                                let bbox = (*recognized_text_region).boundingBox();
                                // Note: bbox from Vision here is RELATIVE to ROI if we use ROI correctly?
                                // Actually, Vision bboxes are typically relative to the WHOLE image if ROI is set on request?
                                let bbox =
                                    cgrect_to_bbox(&bbox, img_width, img_height, rescale_factor);
                                final_results[i].push(OCRLines {
                                    text: rec_text.string().to_string(),
                                    confidence: rec_text.confidence(),
                                    bbox,
                                })
                            }
                        }
                    }
                }
            }
        }

        final_results.into_iter().map(Ok).collect()
    }

    pub(super) fn parse_single_image_ocr(
        image: &DynamicImage,
        rescale_factor: f32,
    ) -> anyhow::Result<Vec<OCRLines>> {
        let (img_width, img_height) = (image.width(), image.height());
        let mut buffer = std::io::Cursor::new(Vec::new());
        image.write_to(&mut buffer, image::ImageFormat::Tiff)?;
        let raw_data = buffer.into_inner();

        let mut ocr_result = Vec::new();
        unsafe {
            let request = VNRecognizeTextRequest::new();
            request.setRecognitionLevel(objc2_vision::VNRequestTextRecognitionLevel::Accurate);
            request.setUsesLanguageCorrection(true);

            let handler = VNImageRequestHandler::initWithData_options(
                VNImageRequestHandler::alloc(),
                &NSData::with_bytes(&raw_data),
                &NSDictionary::new(),
            );

            let requests = NSArray::from_slice(&[request.as_ref() as &VNRequest]);
            handler.performRequests_error(&requests)?;

            if let Some(result) = request.results() {
                for recognized_text_region in result.to_vec() {
                    if (*recognized_text_region).confidence() > CONFIDENCE_THRESHOLD {
                        if let Some(rec_text) = recognized_text_region.topCandidates(1).first() {
                            let bbox = (*recognized_text_region).boundingBox();
                            let bbox = cgrect_to_bbox(&bbox, img_width, img_height, rescale_factor);
                            ocr_result.push(OCRLines {
                                text: rec_text.string().to_string(),
                                confidence: rec_text.confidence(),
                                bbox,
                            })
                        }
                    }
                }
            }
        }
        Ok(ocr_result)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use image::ImageReader;
        use std::{path::Path, time::Instant};

        #[tokio::test]
        async fn test_ocr_apple_vision() {
            if Path::new("./test_data/double_cols.jpg").exists() {
                let image = ImageReader::open("./test_data/double_cols.jpg")
                    .unwrap()
                    .decode()
                    .unwrap();

                let s = Instant::now();
                let ocr_result = parse_image_ocr(&image, None, 1f32).await;
                assert!(ocr_result.is_ok());

                println!(
                    "OCR took: {}ms",
                    Instant::now().duration_since(s).as_millis()
                );
            }
        }

        #[test]
        fn test_ocr_batching_perf() {
            let image_path = if Path::new("./test_data/double_cols.jpg").exists() {
                "./test_data/double_cols.jpg"
            } else {
                "../test_data/double_cols.jpg"
            };
            let image = ImageReader::open(image_path).unwrap().decode().unwrap();
            let n = 5;

            // 1. Parallel handles
            let s = Instant::now();
            let mut handles = vec![];
            for _ in 0..n {
                let img = image.clone();
                handles.push(std::thread::spawn(move || {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let _ = rt.block_on(parse_image_ocr(&img, None, 1.0));
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
            let parallel_duration = s.elapsed();
            eprintln!(
                "Parallel execution ({} threads, 1 request each) took: {:?}",
                n, parallel_duration
            );

            // 2. Batch requests
            let s = Instant::now();
            unsafe {
                let mut buffer = std::io::Cursor::new(Vec::new());
                image
                    .write_to(&mut buffer, image::ImageFormat::Tiff)
                    .unwrap();
                let raw_data = buffer.into_inner();

                let mut requests = vec![];
                for _ in 0..n {
                    let request = VNRecognizeTextRequest::new();
                    request
                        .setRecognitionLevel(objc2_vision::VNRequestTextRecognitionLevel::Accurate);
                    request.setUsesLanguageCorrection(true);
                    requests.push(request);
                }

                let handler = VNImageRequestHandler::initWithData_options(
                    VNImageRequestHandler::alloc(),
                    &NSData::with_bytes(&raw_data),
                    &NSDictionary::new(),
                );

                let v_requests: Vec<&VNRequest> =
                    requests.iter().map(|r| r.as_ref() as &VNRequest).collect();
                let ns_requests = NSArray::from_slice(&v_requests);
                let _ = handler.performRequests_error(&ns_requests);
            }
            let batch_duration = s.elapsed();
            eprintln!(
                "Batch execution (1 handler, {} requests) took: {:?}",
                n, batch_duration
            );
        }

        #[tokio::test]
        async fn test_ocr_parser_batching() {
            if Path::new("./test_data/double_cols.jpg").exists() {
                let image = ImageReader::open("./test_data/double_cols.jpg")
                    .unwrap()
                    .decode()
                    .unwrap();
                let parser = OCRParser::new();
                let n = 3;
                let mut set = tokio::task::JoinSet::new();

                let start = Instant::now();
                for i in 0..n {
                    let img = image.clone();
                    let p = parser.clone();
                    set.spawn(async move { p.parse(&img, 1.0).await });
                }

                let mut results = Vec::new();
                while let Some(res) = set.join_next().await {
                    results.push(res.unwrap().unwrap());
                }
                let duration = start.elapsed();
                eprintln!("OCR Parser batching ({} requests) took: {:?}", n, duration);
                assert_eq!(results.len(), n);
                for res in results {
                    assert!(!res.is_empty());
                }
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod ocr_linux {

    use super::*;

    pub(super) fn parse_images_ocr_batch(
        _inputs: Vec<(Arc<DynamicImage>, f32)>,
    ) -> Vec<anyhow::Result<Vec<OCRLines>>> {
        vec![Err(anyhow::anyhow!("not implemented yet"))]
    }

    pub(super) fn parse_single_image_ocr(
        _image: &DynamicImage,
        _rescale_factor: f32,
    ) -> anyhow::Result<Vec<OCRLines>> {
        anyhow::bail!("not implemented yet")
    }
}
