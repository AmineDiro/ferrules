use image::DynamicImage;

// These entities are expected to be in a parent module or crate root
// For this example, we define them here for completeness.
// In your actual project, you would remove these placeholder definitions.
mod entities {
    #[derive(Debug, Clone)]
    pub struct BBox {
        pub x0: f32,
        pub y0: f32,
        pub x1: f32,
        pub y1: f32,
    }
    #[derive(Debug)]
    pub struct Line {
        pub text: String,
        pub bbox: BBox,
        pub rotation: f32,
        pub spans: Vec<()>, // Assuming spans is an empty vec for now
    }
}
// --- End of placeholder definitions ---

use crate::entities::{BBox, Line};

#[cfg(target_os = "linux")]
use ocr_linux::parse_image_ocr as parse_image_ocr_inner;

#[cfg(target_os = "macos")]
use ocr_mac::parse_image_ocr as parse_image_ocr_inner;

#[derive(Debug)]
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

pub(crate) fn parse_image_ocr(
    image: &DynamicImage,
    rescale_factor: f32,
) -> anyhow::Result<Vec<OCRLines>> {
    parse_image_ocr_inner(image, rescale_factor)
}

#[cfg(target_os = "macos")]
mod ocr_mac {
    // ... (Your existing ocr_mac code remains unchanged) ...
    use super::*;
    use objc2::ClassType;
    use objc2_foundation::{CGRect, NSArray, NSData, NSDictionary};
    use objc2_vision::{VNImageRequestHandler, VNRecognizeTextRequest, VNRequest};
    use std::io::Cursor;
    const CONFIDENCE_THRESHOLD: f32 = 0f32;

    #[inline]
    fn cgrect_to_bbox(
        bbox: &CGRect,
        img_width: u32,
        img_height: u32,
        downscale_factor: f32,
    ) -> BBox {
        let bx0 = bbox.origin.x as f32;
        let by0 = bbox.origin.y as f32;
        let bw = bbox.size.width as f32;
        let bh = bbox.size.height as f32;

        let x0 = bx0 * img_width as f32;
        let y1 = (1f32 - by0) * (img_height as f32);
        let x1 = x0 + bw * (img_width as f32);
        let y0 = y1 - bh * (img_height as f32);

        BBox {
            x0: x0 * downscale_factor,
            y0: y0 * downscale_factor,
            x1: x1 * downscale_factor,
            y1: y1 * downscale_factor,
        }
    }

    pub(super) fn parse_image_ocr(
        image: &DynamicImage,
        rescale_factor: f32,
    ) -> anyhow::Result<Vec<OCRLines>> {
        let (img_width, img_height) = (image.width(), image.height());
        let mut buffer: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        image.write_to(&mut buffer, image::ImageFormat::Png)?;

        let mut ocr_result = Vec::new();
        unsafe {
            let request = VNRecognizeTextRequest::new();
            request.setRecognitionLevel(objc2_vision::VNRequestTextRecognitionLevel::Accurate);
            request.setUsesLanguageCorrection(true);
            let handler = VNImageRequestHandler::initWithData_options(
                VNImageRequestHandler::alloc(),
                &NSData::with_bytes(buffer.get_ref()),
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
}


// --- NEW TESSERACT-BASED IMPLEMENTATION FOR LINUX ---
// --- ULTIMATE COMPATIBILITY FIX (HOCR PARSING) ---
// --- MODERN TESSERACT-BASED IMPLEMENTATION (Use after deleting Cargo.lock) ---
#[cfg(not(target_os = "macos"))]
mod ocr_linux {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Luma, imageops};
    use std::io::Cursor;
    use tesseract::{Tesseract, PageSegMode, PageIteratorLevel};

    fn binarize_image(img: &DynamicImage, threshold: u8) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut gray = img.to_luma8();
        image::imageops::binarize(&mut gray, threshold);
        gray
    }

    fn enhance_image(img: &DynamicImage) -> DynamicImage {
        let rgba_img = img.to_rgba8();
        let contrasted = imageops::contrast(&rgba_img, 15.0);
        let sharpened = imageops::unsharpen(&contrasted, 5.0, 10);
        DynamicImage::ImageRgba8(sharpened)
    }

    fn perform_ocr_on_buffer(
        image_buffer: &ImageBuffer<Luma<u8>, Vec<u8>>,
        rescale_factor: f32
    ) -> anyhow::Result<Vec<OCRLines>> {
        let mut buf = Vec::new();
        image_buffer.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)?;

        let mut tess = Tesseract::new(None, Some("eng"))?;
        tess.set_page_seg_mode(PageSegMode::PsmAuto);
        let tess = tess.set_image_from_mem(&buf)?;

        let mut ocr_lines = Vec::new();
        let iter = tess.iter(PageIteratorLevel::Textline);

        for line in iter {
            if let (Ok(text), Some(bbox), Ok(confidence)) = (line.text(), line.bounding_box(), line.confidence()) {
                let ocr_bbox = BBox {
                    x0: bbox.x as f32 * rescale_factor,
                    y0: bbox.y as f32 * rescale_factor,
                    x1: (bbox.x + bbox.w) as f32 * rescale_factor,
                    y1: (bbox.y + bbox.h) as f32 * rescale_factor,
                };
                ocr_lines.push(OCRLines { text: text.trim().to_string(), confidence, bbox: ocr_bbox });
            }
        }
        Ok(ocr_lines)
    }

    pub(super) fn parse_image_ocr(
        image: &DynamicImage,
        rescale_factor: f32,
    ) -> anyhow::Result<Vec<OCRLines>> {
        let thresholds = [180, 150, 128, 100, 70];
        for &th in &thresholds {
            let bin_img = binarize_image(image, th);
            let ocr_result = perform_ocr_on_buffer(&bin_img, rescale_factor)?;
            if !ocr_result.is_empty() { return Ok(ocr_result); }
        }
        let enhanced_img = enhance_image(image);
        for &th in &thresholds {
            let bin_img = binarize_image(&enhanced_img, th);
            let ocr_result = perform_ocr_on_buffer(&bin_img, rescale_factor)?;
            if !ocr_result.is_empty() { return Ok(ocr_result); }
        }
        Ok(Vec.new())
    }
}