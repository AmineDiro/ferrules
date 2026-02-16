use image::DynamicImage;

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
    use super::*;
    use objc2::ClassType;
    use objc2_foundation::{CGRect, NSArray, NSData, NSDictionary};
    use objc2_vision::{VNImageRequestHandler, VNRecognizeTextRequest, VNRequest};
    use std::io::Cursor;
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
            // TODO set the languages array
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

    #[cfg(test)]
    mod tests {
        use super::*;
        use image::ImageReader;
        use std::{path::Path, time::Instant};

        #[test]
        fn test_ocr_apple_vision() {
            if Path::new("./test_data/double_cols.jpg").exists() {
                let image = ImageReader::open("./test_data/double_cols.jpg")
                    .unwrap()
                    .decode()
                    .unwrap();

                let s = Instant::now();
                let ocr_result = parse_image_ocr(&image, 1f32);
                assert!(ocr_result.is_ok());

                println!(
                    "OCR took: {}ms",
                    Instant::now().duration_since(s).as_millis()
                );
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod ocr_linux {

    use super::*;

    pub(super) fn parse_image_ocr(
        _image: &DynamicImage,
        _rescale_factor: f32,
    ) -> anyhow::Result<Vec<OCRLines>> {
        anyhow::bail!("not implemented yet")
    }
}
