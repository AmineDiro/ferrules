// use std::path::Path;

use anyhow::Context;
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use lazy_static::lazy_static;
use ndarray::{s, Array4, ArrayBase, Axis, Dim, OwnedRepr};
use ort::{
    execution_providers::{
        CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
        TensorRTExecutionProvider,
    },
    session::{builder::GraphOptimizationLevel, Session},
};
use rayon::prelude::*;

use crate::entities::BBox;

pub const LAYOUT_MODEL_BYTES: &[u8] = include_bytes!("../../models/yolov8s-doclaynet.onnx");

lazy_static! {
    static ref ID2LABEL: [&'static str; 11] = [
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title",
    ];
}

#[derive(Debug, Default, Clone)]
pub struct LayoutBBox {
    pub id: i32,
    pub bbox: BBox,
    pub label: &'static str,
    pub proba: f32,
}

impl LayoutBBox {
    pub fn is_text_block(&self) -> bool {
        self.label == "Text"
            || self.label == "Caption"
            || self.label == "Footnote"
            || self.label == "Formula"
            || self.label == "List-item"
            || self.label == "Page-footer"
            || self.label == "Page-header"
            || self.label == "Section-header"
            || self.label == "Title"
    }
}

#[derive(Debug)]
pub struct ORTLayoutParser {
    session: Session,
    input_name: String,
    output_name: String,
}

impl ORTLayoutParser {
    /// Required width of the input image for layout parsing.
    pub const REQUIRED_WIDTH: u32 = 1024;
    /// Required height of the input image for layout parsing.
    pub const REQUIRED_HEIGHT: u32 = 1024;

    // Output size of the tensor from the ONNX model.
    // It has dimensions [batch_size = 1, classes + bbox = 15, candidate_boxes = 21504].
    const OUTPUT_SIZE: [usize; 3] = [1, 15, 21504];

    /// Confidence threshold for filtering out low probability bounding boxes.
    /// Bounding boxes with probability below this threshold will be ignored.
    pub const CONF_THRESHOLD: f32 = 0.3;

    /// Intersection over Union (IOU) threshold for non-maximum suppression (NMS) algorithm.
    /// It determines the overlap between bounding boxes before suppression.
    pub const IOU_THRESHOLD: f32 = 0.8;

    pub fn new() -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([
                TensorRTExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
                CoreMLExecutionProvider::default()
                    // .with_ane_only()
                    .with_subgraphs()
                    .build(),
                CPUExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            // TODO: Configure using param
            .with_intra_threads(12)?
            .with_inter_threads(12)?
            .commit_from_memory(LAYOUT_MODEL_BYTES)?;

        let input_name = session
            .inputs
            .first()
            .map(|i| &i.name)
            .context("can't find name for first input")?
            .to_owned();

        let output_name = session
            .outputs
            .first()
            .map(|i| &i.name)
            .context("can't find name output input")?
            .to_owned();

        Ok(Self {
            session,
            input_name,
            output_name,
        })
    }

    pub fn run(
        &self,
        input: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    ) -> anyhow::Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>> {
        let outputs = &self
            .session
            .run(ort::inputs![&self.input_name=> input.view()]?)?;

        let output_tensor = outputs
            .get(&self.output_name)
            .context("can't get the value of first output")?
            .try_extract_tensor::<f32>()?;

        let output_tensor = output_tensor
            .to_shape(Self::OUTPUT_SIZE)
            .unwrap()
            .to_owned();

        Ok(output_tensor)
    }
    pub fn parse_layout_batch(
        &self,
        page_imgs: &[DynamicImage],
        bbox_rescale_factors: &[f32],
    ) -> anyhow::Result<Vec<Vec<LayoutBBox>>> {
        let bboxes: Result<Vec<_>, _> = page_imgs
            .into_par_iter()
            .zip(bbox_rescale_factors.par_iter())
            .map(|(page_img, bbox_rescale_factor)| {
                let bboxes = self.parse_layout(page_img, *bbox_rescale_factor)?;
                Ok(bboxes)
            })
            .collect();
        bboxes
    }

    pub fn parse_layout(
        &self,
        page_img: &DynamicImage,
        bbox_rescale_factor: f32,
    ) -> anyhow::Result<Vec<LayoutBBox>> {
        let (img_width, img_height) = (page_img.width(), page_img.height());
        let input = self.preprocess(page_img);
        let output_tensor = self.run(input)?;
        let mut bboxes =
            self.extract_bboxes(output_tensor, img_width, img_height, bbox_rescale_factor);
        nms(&mut bboxes, Self::IOU_THRESHOLD);

        Ok(bboxes)
    }

    fn extract_bboxes(
        &self,
        output: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>,
        original_width: u32,
        original_height: u32,
        rescale_factor: f32,
    ) -> Vec<LayoutBBox> {
        // Tensor shape: (bs, bbox(4) + classes(15), anchors )
        let mut result = Vec::new();
        let output = output.slice(s![0, .., ..]);
        let mut bbox_id = 0;
        for prediction in output.axis_iter(Axis(1)) {
            // Prediction dim: (15,) -> (4 bbox, 11 labels)
            const CXYWH_OFFSET: usize = 4;
            let bbox = prediction.slice(s![0..CXYWH_OFFSET]);
            let classes = prediction.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + ID2LABEL.len()]);
            let (max_prob_idx, &proba) = classes
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            if proba < Self::CONF_THRESHOLD {
                continue;
            }
            let label = ID2LABEL[max_prob_idx];
            let ratio = (Self::REQUIRED_WIDTH as f32 / original_width as f32)
                .min(Self::REQUIRED_HEIGHT as f32 / original_height as f32);
            let xc = bbox[0_usize] / ratio;
            let yc = bbox[1_usize] / ratio;
            let w = bbox[2_usize] / ratio;
            let h = bbox[3_usize] / ratio;
            // Change to (upper-left, lower-right)
            let x0 = (xc - (w / 2.0)).min(original_width as f32).max(0f32);
            let y0 = (yc - (h / 2.0)).min(original_height as f32).max(0f32);
            let x1 = (xc + (w / 2.0)).max(0f32).min(original_width as f32);
            let y1 = (yc + (h / 2.0)).max(0f32).min(original_height as f32);

            debug_assert!(x0 <= x1 && x1 <= original_width as f32);
            debug_assert!(y0 <= y1 && y1 <= original_height as f32);

            if x0 > x1 || y0 > y1 {
                eprintln!("bbox error: ({x0},{y1}), ({x1},{y1})");
                continue;
            }

            result.push(LayoutBBox {
                id: bbox_id,
                bbox: BBox {
                    x0: x0 * rescale_factor,
                    y0: y0 * rescale_factor,
                    x1: x1 * rescale_factor,
                    y1: y1 * rescale_factor,
                },
                proba,
                label,
            });
            bbox_id += 1;
        }

        result
    }
    fn scale_wh(&self, w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }

    fn _preprocess_batch(&self, batch_imgs: &[DynamicImage]) -> Array4<f32> {
        let (w0, h0) = batch_imgs.first().unwrap().dimensions();
        let (_, w_new, h_new) = self.scale_wh(
            w0 as f32,
            h0 as f32,
            Self::REQUIRED_WIDTH as f32,
            Self::REQUIRED_HEIGHT as f32,
        ); // f32 round

        let mut input_tensor = Array4::ones([
            batch_imgs.len(),
            3,
            Self::REQUIRED_HEIGHT as usize,
            Self::REQUIRED_WIDTH as usize,
        ]);

        input_tensor.fill(144.0 / 255.0);

        for (idx, img) in batch_imgs.iter().enumerate() {
            let resized_img = img.resize_exact(w_new as u32, h_new as u32, FilterType::Triangle);
            for (x, y, pixel) in resized_img.pixels() {
                let x = x as usize;
                let y = y as _;
                let [r, g, b, _] = pixel.0;
                input_tensor[[idx, 0, y, x]] = r as f32 / 255.0;
                input_tensor[[idx, 1, y, x]] = g as f32 / 255.0;
                input_tensor[[idx, 2, y, x]] = b as f32 / 255.0;
            }
        }

        input_tensor
    }

    fn preprocess(&self, img: &DynamicImage) -> Array4<f32> {
        let (w0, h0) = img.dimensions();
        let (_, w_new, h_new) = self.scale_wh(
            w0 as f32,
            h0 as f32,
            Self::REQUIRED_WIDTH as f32,
            Self::REQUIRED_HEIGHT as f32,
        ); // f32 round
        let resized_img = img.resize_exact(w_new as u32, h_new as u32, FilterType::Triangle);
        // TODO: reuse this buffer between batches
        let mut input_tensor = Array4::ones([
            1,
            3,
            Self::REQUIRED_HEIGHT as usize,
            Self::REQUIRED_WIDTH as usize,
        ]);
        input_tensor.fill(144.0 / 255.0);
        for (x, y, pixel) in resized_img.pixels() {
            let x = x as usize;
            let y = y as _;
            let [r, g, b, _] = pixel.0;
            input_tensor[[0, 0, y, x]] = r as f32 / 255.0;
            input_tensor[[0, 1, y, x]] = g as f32 / 255.0;
            input_tensor[[0, 2, y, x]] = b as f32 / 255.0;
        }
        input_tensor
    }
}

/// runs nms on without taking into account which class
fn nms(raw_bboxes: &mut Vec<LayoutBBox>, iou_threshold: f32) {
    raw_bboxes.sort_by(|r1, r2| r2.proba.partial_cmp(&r1.proba).unwrap());
    let mut current_index = 0;
    for index in 0..raw_bboxes.len() {
        let mut drop = false;
        for prev_index in 0..current_index {
            let iou = raw_bboxes[prev_index]
                .bbox
                .relaxed_iou(&raw_bboxes[index].bbox);
            if iou > iou_threshold {
                drop = true;
                break;
            }
        }
        if !drop {
            raw_bboxes.swap(current_index, index);
            current_index += 1;
        }
    }
    // Everything after has been swapped
    raw_bboxes.truncate(current_index);
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_nms_high_overlap_contained_box() {
        let mut raw_bboxes = vec![
            LayoutBBox {
                id: 0,
                bbox: BBox {
                    x0: 0.0,
                    y0: 0.0,
                    x1: 3.0,
                    y1: 3.0,
                },
                label: "A",
                proba: 0.85,
            },
            LayoutBBox {
                id: 1,
                // Box fully contained within box #0
                bbox: BBox {
                    x0: 1.0,
                    y0: 1.0,
                    x1: 2.0,
                    y1: 2.0,
                },
                label: "A",
                proba: 0.95,
            },
        ];

        let iou_threshold = 0.5;
        nms(&mut raw_bboxes, iou_threshold);

        assert_eq!(raw_bboxes.len(), 1);
        // assert_eq!(raw_bboxes[0].id, 1);
    }

    #[test]
    fn test_nms_no_overlap() {
        let mut raw_bboxes = vec![
            LayoutBBox {
                id: 0,
                bbox: BBox {
                    x0: 0.0,
                    y0: 0.0,
                    x1: 1.0,
                    y1: 1.0,
                },
                label: "A",
                proba: 0.9,
            },
            LayoutBBox {
                id: 1, // Added id
                bbox: BBox {
                    x0: 2.0,
                    y0: 2.0,
                    x1: 3.0,
                    y1: 3.0,
                },
                label: "A",
                proba: 0.95,
            },
            LayoutBBox {
                id: 2, // Added id
                bbox: BBox {
                    x0: 4.0,
                    y0: 4.0,
                    x1: 5.0,
                    y1: 5.0,
                },
                label: "A",
                proba: 0.85,
            },
        ];

        let iou_threshold = 0.5;
        nms(&mut raw_bboxes, iou_threshold);

        assert_eq!(raw_bboxes.len(), 3);
    }

    #[test]
    fn test_nms_high_overlap_same_label() {
        let mut raw_bboxes = vec![
            LayoutBBox {
                id: 0,
                bbox: BBox {
                    x0: 0.0,
                    y0: 0.0,
                    x1: 2.0,
                    y1: 2.0,
                },
                label: "A",
                proba: 0.85,
            },
            LayoutBBox {
                id: 1,
                // Shifted slightly inside box #1 so intersection is large
                bbox: BBox {
                    x0: 0.5,
                    y0: 0.5,
                    x1: 2.0,
                    y1: 2.0,
                },
                label: "A",
                proba: 0.95,
            },
            LayoutBBox {
                id: 2,
                // Exactly the same as box #1 => IOU=1 with box #1
                bbox: BBox {
                    x0: 0.0,
                    y0: 0.0,
                    x1: 2.0,
                    y1: 2.0,
                },
                label: "A",
                proba: 0.90,
            },
        ];

        // Now all pairwise IOUs exceed 0.5, so NMS should keep only the
        // box with the highest probability (0.95).
        let iou_threshold = 0.5;
        nms(&mut raw_bboxes, iou_threshold);

        // We expect exactly one box left, with proba = 0.95.
        assert_eq!(raw_bboxes.len(), 1);
        assert_eq!(raw_bboxes[0].proba, 0.95);
    }

    #[test]
    fn test_multithreaded() {
        let session = Session::builder()
            .unwrap()
            .with_execution_providers([CoreMLExecutionProvider::default()
                // .with_ane_only()
                .with_subgraphs()
                .build()])
            .unwrap()
            .commit_from_memory(LAYOUT_MODEL_BYTES)
            .unwrap();

        use rayon::prelude::*;

        (0..20).into_par_iter().for_each(|_| {
            let input: Array4<f32> = Array4::ones([1, 3, 1024, 1024]);
            session
                .run(ort::inputs!["images"=> input].unwrap())
                .unwrap();
        })
    }
}
