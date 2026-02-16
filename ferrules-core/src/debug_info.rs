use crate::{
    blocks::Block,
    entities::Element,
    entities::{Line, PDFPath},
    layout::model::LayoutBBox,
};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

#[derive(Archive, RkyvDeserialize, RkyvSerialize, Debug, Clone)]
pub struct DebugPage {
    pub page_number: usize,
    /// Native lines from the PDF
    pub native_lines: Vec<Line>,
    /// Native paths/drawings from the PDF
    pub paths: Vec<PDFPath>,
    /// Layout bounding boxes from the model
    pub layout_bboxes: Vec<LayoutBBox>,
    /// OCR lines (if OCR was run)
    pub ocr_lines: Vec<Line>,
    /// Elements after merging native lines and layout
    pub elements: Vec<Element>,
    /// Final blocks after layout analysis
    pub blocks: Vec<Block>,
    /// Page image data (PNG encoded)
    pub image_data: Vec<u8>,
    pub width: f32,
    pub height: f32,
}

#[derive(Archive, RkyvDeserialize, RkyvSerialize, Debug, Clone)]
pub struct DebugDocument {
    pub name: String,
    pub pages: Vec<DebugPage>,
}
