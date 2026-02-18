use crate::{
    entities::{BBox, Element, ElementType, PageID},
    error::FerrulesError,
};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

pub type TitleLevel = u8;

#[derive(
    Clone, Debug, Default, Deserialize, Serialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct ImageBlock {
    pub(crate) id: usize,
    pub(crate) caption: Option<String>,
}

impl ImageBlock {
    pub(crate) fn path(&self) -> String {
        format!("img_{}.png", self.id)
    }
}

#[derive(
    Clone, Debug, Default, Deserialize, Serialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct TextBlock {
    pub text: String,
}

#[derive(
    Clone, Debug, Default, Deserialize, Serialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct List {
    pub items: Vec<String>,
}

#[derive(
    Clone, Debug, Default, Deserialize, Serialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub enum TableAlgorithm {
    #[default]
    Unknown,
    Lattice,
    Stream,
    Vision,
}

#[derive(
    Clone, Debug, Default, Deserialize, Serialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct TableBlock {
    pub(crate) id: usize,
    pub(crate) caption: Option<String>,
    pub rows: Vec<TableRow>,
    pub has_borders: bool,
    pub algorithm: TableAlgorithm,
}

impl TableBlock {
    pub(crate) fn path(&self) -> String {
        format!("table_{}.png", self.id)
    }
}

#[derive(
    Clone, Debug, Default, Deserialize, Serialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct TableRow {
    pub cells: Vec<TableCell>,
    pub is_header: bool,
    pub bbox: BBox,
}

#[derive(
    Clone, Debug, Default, Deserialize, Serialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct TableCell {
    /// IDs of blocks contained within this cell.
    /// This avoids recursion in serializable structures.
    pub content_ids: Vec<usize>,
    pub text: String,
    pub row_span: u8,
    pub col_span: u8,
    pub bbox: BBox,
}

#[derive(
    Clone, Debug, Default, Deserialize, Serialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct Title {
    pub level: TitleLevel,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Archive, RkyvDeserialize, RkyvSerialize)]
#[serde(tag = "block_type")]
pub enum BlockType {
    Header(TextBlock),
    Footer(TextBlock),
    Title(Title),
    ListBlock(List),
    TextBlock(TextBlock),
    Image(ImageBlock),
    Table(TableBlock),
}

impl std::fmt::Display for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Archive, RkyvDeserialize, RkyvSerialize)]
pub struct Block {
    pub id: usize,
    pub kind: BlockType,
    pub pages_id: Vec<PageID>,
    pub bbox: BBox,
}

impl Block {
    pub(crate) fn merge(&mut self, element: Element) -> Result<(), FerrulesError> {
        match &mut self.kind {
            BlockType::TextBlock(text) => {
                if let ElementType::Text = &element.kind {
                    self.bbox.merge(&element.bbox);
                    text.text.push('\n');
                    text.text.push_str(&element.text_block.text);

                    // add page_id
                    Ok(())
                } else {
                    Err(FerrulesError::BlockMergeError {
                        element: Box::new(element),
                        block_id: self.id,
                        kind: self.kind.clone(),
                    })
                }
            }
            BlockType::ListBlock(list) => {
                if let ElementType::ListItem = &element.kind {
                    self.bbox.merge(&element.bbox);
                    let txt = element.text_block.text.trim();
                    list.items.push(txt.to_owned());
                    Ok(())
                } else {
                    Err(FerrulesError::BlockMergeError {
                        element: Box::new(element),
                        block_id: self.id,
                        kind: self.kind.clone(),
                    })
                }
            }
            BlockType::Header(header) => {
                if let ElementType::Header = &element.kind {
                    self.bbox.merge(&element.bbox);
                    header.text.push_str(&element.text_block.text);
                    Ok(())
                } else {
                    Err(FerrulesError::BlockMergeError {
                        element: Box::new(element),
                        block_id: self.id,
                        kind: self.kind.clone(),
                    })
                }
            }
            BlockType::Footer(footer) => {
                if let ElementType::Footer = &element.kind {
                    self.bbox.merge(&element.bbox);
                    footer.text.push_str(&element.text_block.text);
                    Ok(())
                } else {
                    Err(FerrulesError::BlockMergeError {
                        element: Box::new(element),
                        block_id: self.id,
                        kind: self.kind.clone(),
                    })
                }
            }
            BlockType::Title(_title) => todo!(),
            BlockType::Image(_image_block) => todo!(),
            BlockType::Table(table) => {
                if let ElementType::Table(incoming_table_opt) = &element.kind {
                    self.bbox.merge(&element.bbox);
                    if let Some(incoming_table) = incoming_table_opt {
                        table.rows.extend(incoming_table.rows.clone());
                    }
                    Ok(())
                } else {
                    Err(FerrulesError::BlockMergeError {
                        element: Box::new(element),
                        block_id: self.id,
                        kind: self.kind.clone(),
                    })
                }
            }
        }
    }

    pub(crate) fn label(&self) -> &str {
        match self.kind {
            BlockType::Header(_) => "HEADER",
            BlockType::Footer(_) => "FOOTER",
            BlockType::TextBlock(_) => "TEXT",
            BlockType::Title(_) => "TITLE",
            BlockType::ListBlock(_) => "LIST",
            BlockType::Image(_) => "IMAGE",
            BlockType::Table(_) => "TABLE",
        }
    }
}
