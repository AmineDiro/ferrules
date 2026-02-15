use crate::{
    entities::{BBox, Element, ElementType, PageID},
    error::FerrulesError,
};
use serde::{Deserialize, Serialize};

pub type TitleLevel = u8;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ImageBlock {
    pub(crate) id: usize,
    pub(crate) caption: Option<String>,
}

impl ImageBlock {
    pub(crate) fn path(&self) -> String {
        format!("img_{}.png", self.id)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TextBlock {
    pub(crate) text: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct List {
    pub(crate) items: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Title {
    pub level: TitleLevel,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "block_type")]
pub enum BlockType {
    Header(TextBlock),
    Footer(TextBlock),
    Title(Title),
    ListBlock(List),
    TextBlock(TextBlock),
    Image(ImageBlock),
    Table,
}

impl std::fmt::Display for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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
                        element,
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
                        element,
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
                        element,
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
                        element,
                        block_id: self.id,
                        kind: self.kind.clone(),
                    })
                }
            }
            BlockType::Title(_title) => todo!(),
            BlockType::Image(_image_block) => todo!(),
            BlockType::Table => todo!(),
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
            BlockType::Table => "TABLE",
        }
    }
}
