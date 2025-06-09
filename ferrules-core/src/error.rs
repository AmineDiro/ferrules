use std::path::PathBuf;

use thiserror::Error;

use crate::{
    blocks::BlockType,
    entities::{Element, PageID},
};

#[derive(Error, Debug)]
pub enum FerrulesError {
    #[error("error occured parsing document natively")]
    ParseNativeError,
    #[error("layout parsing error")]
    LayoutParsingError,
    #[error("merging line into block error")]
    LineMergeError,
    #[error("merging elements into block error")]
    BlockMergeError {
        block_id: usize,
        kind: BlockType,
        element: Element,
    },
    #[error("saving error page number {page_idx} in :{tmp_dir:?}")]
    DebugPageError { tmp_dir: PathBuf, page_idx: PageID },

    #[error("saving error page number {page_idx} in :{tmp_dir:?}")]
    ParseTextError { tmp_dir: PathBuf, page_idx: PageID },
}
