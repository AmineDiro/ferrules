use std::{
    ops::Range,
    path::{Path, PathBuf},
    time::Instant,
};
use tracing::Instrument;

use pdfium_render::prelude::Pdfium;
use tokio::{sync::mpsc, task::JoinSet};
use tracing::instrument;

use crate::{
    entities::{DocumentMetadata, Page, PageID, ParsedDocument, StructuredPage},
    layout::ParseLayoutQueue,
};

use super::{
    merge::merge_elements_into_blocks,
    native::{ParseNativePageResult, ParseNativeQueue, ParseNativeRequest},
    page::parse_page_full,
    titles::title_levels_kmeans,
};

async fn parse_task<F>(
    parse_native_result: ParseNativePageResult,
    debug_dir: Option<PathBuf>,
    layout_queue: ParseLayoutQueue,
    callback: Option<F>,
) -> anyhow::Result<StructuredPage>
where
    F: FnOnce(PageID) + Send + 'static + Clone,
{
    let page_id = parse_native_result.page_id;

    let result = parse_page_full(parse_native_result, debug_dir, layout_queue).await;
    if let Some(callback) = callback {
        callback(page_id)
    }
    result
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
async fn parse_doc_pages<F>(
    data: &[u8],
    flatten_pdf: bool,
    password: Option<&str>,
    page_range: Option<Range<usize>>,
    debug_dir: Option<PathBuf>,
    layout_queue: ParseLayoutQueue,
    native_queue: ParseNativeQueue,
    callback: Option<F>,
) -> anyhow::Result<Vec<StructuredPage>>
where
    // TODO: callback on function result
    F: FnOnce(PageID) + Send + 'static + Clone,
{
    let mut set = JoinSet::new();
    let (native_tx, mut native_rx) = mpsc::channel(32);
    let req = ParseNativeRequest::new(data, password, flatten_pdf, page_range, native_tx);
    native_queue.push(req).await?;

    while let Some(native_page) = native_rx.recv().await {
        match native_page {
            Ok(parse_native_result) => {
                let layout_queue = layout_queue.clone();
                let tmp_dir = debug_dir.clone();
                let callback = callback.clone();
                set.spawn(
                    parse_task(parse_native_result, tmp_dir, layout_queue, callback)
                        .in_current_span(),
                );
            }
            Err(_) => todo!(),
        }
    }

    // Get results
    let mut parsed_pages = Vec::new();
    while let Some(result) = set.join_next().await {
        match result {
            Ok(Ok(page)) => {
                parsed_pages.push(page);
            }
            Ok(Err(e)) => {
                tracing::error!("Error parsing page : {e:?}")
            }
            Err(e) => {
                tracing::error!("Error Joining : {e:?}")
            }
        }
    }
    parsed_pages.sort_by(|p1, p2| p1.id.cmp(&p2.id));
    Ok(parsed_pages)
}

pub fn get_doc_length<P: AsRef<Path>>(
    path: P,
    password: Option<&str>,
    page_range: Option<Range<usize>>,
) -> anyhow::Result<usize> {
    // TODO : This panic ! should be handlered
    let pdfium = Pdfium::new(Pdfium::bind_to_statically_linked_library().unwrap());
    let document = pdfium.load_pdf_from_file(&path, password).unwrap();
    let pages: Vec<_> = document.pages().iter().enumerate().collect();
    match page_range {
        Some(range) => {
            if range.end > pages.len() {
                anyhow::bail!(
                    "Page range end ({}) exceeds document length ({})",
                    range.end,
                    pages.len()
                );
            }
            Ok(range.len())
        }
        None => Ok(pages.len()),
    }
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip(doc, password, layout_queue, native_queue, page_callback, debug_dir))]
pub async fn parse_document<F>(
    doc: &[u8],
    doc_name: String,
    password: Option<&str>,
    flatten_pdf: bool,
    page_range: Option<Range<usize>>,
    layout_queue: ParseLayoutQueue,
    native_queue: ParseNativeQueue,
    debug_dir: Option<PathBuf>,
    page_callback: Option<F>,
) -> anyhow::Result<ParsedDocument>
where
    F: FnOnce(PageID) + Send + 'static + Clone,
{
    let start_time = Instant::now();
    let parsed_pages = parse_doc_pages(
        doc,
        flatten_pdf,
        password,
        page_range,
        debug_dir.clone(),
        layout_queue,
        native_queue,
        page_callback,
    )
    .await?;

    let all_elements = parsed_pages
        .iter()
        .flat_map(|p| p.elements.clone())
        .collect::<Vec<_>>();

    let titles = all_elements
        .iter()
        .filter(|e| {
            matches!(
                e.kind,
                crate::entities::ElementType::Title | crate::entities::ElementType::Subtitle
            )
        })
        .collect::<Vec<_>>();

    let title_level = title_levels_kmeans(&titles, 6);

    let doc_pages = parsed_pages
        .into_iter()
        .map(|sp| Page {
            id: sp.id,
            width: sp.width,
            height: sp.height,
            need_ocr: sp.need_ocr,
            image: sp.image,
        })
        .collect();

    let blocks = merge_elements_into_blocks(all_elements, title_level)?;

    let duration = start_time.elapsed();

    Ok(ParsedDocument {
        doc_name,
        pages: doc_pages,
        blocks,
        debug_path: debug_dir,
        metadata: DocumentMetadata::new(duration),
    })
}
