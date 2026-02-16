use std::{path::PathBuf, sync::Arc, time::Instant};

use std::ops::Range;

use tokio::{sync::mpsc, task::JoinSet};
use tracing::Instrument;

use super::native::{ParseNativeQueue, ParseNativeRequest};
use super::{
    merge::merge_elements_into_blocks, native::ParseNativePageResult, page::parse_page_full,
    titles::title_levels_kmeans,
};
use crate::entities::DocumentMetadata;
use crate::error::FerrulesError;
use crate::{
    blocks::Block,
    entities::{ElementType, Page, PageID, ParsedDocument, StructuredPage},
    layout::{
        model::{ORTConfig, ORTLayoutParser},
        ParseLayoutQueue,
    },
    parse::table::{ParseTableQueue, TableParser, TableTransformer},
};

/// Configuration options for parsing documents with FerrulesParser
#[derive(Debug, Clone)]
pub struct FerrulesParseConfig<'a> {
    /// Optional password for encrypted PDF documents
    pub password: Option<&'a str>,

    /// Whether to flatten the PDF before parsing. When true, flattens form fields and annotations
    /// into the document content for more consistent parsing results
    pub flatten_pdf: bool,

    /// Optional range of pages to parse. When None, parses all pages
    /// The range uses 0-based indexing (e.g., 0..5 parses first 5 pages)
    pub page_range: Option<std::ops::Range<usize>>,

    /// Optional directory path for debug output. When provided, saves intermediate parsing
    /// results and visualizations to this directory
    pub debug_dir: Option<std::path::PathBuf>,
}

impl Default for FerrulesParseConfig<'_> {
    fn default() -> Self {
        Self {
            password: None,
            flatten_pdf: true,
            page_range: None,
            debug_dir: None,
        }
    }
}

async fn parse_task<F>(
    parse_native_result: ParseNativePageResult,
    layout_queue: ParseLayoutQueue,
    table_queue: ParseTableQueue,
    debug_dir: Option<PathBuf>,
    callback: Option<F>,
) -> Result<StructuredPage, FerrulesError>
where
    F: FnOnce(PageID) + Send + 'static + Clone,
{
    let page_id = parse_native_result.page_id;

    let result = parse_page_full(
        parse_native_result,
        debug_dir,
        layout_queue.clone(),
        table_queue.clone(),
    )
    .await;
    if let Some(callback) = callback {
        callback(page_id)
    }
    result
}

/// Core class Document parser that extracts structured content from PDF documents.
///
/// FerrulesParser uses a combination of native PDF parsing and machine learning-based
/// layout analysis to extract text, structural elements, and content hierarchies from documents.
#[derive(Clone)]
pub struct FerrulesParser {
    layout_queue: ParseLayoutQueue,
    native_queue: ParseNativeQueue,
    table_queue: ParseTableQueue,
}

impl FerrulesParser {
    /// Creates a new FerrulesParser instance with the specified layout model configuration
    ///
    /// # Arguments
    /// * `layout_config` - Configuration for the ONNX Runtime layout analysis model
    ///
    /// # Returns
    /// A new FerrulesParser instance
    ///
    /// # Panics
    /// Panics if the layout model cannot be loaded with the given configuration
    pub fn new(layout_config: ORTConfig) -> Self {
        let layout_model =
            Arc::new(ORTLayoutParser::new(layout_config.clone()).expect("can't load layout model"));
        let native_queue = ParseNativeQueue::new();
        let layout_queue = ParseLayoutQueue::new(layout_model);
        let transformer = TableTransformer::new(&layout_config).ok();
        let table_parser = Arc::new(TableParser::new(transformer));
        let table_queue = ParseTableQueue::new(table_parser);
        Self {
            layout_queue,
            native_queue,
            table_queue,
        }
    }
    /// Parses a document into a structured format with optional page-level progress callback
    ///
    /// # Arguments
    /// * `doc` - Raw bytes of the document to parse
    /// * `doc_name` - Name of the document
    /// * `config` - Parsing configuration options
    /// * `page_callback` - Optional callback function called after each page is processed
    ///
    /// # Returns
    /// A Result containing the parsed document structure or an error
    ///
    /// # Examples
    /// ```no_run
    /// use ferrules_core::{FerrulesParser, FerrulesParseConfig, layout::model::ORTConfig};
    ///
    /// async fn parse() {
    ///     let parser = FerrulesParser::new(ORTConfig::default());
    ///     let config = FerrulesParseConfig::default();
    ///
    ///     let doc_bytes = std::fs::read("document.pdf").unwrap();
    ///     let parsed = parser.parse_document(
    ///         &doc_bytes,
    ///         "document.pdf".to_string(),
    ///         config,
    ///         Some(|page_id| println!("Parsed page {}", page_id))
    ///     ).await.unwrap();
    /// }
    #[allow(clippy::too_many_arguments)]
    pub async fn parse_document<F>(
        &self,
        doc: &[u8],
        doc_name: String,
        config: FerrulesParseConfig<'_>,
        page_callback: Option<F>,
    ) -> Result<ParsedDocument, FerrulesError>
    where
        F: FnOnce(PageID) + Send + 'static + Clone,
    {
        let FerrulesParseConfig {
            password,
            flatten_pdf,
            page_range,
            debug_dir,
        } = config;
        let start_time = Instant::now();
        let parsed_pages = self
            .parse_doc_pages(
                doc,
                flatten_pdf,
                password,
                page_range,
                debug_dir.clone(),
                page_callback,
            )
            .await?;

        let all_elements = parsed_pages
            .iter()
            .flat_map(|p| p.elements.clone())
            .collect::<Vec<_>>();

        let titles = all_elements
            .iter()
            .filter(|e| matches!(e.kind, ElementType::Title | ElementType::Subtitle))
            .collect::<Vec<_>>();

        let title_level = title_levels_kmeans(&titles, 6);

        let doc_pages = parsed_pages
            .iter()
            .map(|sp| Page {
                id: sp.id,
                width: sp.width,
                height: sp.height,
                need_ocr: sp.need_ocr,
                image: sp.image.clone(),
            })
            .collect();

        let blocks = merge_elements_into_blocks(all_elements, title_level)?;

        if let Some(ref debug_dir) = debug_dir {
            self.save_debug_binary(debug_dir, &doc_name, &parsed_pages, &blocks);
        }

        let duration = start_time.elapsed();

        Ok(ParsedDocument {
            doc_name,
            pages: doc_pages,
            blocks,
            debug_path: debug_dir,
            metadata: DocumentMetadata::new(duration),
        })
    }

    fn save_debug_binary(
        &self,
        debug_dir: &std::path::Path,
        doc_name: &str,
        parsed_pages: &[StructuredPage],
        blocks: &[Block],
    ) {
        let mut debug_pages = Vec::new();
        for sp in parsed_pages {
            let mut page_blocks = Vec::new();
            for block in blocks {
                if block.pages_id.contains(&sp.id) {
                    page_blocks.push(block.clone());
                }
            }

            let mut image_data = Vec::new();
            let _ = sp.image.write_to(
                &mut std::io::Cursor::new(&mut image_data),
                image::ImageFormat::Png,
            );

            debug_pages.push(crate::debug_info::DebugPage {
                page_number: sp.id,
                native_lines: sp.native_lines.clone(),
                paths: sp.paths.clone(),
                layout_bboxes: sp.layout.clone(),
                ocr_lines: sp.ocr_lines.clone(),
                elements: sp.elements.clone(),
                blocks: page_blocks,
                image_data,
                width: sp.width,
                height: sp.height,
            });
        }
        let debug_doc = crate::debug_info::DebugDocument {
            name: doc_name.to_string(),
            pages: debug_pages,
        };

        let debug_file = debug_dir.join(format!("{}.ferr", doc_name));
        let bytes = rkyv::to_bytes::<_, 1024>(&debug_doc).expect("failed to serialize debug doc");
        std::fs::write(debug_file, bytes).expect("failed to write debug file");
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all)]
    async fn parse_doc_pages<F>(
        &self,
        data: &[u8],
        flatten_pdf: bool,
        password: Option<&str>,
        page_range: Option<Range<usize>>,
        debug_dir: Option<PathBuf>,
        callback: Option<F>,
    ) -> Result<Vec<StructuredPage>, FerrulesError>
    where
        F: FnOnce(PageID) + Send + 'static + Clone,
    {
        let mut set = JoinSet::new();
        let (native_tx, mut native_rx) = mpsc::channel(32);
        let req = ParseNativeRequest::new(data, password, flatten_pdf, page_range, native_tx);
        self.native_queue.push(req).await?;

        while let Some(native_page) = native_rx.recv().await {
            match native_page {
                Ok(parse_native_result) => {
                    let tmp_dir = debug_dir.clone();
                    let callback = callback.clone();
                    set.spawn(
                        parse_task(
                            parse_native_result,
                            self.layout_queue.clone(),
                            self.table_queue.clone(),
                            tmp_dir,
                            callback,
                        )
                        .in_current_span(),
                    );
                }
                Err(_) => eprintln!("Error occured parsing page in doc"),
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
}
