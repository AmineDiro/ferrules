use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

use image::DynamicImage;
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{Instrument, Span};

use crate::blocks::{TableAlgorithm, TableBlock};
use crate::entities::{BBox, PDFPath, PageID};
use crate::error::FerrulesError;

pub mod lattice;
pub mod stream;
pub mod table_transformer;

use lattice::parse_table_lattice;
use stream::parse_table_stream;
pub use table_transformer::TableTransformer;

#[derive(Debug)]
pub struct TableMetadata {
    pub(crate) response_tx: oneshot::Sender<Result<ParseTableResponse, FerrulesError>>,
    pub(crate) queue_time: Instant,
}

#[derive(Debug)]
pub(crate) struct ParseTableRequest {
    pub(crate) page_id: PageID,
    pub(crate) page_image: Arc<DynamicImage>,
    pub(crate) lines: Arc<Vec<crate::entities::Line>>,
    pub(crate) paths: Arc<Vec<crate::entities::PDFPath>>,
    pub(crate) table_bbox: BBox,
    pub(crate) downscale_factor: f32,
    pub(crate) metadata: TableMetadata,
}

#[derive(Debug)]
pub(crate) struct ParseTableResponse {
    pub(crate) table_block: TableBlock,
    pub(crate) table_parse_duration_ms: u128,
    pub(crate) table_queue_time_ms: u128,
}

#[derive(Debug, Clone)]
pub struct ParseTableQueue {
    queue: mpsc::Sender<(ParseTableRequest, Span)>,
}

impl ParseTableQueue {
    pub fn new(table_parser: Arc<TableParser>) -> Self {
        let (queue_sender, queue_receiver) = mpsc::channel(16); // Buffer size

        tokio::task::spawn(start_table_parser(table_parser, queue_receiver));
        Self {
            queue: queue_sender,
        }
    }

    pub(crate) async fn push(&self, req: ParseTableRequest) -> Result<(), FerrulesError> {
        let span = Span::current();
        self.queue.send((req, span)).await.map_err(|_| {
            FerrulesError::TableParserError("Failed to push table request".to_string())
        }) // Reusing error or add specific
    }
}

async fn start_table_parser(
    table_parser: Arc<TableParser>,
    mut input_rx: mpsc::Receiver<(ParseTableRequest, Span)>,
) {
    // TODO: make this configurable
    let s = Arc::new(Semaphore::new(8));
    while let Some((req, span)) = input_rx.recv().await {
        let queue_time = req.metadata.queue_time.elapsed().as_millis();
        let page_id = req.page_id;
        tracing::debug!("table request queue time for page {page_id} took: {queue_time}ms");
        tokio::spawn(
            handle_table_request(s.clone(), table_parser.clone(), req, queue_time).instrument(span),
        );
    }
}

async fn handle_table_request(
    s: Arc<Semaphore>,
    _parser: Arc<TableParser>,
    req: ParseTableRequest,
    table_queue_time_ms: u128,
) {
    let _permit = s.acquire().await.unwrap();

    let ParseTableRequest {
        page_id,
        page_image,
        lines,
        paths,
        table_bbox,
        downscale_factor,
        metadata,
    } = req;

    let parser = _parser.clone();
    let lines = lines.clone();
    let paths = paths.clone();
    let page_image = page_image.clone();
    let table_bbox = table_bbox.clone();

    let start = Instant::now();
    let table_result = parser
        .parse(
            page_id,
            &lines,
            &paths,
            &table_bbox,
            &page_image,
            downscale_factor,
        )
        .await;
    let inference_duration = start.elapsed().as_millis();
    tracing::debug!("table inference time for page {page_id} took: {inference_duration} ms");

    drop(_permit);

    let response = table_result.map(|t| ParseTableResponse {
        table_block: t,
        table_parse_duration_ms: inference_duration,
        table_queue_time_ms,
    });

    let _ = metadata.response_tx.send(response);
}

#[derive(Clone)]
pub struct TableParser {
    transformer: Option<TableTransformer>,
    table_id_counter: Arc<AtomicUsize>,
}

impl TableParser {
    /// Minimum cell count below which a table is suspicious (when paired with a large area).
    const MIN_CELL_COUNT: usize = 2;
    /// Area threshold used when the cell count is low.
    const SUSPICIOUS_AREA_LOW_CELLS: f32 = 1500.0;
    /// Minimum row count below which a table is suspicious (when paired with a large area).
    const MIN_ROW_COUNT: usize = 1;
    /// Area threshold used when the row count is low.
    const SUSPICIOUS_AREA_LOW_ROWS: f32 = 3000.0;
    /// Large-area threshold: tables above this always try vision.
    const LARGE_AREA_THRESHOLD: f32 = 5000.0;
    /// Minimum ratio of total cell area to table area.
    /// Below this the stream result is considered incomplete.
    const CELL_COVERAGE_THRESHOLD: f32 = 0.3;

    pub fn new(transformer: Option<TableTransformer>) -> Self {
        Self {
            transformer,
            table_id_counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Heuristic to decide whether the Vision (Table Transformer) fallback
    /// should be attempted after a Stream parse.
    ///
    /// Returns `true` when:
    /// - Stream found **no rows** at all.
    /// - Few cells in a suspiciously large area.
    /// - Few rows in a suspiciously large area.
    /// - The table area exceeds `LARGE_AREA_THRESHOLD`.
    /// - The total cell area covers less than `CELL_COVERAGE_THRESHOLD` of the table area.
    fn should_try_vision(&self, table: &TableBlock, table_area: f32) -> bool {
        let row_count = table.rows.len();
        if row_count == 0 {
            return true;
        }

        let cell_count: usize = table.rows.iter().map(|r| r.cells.len()).sum();

        let is_suspicious = (cell_count <= Self::MIN_CELL_COUNT
            && table_area > Self::SUSPICIOUS_AREA_LOW_CELLS)
            || (row_count <= Self::MIN_ROW_COUNT && table_area > Self::SUSPICIOUS_AREA_LOW_ROWS);

        if is_suspicious || table_area > Self::LARGE_AREA_THRESHOLD {
            return true;
        }

        // Check whether the detected cells actually cover a reasonable
        // fraction of the table bounding box. A low ratio means stream
        // parsing likely missed significant content.
        if table_area > 0.0 {
            let total_cell_area: f32 = table
                .rows
                .iter()
                .flat_map(|r| &r.cells)
                .map(|c| c.bbox.area())
                .sum();
            if total_cell_area / table_area < Self::CELL_COVERAGE_THRESHOLD {
                return true;
            }
        }

        false
    }

    #[tracing::instrument(name="table_parse",skip(self, lines, paths, page_image), fields(page_id = %page_id, table_bbox = ?table_bbox))]
    pub async fn parse(
        &self,
        page_id: PageID,
        lines: &[crate::entities::Line],
        paths: &[PDFPath],
        table_bbox: &BBox,
        page_image: &DynamicImage,
        downscale_factor: f32,
    ) -> Result<TableBlock, FerrulesError> {
        // TODO: Decide between Lattice and Stream
        if !paths.is_empty() {
            let span = tracing::debug_span!("lattice_attempt", paths_count = paths.len());
            let lines_vec = lines.to_vec();
            let paths_vec = paths.to_vec();
            let table_bbox_clone = table_bbox.clone();
            let counter = self.table_id_counter.clone();

            let table = async move {
                tracing::debug!(
                    "Page {} - BBox {:?} - {} paths. Trying Lattice...",
                    page_id,
                    table_bbox_clone,
                    paths_vec.len()
                );
                tokio::task::spawn_blocking(move || {
                    parse_table_lattice(counter, &lines_vec, &paths_vec, &table_bbox_clone)
                })
                .await
                .map_err(|_| FerrulesError::TableParserError("Failed to parse table".to_string()))
            }
            .instrument(span)
            .await?;

            if let Some(mut table) = table {
                table.algorithm = TableAlgorithm::Lattice;
                tracing::debug!("Page {} - Lattice successful.", page_id);
                return Ok(table);
            }
            tracing::debug!("Page {} - Lattice failed.", page_id);
        } else {
            tracing::debug!("Page {} has no paths. Skipping Lattice.", page_id);
        }

        let mut table = {
            let span = tracing::debug_span!("stream_attempt");
            let lines_vec = lines.to_vec();
            let table_bbox_clone = table_bbox.clone();
            let counter = self.table_id_counter.clone();
            async move {
                tokio::task::spawn_blocking(move || {
                    parse_table_stream(counter, &lines_vec, &table_bbox_clone)
                })
                .await
                .map_err(|_| FerrulesError::TableParserError("Failed to parse table".to_string()))?
            }
            .instrument(span)
            .await?
        };

        table.algorithm = TableAlgorithm::Stream;

        let area = table_bbox.area();
        let cell_count: usize = table.rows.iter().map(|r| r.cells.len()).sum();
        let row_count = table.rows.len();

        if self.should_try_vision(&table, area) {
            let span = tracing::debug_span!(
                "vision_attempt",
                stream_cells = cell_count,
                stream_rows = row_count
            );
            let vision_result = async {
                tracing::debug!(
                    "Page {} - Stream suspicious ({} cells, {} rows). Trying Vision comparison...",
                    page_id,
                    cell_count,
                    row_count
                );

                if let Some(transformer) = &self.transformer {
                    if let Ok(vision_table) = transformer
                        .parse_table_transformer(
                            &self.table_id_counter,
                            page_image,
                            lines,
                            table_bbox,
                            downscale_factor,
                        )
                        .await
                    {
                        let vision_cell_count: usize =
                            vision_table.rows.iter().map(|r| r.cells.len()).sum();

                        // Pick vision if it found significantly more cells OR if stream was empty
                        if vision_cell_count > cell_count
                            || (cell_count == 0 && !vision_table.rows.is_empty())
                        {
                            tracing::debug!(
                                "Page {} - Vision ({} cells) preferred over Stream ({} cells).",
                                page_id,
                                vision_cell_count,
                                cell_count
                            );
                            return Ok(Some(vision_table));
                        }
                    }
                }
                Ok(None)
            }
            .instrument(span)
            .await?;

            if let Some(vision_table) = vision_result {
                return Ok(vision_table);
            }
        }

        tracing::debug!(
            "Page {} - Stream successful ({} cells).",
            page_id,
            cell_count
        );
        Ok(table)
    }
}
