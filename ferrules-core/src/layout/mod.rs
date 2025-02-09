use std::{sync::Arc, time::Instant};

use anyhow::Context;
use image::DynamicImage;
use model::{LayoutBBox, ORTLayoutParser};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::{oneshot, Semaphore};
use tracing::{Instrument, Span};

use crate::entities::PageID;

pub mod model;
// mod infer;

#[derive(Debug)]
pub struct Metadata {
    pub(crate) response_tx: oneshot::Sender<anyhow::Result<ParseLayoutResponse>>,
    pub(crate) queue_time: Instant,
}

#[derive(Debug)]
pub(crate) struct ParseLayoutRequest {
    pub(crate) page_id: PageID,
    pub(crate) page_image: Arc<DynamicImage>,
    pub(crate) downscale_factor: f32,
    pub(crate) metadata: Metadata,
}

#[derive(Debug)]
pub(crate) struct ParseLayoutResponse {
    pub(crate) page_id: PageID,
    pub(crate) layout_bbox: Vec<LayoutBBox>,
    pub(crate) layout_parse_duration_ms: u128,
    pub(crate) layout_queue_time_ms: u128,
}

#[derive(Debug, Clone)]
pub struct ParseLayoutQueue {
    queue: Sender<(ParseLayoutRequest, Span)>,
}

impl ParseLayoutQueue {
    pub fn new(layout_parser: Arc<ORTLayoutParser>) -> Self {
        let (queue_sender, queue_receiver) = mpsc::channel(layout_parser.config.intra_threads);

        tokio::task::spawn(start_layout_parser(layout_parser, queue_receiver));
        Self {
            queue: queue_sender,
        }
    }

    pub(crate) async fn push(&self, req: ParseLayoutRequest) -> anyhow::Result<()> {
        let span = Span::current();
        self.queue
            .send((req, span))
            .await
            .context("error sending  parse req")
    }
}

async fn start_layout_parser(
    layout_parser: Arc<ORTLayoutParser>,
    mut input_rx: Receiver<(ParseLayoutRequest, Span)>,
) {
    let s = Arc::new(Semaphore::new(layout_parser.config.intra_threads));
    while let Some((req, span)) = input_rx.recv().await {
        let queue_time = req.metadata.queue_time.elapsed().as_millis();
        let page_id = req.page_id;
        tracing::debug!("layout request queue time for page {page_id} took: {queue_time}ms");
        let _guard = span.enter();
        tokio::spawn(
            handle_request(s.clone(), layout_parser.clone(), req, queue_time).in_current_span(),
        );
    }
}

async fn handle_request(
    s: Arc<Semaphore>,
    parser: Arc<ORTLayoutParser>,
    req: ParseLayoutRequest,
    layout_queue_time_ms: u128,
) {
    let _permit = s.acquire().await.unwrap();

    let ParseLayoutRequest {
        page_id,
        page_image,
        downscale_factor,
        metadata,
    } = req;

    let start = Instant::now();
    let layout_result = parser
        .parse_layout_async(&page_image, downscale_factor)
        .await;
    let inference_duration = start.elapsed().as_millis();
    drop(_permit);
    tracing::debug!("layout inference time for page {page_id} took: {inference_duration} ms");

    let layout_result = layout_result.map(|l| ParseLayoutResponse {
        page_id,
        layout_bbox: l,
        layout_parse_duration_ms: inference_duration,
        layout_queue_time_ms,
    });
    metadata
        .response_tx
        .send(layout_result)
        .expect("can't send parsed result over oneshot chan");
}
