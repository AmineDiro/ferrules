use std::{sync::Arc, time::Instant};

use anyhow::Context;
use image::DynamicImage;
use model::{LayoutBBox, ORTLayoutParser};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::{oneshot, Mutex, Notify, Semaphore};
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
pub struct ParseLayoutRequest {
    pub page_id: PageID,
    pub page_image: Arc<DynamicImage>,
    pub downscale_factor: f32,
    pub metadata: Metadata,
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
    notifier: Arc<Notify>,
}

impl ParseLayoutQueue {
    pub fn new(ort_config: model::ORTConfig, n_parser: usize) -> Self {
        let (queue_sender, queue_receiver) = mpsc::channel(ort_config.intra_threads);

        let queue_receiver = Arc::new(Mutex::new(queue_receiver));
        let notifier = Arc::new(Notify::new());
        for _ in 0..n_parser {
            let layout_model = Arc::new(
                ORTLayoutParser::new(ort_config.clone()).expect("Failed to load layout model"),
            );
            tokio::task::spawn(start_layout_parser(
                layout_model.clone(),
                queue_receiver.clone(),
                notifier.clone(),
            ));
        }
        Self {
            queue: queue_sender,
            notifier,
        }
    }

    pub(crate) async fn push(&self, req: ParseLayoutRequest) -> anyhow::Result<()> {
        let span = Span::current();
        self.queue
            .send((req, span))
            .await
            .context("error sending  parse req")?;
        self.notifier.notify_one();
        Ok(())
    }
}

async fn start_layout_parser(
    layout_parser: Arc<ORTLayoutParser>,
    input_rx: Arc<Mutex<Receiver<(ParseLayoutRequest, Span)>>>,
    notify: Arc<Notify>,
) {
    let s = Arc::new(Semaphore::new(layout_parser.config.intra_threads));

    loop {
        let next_message = {
            let mut lock = input_rx.lock().await;
            lock.recv().await
        };

        if let Some((req, span)) = next_message {
            let queue_time = req.metadata.queue_time.elapsed().as_millis();
            let page_id = req.page_id;
            tracing::debug!("layout request queue time for page {page_id} took: {queue_time}ms");
            let _guard = span.enter();
            tokio::spawn(
                handle_request(s.clone(), layout_parser.clone(), req, queue_time).in_current_span(),
            );
        }
        notify.notified().await;
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
