use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Default, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct StepMetrics {
    pub queue_time_ms: u128,
    pub execution_time_ms: u128,
    pub idle_time_ms: u128, // Time spent waiting for resources (e.g., semaphore)
}

impl StepMetrics {
    pub(crate) fn new(execution_time_ms: u128) -> Self {
        Self {
            execution_time_ms,
            // Not applicable for native parsing it is the first step always
            queue_time_ms: 0,
            idle_time_ms: 0,
        }
    }
}

#[derive(
    Debug, Clone, Default, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct OCRMetrics {
    pub step_metrics: StepMetrics,
    pub lines_count: usize,
}

#[derive(
    Debug, Clone, Default, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct PageMetrics {
    pub page_id: usize,
    pub total_duration_ms: u128,
    pub native_step: StepMetrics,
    pub layout_step: StepMetrics,
    pub table_step: StepMetrics,
    pub ocr_step: Option<OCRMetrics>,
}

impl PageMetrics {
    #[cfg(feature = "metrics")]
    pub fn record(&self) {
        metrics::histogram!("page_processing_duration_ms").record(self.total_duration_ms as f64);

        metrics::histogram!("layout_execution_time_ms")
            .record(self.layout_step.execution_time_ms as f64);
        metrics::histogram!("layout_queue_time_ms").record(self.layout_step.queue_time_ms as f64);
        metrics::histogram!("layout_idle_time_ms").record(self.layout_step.idle_time_ms as f64);

        metrics::histogram!("native_execution_time_ms")
            .record(self.native_step.execution_time_ms as f64);

        if self.table_step.execution_time_ms > 0 {
            metrics::histogram!("table_execution_time_ms")
                .record(self.table_step.execution_time_ms as f64);
            metrics::histogram!("table_queue_time_ms").record(self.table_step.queue_time_ms as f64);
        }

        if let Some(ocr) = &self.ocr_step {
            metrics::histogram!("ocr_execution_time_ms")
                .record(ocr.step_metrics.execution_time_ms as f64);
            metrics::histogram!("ocr_idle_time_ms").record(ocr.step_metrics.idle_time_ms as f64);
        }
    }

    #[cfg(not(feature = "metrics"))]
    pub fn record(&self) {}

    pub fn record_span(&self, span: &tracing::Span) {
        span.record("layout_queue_time_ms", self.layout_step.queue_time_ms);
        span.record(
            "layout_parse_duration_ms",
            self.layout_step.execution_time_ms,
        );

        span.record(
            "parse_native_duration_ms",
            self.native_step.execution_time_ms,
        );
        if let Some(ocr_metrics) = &self.ocr_step {
            span.record("ocr_queue_time_ms", ocr_metrics.step_metrics.queue_time_ms);
            span.record(
                "ocr_parse_duration_ms",
                ocr_metrics.step_metrics.execution_time_ms,
            );
        }
        span.record("table_parse_duration_ms", self.table_step.execution_time_ms);
        span.record("table_queue_time_ms", self.table_step.queue_time_ms);
    }
}

#[derive(
    Debug, Clone, Default, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct ParsingMetrics {
    pub total_duration_ms: u128,
    pub pages: Vec<PageMetrics>,
}
