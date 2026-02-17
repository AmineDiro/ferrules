use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

use crate::blocks::TableAlgorithm;

#[derive(
    Debug, Clone, Default, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct StepMetrics {
    pub queue_time_ms: f64,
    pub execution_time_ms: f64,
    pub idle_time_ms: f64, // Time spent waiting for resources (e.g., semaphore)
}

impl StepMetrics {
    pub(crate) fn new(execution_time_ms: f64) -> Self {
        Self {
            execution_time_ms,
            // Not applicable for native parsing it is the first step always
            queue_time_ms: 0.0,
            idle_time_ms: 0.0,
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
pub struct TableMetrics {
    pub step_metrics: StepMetrics,
    pub algorithm: TableAlgorithm,
}

#[derive(
    Debug, Clone, Default, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct PageMetrics {
    pub page_id: usize,
    pub total_duration_ms: f64,
    pub native_step: StepMetrics,
    pub layout_step: StepMetrics,
    pub table_steps: Vec<TableMetrics>,
    pub ocr_step: Option<OCRMetrics>,
}

impl PageMetrics {
    #[cfg(feature = "metrics")]
    pub fn record(&self) {
        let ocr_label = if self.ocr_step.is_some() {
            "true"
        } else {
            "false"
        };
        metrics::histogram!("page_processing_duration_ms", "ocr" => ocr_label)
            .record(self.total_duration_ms as f64);

        metrics::histogram!("layout_execution_time_ms")
            .record(self.layout_step.execution_time_ms as f64);
        metrics::histogram!("layout_queue_time_ms").record(self.layout_step.queue_time_ms as f64);
        metrics::histogram!("layout_idle_time_ms").record(self.layout_step.idle_time_ms as f64);

        metrics::histogram!("native_execution_time_ms")
            .record(self.native_step.execution_time_ms as f64);

        for table in &self.table_steps {
            let algo_str = match table.algorithm {
                TableAlgorithm::Lattice => "lattice",
                TableAlgorithm::Stream => "stream",
                TableAlgorithm::Vision => "vision",
                TableAlgorithm::Unknown => "unknown",
            };

            metrics::histogram!("table_execution_time_ms", "method" => algo_str)
                .record(table.step_metrics.execution_time_ms as f64);
            metrics::histogram!("table_queue_time_ms", "method" => algo_str)
                .record(table.step_metrics.queue_time_ms as f64);

            // Still record global table metrics if needed, or just let prometheus aggregate
            metrics::histogram!("table_execution_time_ms")
                .record(table.step_metrics.execution_time_ms as f64);
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
        span.record("layout_idle_time_ms", self.layout_step.idle_time_ms);
        span.record(
            "layout_parse_duration_ms",
            self.layout_step.execution_time_ms,
        );

        span.record(
            "parse_native_duration_ms",
            self.native_step.execution_time_ms,
        );
        if let Some(ocr_metrics) = &self.ocr_step {
            span.record("ocr_idle_time_ms", ocr_metrics.step_metrics.idle_time_ms);
            span.record(
                "ocr_parse_duration_ms",
                ocr_metrics.step_metrics.execution_time_ms,
            );
        }

        let total_table_duration: f64 = self
            .table_steps
            .iter()
            .map(|t| t.step_metrics.execution_time_ms)
            .sum();
        let total_table_queue: f64 = self
            .table_steps
            .iter()
            .map(|t| t.step_metrics.queue_time_ms)
            .sum();

        span.record("table_parse_duration_ms", total_table_duration);
        span.record("table_queue_time_ms", total_table_queue);
    }
}

#[derive(
    Debug, Clone, Default, Serialize, Deserialize, Archive, RkyvDeserialize, RkyvSerialize,
)]
pub struct ParsingMetrics {
    pub total_duration_ms: f64,
    pub pages: Vec<PageMetrics>,
}
