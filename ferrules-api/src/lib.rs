use opentelemetry::{global, trace::TracerProvider, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_semantic_conventions::resource::SERVICE_NAME;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

pub fn init_tracing(
    otlp_endpoint: Option<&str>,
    otlp_service_name: String,
    json_output: bool,
    use_sentry: bool,
) -> anyhow::Result<()> {
    global::set_text_map_propagator(opentelemetry_sdk::propagation::TraceContextPropagator::new());
    let mut layers = Vec::new();

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        // .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .with_line_number(true)
        .with_timer(tracing_subscriber::fmt::time::uptime());
    let fmt_layer = match json_output {
        true => fmt_layer.json().flatten_event(true).boxed(),
        false => fmt_layer.boxed(),
    };
    layers.push(fmt_layer);

    if let Some(otlp_endpoint) = otlp_endpoint {
        let provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_batch_exporter(
                opentelemetry_otlp::SpanExporter::builder()
                    .with_tonic()
                    .with_endpoint(otlp_endpoint)
                    .build()?,
                opentelemetry_sdk::runtime::Tokio,
            )
            .with_resource(Resource::new(vec![KeyValue::new(
                SERVICE_NAME,
                otlp_service_name,
            )]))
            .build();
        let tracer = provider.tracer("default_tracer_name");
        global::set_tracer_provider(provider);
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer).boxed();
        layers.push(otel_layer)
    }

    if use_sentry {
        layers.push(sentry_tracing::layer().boxed())
    }

    // Env filter for all
    let env_filter = EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| {
        EnvFilter::new(
            "ferrules_api=debug,ferrules_core=debug,axum_tracing_opentelemetry=info,otel=debug,opentelemetry=debug,tonic=debug,h2=info",
        )
    });
    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();

    Ok(())
}

// pub fn init_tracing(
//     otlp_endpoint: Option<&str>,
//     otlp_service_name: String,
//     json_output: bool,
// ) -> bool {
// let mut layers = Vec::new();

//     // STDOUT/STDERR layer
//     let fmt_layer = tracing_subscriber::fmt::layer()
//         .with_file(true)
//         .with_line_number(true);

//     let fmt_layer = match json_output {
//         true => tracing_subscriber::Layer::boxed(fmt_layer.json().flatten_event(true)),
//         false => tracing_subscriber::Layer::boxed(fmt_layer),
//     };
//     layers.push(fmt_layer);

//     // OpenTelemetry tracing layer
//     let mut global_tracer = false;
//     if let Some(otlp_endpoint) = otlp_endpoint {
//         global::set_text_map_propagator(
//             opentelemetry_sdk::propagation::TraceContextPropagator::new(),
//         );

//         let tracer = opentelemetry_otlp::new_pipeline()
//             .tracing()
//             .with_exporter(
//                 opentelemetry_otlp::new_exporter()
//                     .tonic()
//                     .with_endpoint(otlp_endpoint),
//             )
//             .with_trace_config(
//                 Config::default()
//                     .with_resource(Resource::new(vec![KeyValue::new(
//                         SERVICE_NAME,
//                         otlp_service_name,
//                     )]))
//                     .with_sampler(Sampler::AlwaysOn),
//             )
//             .install_batch(opentelemetry_sdk::runtime::Tokio);

//         let tracer = tracer.unwrap();
//         layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
//         global_tracer = true;
//     }

//     // Filter events with LOG_LEVEL
//     let env_filter = EnvFilter::try_from_env("LOG_LEVEL")
//         .unwrap_or_else(|_| EnvFilter::new("ferrules_api=debug,ferrules_core=debug"));

//     tracing_subscriber::registry()
//         .with(env_filter)
//         .with(layers)
//         .init();
//     global_tracer
// }
