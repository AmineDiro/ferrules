use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::{
        header::{ACCEPT, CONTENT_TYPE},
        HeaderMap, Response, StatusCode,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;
use clap::Parser;
use ferrules_api::init_tracing;
use ferrules_core::{
    layout::model::{ORTConfig, OrtExecutionProvider},
    render::markdown::to_markdown,
    FerrulesParseConfig, FerrulesParser,
};
use memmap2::Mmap;
use mimalloc::MiMalloc;
use serde::{Deserialize, Serialize};
use std::io::{Seek, Write};
use tempfile::NamedTempFile;
use tokio::{fs::File, net::TcpListener};
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const MAX_SIZE_LIMIT: usize = 250 * 1024 * 1024;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// OpenTelemetry collector endpoint
    #[arg(long, env = "OTLP_ENDPOINT")]
    otlp_endpoint: Option<String>,

    /// Sentry DSN
    #[arg(long, env = "SENTRY_DSN")]
    sentry_dsn: Option<String>,

    /// Sentry environment
    #[arg(long, env = "SENTRY_ENVIRONMENT", default_value = "dev")]
    sentry_environment: String,

    /// API listen address
    #[arg(long, env = "API_LISTEN_ADDR", default_value = "0.0.0.0:3002")]
    listen_addr: String,

    /// Enable debug mode
    #[arg(long, env = "SENTRY_DEBUG", default_value = "false")]
    sentry_debug: bool,

    /// Use CoreML for layout inference (default: true)
    #[arg(
            long,
            default_value_t = cfg!(target_os = "macos"),
            help = "Enable or disable the use of CoreML for layout inference"
        )]
    pub coreml: bool,

    #[arg(
        long,
        default_value_t = true,
        help = "Enable or disable Apple Neural Engine acceleration (only applies when CoreML is enabled)"
    )]
    pub use_ane: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Enable or disable the use of TensorRT for layout inference"
    )]
    pub trt: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Enable or disable the use of CUDA for layout inference"
    )]
    pub cuda: bool,

    /// CUDA device ID to use for GPU acceleration (e.g. 0 for first GPU)
    #[arg(
        long,
        help = "CUDA device ID to use (0 for first GPU)",
        default_value_t = 0
    )]
    pub device_id: i32,

    /// Number of threads to use within individual operations
    #[arg(
        long,
        short = 'j',
        help = "Number of threads to use for parallel processing within operations",
        default_value = "16"
    )]
    intra_threads: usize,

    /// Number of threads to use for parallel operation execution
    #[arg(
        long,
        help = "Number of threads to use for executing operations in parallel",
        default_value = "4"
    )]
    inter_threads: usize,

    #[arg(long, short = 'O', help = "Ort graph optimization level")]
    graph_opt_level: Option<usize>,
}

fn parse_ep_args(args: &Args) -> Vec<OrtExecutionProvider> {
    let mut providers = Vec::new();
    if args.trt {
        providers.push(OrtExecutionProvider::Trt(args.device_id));
    }
    if args.cuda {
        providers.push(OrtExecutionProvider::CUDA(args.device_id));
    }

    if args.coreml {
        providers.push(OrtExecutionProvider::CoreML {
            ane_only: args.use_ane,
        });
    }
    providers.push(OrtExecutionProvider::CPU);
    providers
}

#[derive(Debug, Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ParseOptions {
    page_range: Option<String>,
    _save_images: Option<bool>,
}

#[derive(Clone)]
struct AppState {
    parser: FerrulesParser,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Check providers
    let providers = parse_ep_args(&args);

    // Initialize Sentry if DSN is provided
    let use_sentry = args.sentry_dsn.is_some();
    let _guard = if let Some(dsn) = args.sentry_dsn {
        Some(sentry::init((
            dsn,
            sentry::ClientOptions {
                release: sentry::release_name!(),
                traces_sample_rate: 1f32,
                sample_rate: 1f32,
                environment: Some(args.sentry_environment.into()),
                ..Default::default()
            },
        )))
    } else {
        None
    };

    init_tracing(
        args.otlp_endpoint.as_deref(),
        "ferrules-api".into(),
        false,
        use_sentry,
    )
    .expect("can't setup tracing for API");

    // Initialize Prometheus exporter
    let builder = metrics_exporter_prometheus::PrometheusBuilder::new()
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Suffix("_ms".to_string()),
            &[
                0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
                90.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0,
                600.0, 700.0, 800.0, 900.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0, 2500.0, 3000.0,
                3500.0, 4000.0, 4500.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 15000.0,
                20000.0, 30000.0, 45000.0, 60000.0, 90000.0, 120000.0, 180000.0, 240000.0,
                300000.0,
            ],
        )
        .expect("failed to set buckets");
    let handle = builder
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    let ort_config = ORTConfig {
        execution_providers: providers,
        intra_threads: args.intra_threads,
        inter_threads: args.inter_threads,
        opt_level: args.graph_opt_level.map(|v| v.try_into().unwrap()),
        warmup: true,
    };
    // Initialize the layout model and queues
    let parser = FerrulesParser::new(ort_config);

    let app_state = AppState { parser };

    // Build our application with a route
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/parse", post(parse_document_handler))
        .route("/metrics", get(move || std::future::ready(handle.render())))
        .with_state(app_state)
        .layer(OtelAxumLayer::default())
        .layer(DefaultBodyLimit::max(MAX_SIZE_LIMIT));

    // Run it
    let listener = TcpListener::bind("0.0.0.0:3002").await.unwrap();
    tracing::info!(
        "Starting ferrules service listening on {}",
        listener.local_addr().unwrap()
    );
    axum::serve(listener, app).await.unwrap();
}

#[tracing::instrument(skip_all)]
async fn health_check() -> impl IntoResponse {
    Json(ApiResponse {
        success: true,
        data: Some("Service is healthy"),
        error: None,
    })
}

#[tracing::instrument(skip_all)]
async fn parse_document_handler(
    headers: HeaderMap,
    state: State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiResponse<()>>)> {
    // Extract the file from multipart form

    let mut temp_file = NamedTempFile::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Failed to create temp file: {}", e)),
            }),
        )
    })?;

    let mut options = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Failed to get next field: {}", e)),
            }),
        )
    })? {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                // Stream the field data to the temp file
                let mut field_stream = field;
                while let Some(chunk) = field_stream.chunk().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ApiResponse {
                            success: false,
                            data: None,
                            error: Some(format!("Failed to read chunk: {}", e)),
                        }),
                    )
                })? {
                    temp_file.write_all(&chunk).map_err(|e| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(ApiResponse {
                                success: false,
                                data: None,
                                error: Some(format!("Failed to write to temp file: {}", e)),
                            }),
                        )
                    })?;
                }
                temp_file.flush().map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ApiResponse {
                            success: false,
                            data: None,
                            error: Some(format!("Failed to flush temp file: {}", e)),
                        }),
                    )
                })?;
                temp_file.seek(std::io::SeekFrom::Start(0)).map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ApiResponse {
                            success: false,
                            data: None,
                            error: Some(format!("Failed to seek temp file: {}", e)),
                        }),
                    )
                })?;
            }
            "options" => {
                let options_str = field.text().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ApiResponse {
                            success: false,
                            data: None,
                            error: Some(format!("Failed to read options: {}", e)),
                        }),
                    )
                })?;
                options = Some(serde_json::from_str::<ParseOptions>(&options_str).map_err(
                    |e| {
                        (
                            StatusCode::BAD_REQUEST,
                            Json(ApiResponse {
                                success: false,
                                data: None,
                                error: Some(format!("Failed to parse options: {}", e)),
                            }),
                        )
                    },
                )?);
            }
            _ => continue,
        }
    }

    let file = File::open(temp_file.path()).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Failed to open temp file: {}", e)),
            }),
        )
    })?;

    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!("Failed to memory map file: {}", e)),
                }),
            )
        })?
    };
    let page_range = if let Some(options) = options {
        if let Some(range_str) = options.page_range {
            Some(parse_page_range(&range_str).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ApiResponse {
                        success: false,
                        data: None,
                        error: Some(e.to_string()),
                    }),
                )
            })?)
        } else {
            None
        }
    } else {
        None
    };

    let config = FerrulesParseConfig {
        password: None,
        flatten_pdf: true,
        page_range,
        debug_dir: None,
    };
    let doc = state
        .parser
        .parse_document(&mmap, Uuid::new_v4().to_string(), config, Some(|_| {}))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(e.to_string()),
                }),
            )
        })?;

    let accept_header = headers.get(ACCEPT).and_then(|h| h.to_str().ok());

    match accept_header {
        Some("text/markdown") => {
            let markdown = to_markdown(&doc, &doc.doc_name, None).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ApiResponse {
                        success: false,
                        data: None,
                        error: Some(format!("Failed to convert to markdown: {}", e)),
                    }),
                )
            })?;

            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/markdown")
                .body::<String>(markdown)
                .unwrap())
        }
        _ => {
            // NOTE: Default to JSON
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "application/json")
                .body(
                    serde_json::to_string(&ApiResponse {
                        success: true,
                        data: Some(doc),
                        error: None,
                    })
                    .unwrap(),
                )
                .unwrap())
        }
    }
}

fn parse_page_range(range_str: &str) -> anyhow::Result<std::ops::Range<usize>> {
    if let Some((start, end)) = range_str.split_once('-') {
        let start: usize = start.trim().parse()?;
        let end: usize = end.trim().parse()?;
        if start > 0 && end >= start {
            Ok(std::ops::Range {
                start: start - 1,
                end,
            })
        } else {
            anyhow::bail!("Invalid page range: start must be > 0 and end must be >= start")
        }
    } else {
        // Single page
        let page: usize = range_str.trim().parse()?;
        if page > 0 {
            Ok(std::ops::Range {
                start: page - 1,
                end: page,
            })
        } else {
            anyhow::bail!("Page number must be greater than 0")
        }
    }
}
