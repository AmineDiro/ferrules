use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;
use clap::Parser;
use ferrules_api::init_tracing;
use ferrules_core::layout::model::{ORTConfig, OrtExecutionProvider};
use ferrules_core::{
    layout::ParseLayoutQueue,
    parse::{document::parse_document, native::ParseNativeQueue},
};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::io::{Seek, Write};
use tempfile::NamedTempFile;
use tokio::{fs::File, net::TcpListener};
use uuid::Uuid;

const MAX_SIZE_LIMIT: usize = 250 * 1024 * 1024;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// OpenTelemetry collector endpoint
    #[arg(long, env = "OTLP_ENDPOINT", default_value = "http://localhost:4317")]
    otlp_endpoint: String,

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
        default_value = "1"
    )]
    inter_threads: usize,

    #[arg(
        long,
        short = 'L',
        help = "Number of parallel layout models",
        default_value = "4"
    )]
    layout_parallel: usize,

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
    layout_queue: ParseLayoutQueue,
    native_queue: ParseNativeQueue,
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
        Some(&args.otlp_endpoint),
        "ferrules-api".into(),
        false,
        use_sentry,
    )
    .expect("can't setup tracing for API");

    let ort_config = ORTConfig {
        execution_providers: providers,
        intra_threads: args.intra_threads,
        inter_threads: args.inter_threads,
        opt_level: args.graph_opt_level.map(|v| v.try_into().unwrap()),
    };

    // Initialize the layout model and queues
    let layout_queue = ParseLayoutQueue::new(ort_config, args.layout_parallel);

    // BG Native worker
    let native_queue = ParseNativeQueue::new();

    let app_state = AppState {
        layout_queue,
        native_queue,
    };

    // Build our application with a route
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/parse", post(parse_document_handler))
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

    let doc = parse_document(
        &mmap,
        Uuid::new_v4().to_string(),
        None,
        true,
        page_range,
        state.layout_queue.clone(),
        state.native_queue.clone(),
        false,
        Some(|_| {}),
    )
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

    Ok(Json(ApiResponse {
        success: true,
        data: Some(doc),
        error: None,
    }))
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
