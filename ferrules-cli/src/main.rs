use clap::Parser;

mod error_formatter;
use error_formatter::format_error;

use ferrules_core::{
    layout::model::{ORTConfig, OrtExecutionProvider},
    utils::{create_dirs, get_doc_length, save_parsed_document},
    FerrulesParseConfig, FerrulesParser,
};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use memmap2::Mmap;
use std::{
    fmt::Write,
    ops::Range,
    path::{Path, PathBuf},
};
use tokio::fs::File;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Ferrules - High-performance document parsing library",
    long_about = "Ferrules is an opinionated high-performance document parsing library designed to generate LLM-ready documents efficiently. Built with Rust for seamless deployment across various platforms."
)]
struct Args {
    /// Path to the PDF file to be parsed
    file_path: PathBuf,

    // /// Process directory instead of single file
    // #[arg(
    //     long,
    //     default_value_t = false,
    //     help = "Process all PDF files in the specified directory"
    // )]
    // directory: bool,
    #[arg(
        long,
        short('r'),
        help = "Specify pages to parse (e.g., '1-5' or '1' for single page)"
    )]
    page_range: Option<String>,

    /// Specifies the target directory where parsing results will be saved
    ///
    /// If not specified, defaults to the current working directory.
    #[arg(
        short = 'o',
        long,
        env = "FERRULES_OUTPUT_DIR",
        help = "Specify the directory to store parsing result"
    )]
    output_dir: Option<PathBuf>,

    #[arg(long, default_value_t = false, help = "Output the document as html")]
    html: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Output the document in markdown"
    )]
    md: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Specify the directory to store parsing result"
    )]
    save_images: bool,

    /// Use CoreML for layout inference (default: true)
    #[arg(
        long,
        default_value_t = cfg!(target_os = "macos"),
        help = "Enable or disable the use of CoreML for layout inference"
    )]
    coreml: bool,

    #[arg(
        long,
        default_value_t = true,
        help = "Enable or disable Apple Neural Engine acceleration (only applies when CoreML is enabled)"
    )]
    use_ane: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Enable or disable the use of TensorRT for layout inference"
    )]
    trt: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Enable or disable the use of CUDA for layout inference"
    )]
    cuda: bool,

    /// CUDA device ID to use for GPU acceleration (e.g. 0 for first GPU)
    #[arg(
        long,
        help = "CUDA device ID to use (0 for first GPU)",
        default_value_t = 0
    )]
    device_id: i32,

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

    /// Enable debug mode to output additional information
    #[arg(
        long,
        default_value_t = false,
        env = "FERRULES_DEBUG",
        help = "Activate debug mode for detailed processing information"
    )]
    debug: bool,

    /// Directory for debug output files
    #[arg(
        long,
        env = "FERRULES_DEBUG_PATH",
        help = "Specify the directory to store debug output files"
    )]
    debug_dir: Option<PathBuf>,
}

fn parse_page_range(range_str: &str) -> anyhow::Result<Range<usize>> {
    if let Some((start, end)) = range_str.split_once('-') {
        let start: usize = start.trim().parse()?;
        let end: usize = end.trim().parse()?;
        if start > 0 && end >= start {
            Ok(Range {
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
            Ok(Range {
                start: page - 1,
                end: page,
            })
        } else {
            anyhow::bail!("Page number must be greater than 0")
        }
    }
}

fn setup_progress_bar(
    file_path: &Path,
    password: Option<&str>,
    page_range: Option<Range<usize>>,
) -> ProgressBar {
    let length_pages = match get_doc_length(file_path, password, page_range.clone()) {
        Ok(pages) => pages,
        Err(e) => {
            format_error(
                "Document Length Detection Failed",
                "Failed to determine the number of pages in the document.",
                vec![
                    ("File", file_path.display().to_string()),
                    ("Error", e.to_string()),
                    (
                        "Suggestion",
                        "Check if the file exists and is a valid PDF".to_string(),
                    ),
                ],
            );
            std::process::exit(1);
        }
    };
    let pb = ProgressBar::new(length_pages as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {msg}",
        )
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
            write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
        })
        .progress_chars("#>-"),
    );
    pb
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

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args = Args::parse();

    // Check providers
    let providers = parse_ep_args(&args);

    let ort_config = ORTConfig {
        execution_providers: providers,
        intra_threads: args.intra_threads,
        inter_threads: args.inter_threads,
        opt_level: args.graph_opt_level.map(|v| v.try_into().unwrap()),
    };
    // Global tasks
    let parser = FerrulesParser::new(ort_config);

    let page_range = match args.page_range {
        Some(ref page_range_str) => match parse_page_range(page_range_str) {
            Ok(range) => Some(range),
            Err(e) => {
                format_error(
                    "Invalid Page Range",
                    &e.to_string(),
                    vec![
                        ("Input", page_range_str.clone()),
                        (
                            "Format",
                            "Use '1-5' for range or '1' for single page".to_string(),
                        ),
                        ("Note", "Page numbers start from 1".to_string()),
                    ],
                );
                std::process::exit(1);
            }
        },
        None => None,
    };

    let pb = setup_progress_bar(&args.file_path, None, page_range.clone());
    let pbc = pb.clone();

    let doc_name = args
        .file_path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.split('.').next().map(|s| s.to_owned()))
        .unwrap_or(Uuid::new_v4().to_string());

    // Create all dirs
    // TODO: refac this
    let save_figs = args.html | args.save_images;
    let (output_dir_path, debug_path) =
        match create_dirs(args.output_dir.as_ref(), &doc_name, args.debug, save_figs) {
            Ok(paths) => paths,
            Err(e) => {
                format_error(
                    "Directory Creation Failed",
                    "Failed to create output directories.",
                    vec![
                        (
                            "Output Directory",
                            args.output_dir
                                .as_ref()
                                .map_or("current directory".to_string(), |p| {
                                    p.display().to_string()
                                }),
                        ),
                        ("Document Name", doc_name.clone()),
                        ("Error", e.to_string()),
                    ],
                );
                std::process::exit(1);
            }
        };
    // TODO : refac memap
    let file = match File::open(&args.file_path).await {
        Ok(f) => f,
        Err(e) => {
            format_error(
                "File Open Failed",
                "Failed to open the PDF file for processing.",
                vec![
                    ("File", args.file_path.display().to_string()),
                    ("Error", e.to_string()),
                    (
                        "Suggestion",
                        "Check file permissions and ensure the file exists".to_string(),
                    ),
                ],
            );
            std::process::exit(1);
        }
    };
    let mmap = match unsafe { Mmap::map(&file) } {
        Ok(m) => m,
        Err(e) => {
            format_error(
                "Memory Mapping Failed",
                "Failed to memory-map the PDF file.",
                vec![
                    ("File", args.file_path.display().to_string()),
                    ("Error", e.to_string()),
                    ("Suggestion", "Check available system memory".to_string()),
                ],
            );
            std::process::exit(1);
        }
    };

    let config = FerrulesParseConfig {
        password: None,
        flatten_pdf: true,
        page_range,
        debug_dir: debug_path,
    };
    let doc = match parser
        .parse_document(
            &mmap,
            doc_name,
            config,
            Some(move |page_id| {
                pbc.set_message(format!("Page #{}", page_id + 1));
                pbc.inc(1u64);
            }),
        )
        .await
    {
        Ok(result) => result,
        Err(e) => {
            match e {
                ferrules_core::error::FerrulesError::ParseNativeError => {
                    format_error(
                        "Native PDF Parsing Failed",
                        "Failed to parse the PDF file using the native parser.",
                        vec![
                            ("File", args.file_path.display().to_string()),
                            (
                                "Suggestion",
                                "Check if the PDF file is valid and not corrupted".to_string(),
                            ),
                        ],
                    );
                }
                ferrules_core::error::FerrulesError::LayoutParsingError => {
                    format_error(
                        "Layout Detection Failed",
                        "Failed to detect document layout structure.",
                        vec![
                            ("File", args.file_path.display().to_string()),
                            (
                                "Suggestion",
                                "Try using a different execution provider (--cuda, --coreml)"
                                    .to_string(),
                            ),
                        ],
                    );
                }
                ferrules_core::error::FerrulesError::LineMergeError => {
                    format_error(
                        "Line Merging Failed",
                        "Failed to merge text lines during document processing.",
                        vec![
                            ("File", args.file_path.display().to_string()),
                            (
                                "Suggestion",
                                "This might indicate complex text layout in the PDF".to_string(),
                            ),
                        ],
                    );
                }
                ferrules_core::error::FerrulesError::BlockMergeError {
                    block_id,
                    kind,
                    element,
                } => {
                    format_error(
                        "Block Merge Error",
                        "Failed to merge document blocks during processing.",
                        vec![
                            ("Block ID", block_id.to_string()),
                            ("Block Type", kind.to_string()),
                            ("Page Number", element.page_id.to_string()),
                            ("Element", format!("{}-{}", element.id, element.kind)),
                            ("File", args.file_path.display().to_string()),
                        ],
                    );
                }
                ferrules_core::error::FerrulesError::DebugPageError { tmp_dir, page_idx } => {
                    format_error(
                        "Debug Page Processing Failed",
                        "Failed to process page in debug mode.",
                        vec![
                            ("Page", format!("#{}", page_idx + 1)),
                            ("Debug Directory", tmp_dir.display().to_string()),
                            ("File", args.file_path.display().to_string()),
                        ],
                    );
                }
                ferrules_core::error::FerrulesError::ParseTextError { tmp_dir, page_idx } => {
                    format_error(
                        "Text Extraction Failed",
                        "Failed to extract text from document page.",
                        vec![
                            ("Page", format!("#{}", page_idx + 1)),
                            ("Temp Directory", tmp_dir.display().to_string()),
                            ("File", args.file_path.display().to_string()),
                            (
                                "Suggestion",
                                "Try processing a different page range with --page-range"
                                    .to_string(),
                            ),
                        ],
                    );
                }
            }
            std::process::exit(1);
        }
    };

    pb.finish_with_message(format!(
        "Parsed document in {}ms",
        doc.metadata.parsing_duration.as_millis()
    ));
    if let Err(e) = save_parsed_document(
        &doc,
        output_dir_path.clone(),
        args.save_images,
        args.html,
        args.md,
    ) {
        format_error(
            "Document Save Failed",
            "Failed to save the parsed document.",
            vec![
                ("Output Directory", output_dir_path.display().to_string()),
                ("Error", e.to_string()),
                ("Formats", {
                    let mut formats = vec![];
                    if args.html {
                        formats.push("HTML");
                    }
                    if args.md {
                        formats.push("Markdown");
                    }
                    if args.save_images {
                        formats.push("Images");
                    }
                    if formats.is_empty() {
                        formats.push("Default");
                    }
                    formats.join(", ")
                }),
            ],
        );
        std::process::exit(1);
    }
}
