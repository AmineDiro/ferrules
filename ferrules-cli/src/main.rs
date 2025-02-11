use clap::Parser;

use ferrules_core::{
    layout::{
        model::{ORTConfig, OrtExecutionProvider},
        ParseLayoutQueue,
    },
    parse::{
        document::{get_doc_length, parse_document},
        native::ParseNativeQueue,
    },
    save_parsed_document,
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
        long,
        env = "FERRULES_OUTPUT_DIR",
        help = "Specify the directory to store parsing result"
    )]
    output_dir: Option<PathBuf>,

    #[arg(
        long,
        default_value_t = false,
        help = "Specify the directory to store parsing result"
    )]
    save_images: bool,

    /// Path to the layout model. If not specified, a default model will be used.
    #[arg(
        long,
        env = "FERRULES_LAYOUT_MODEL_PATH",
        help = "Specify the path to the layout model for document parsing"
    )]
    layout_model_path: Option<PathBuf>,

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
        default_value = "2"
    )]
    intra_threads: usize,

    /// Number of threads to use for parallel operation execution
    #[arg(
        long,
        help = "Number of threads to use for executing operations in parallel",
        default_value = "1"
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
    let length_pages = get_doc_length(file_path, password, page_range.clone()).unwrap();
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
    let layout_queue = ParseLayoutQueue::new(ort_config, 1);
    let native_queue = ParseNativeQueue::new();

    let page_range = args
        .page_range
        .map(|page_range_str| parse_page_range(&page_range_str).unwrap());

    let pb = setup_progress_bar(&args.file_path, None, page_range.clone());
    let pbc = pb.clone();

    let doc_name = args
        .file_path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.split('.').next().map(|s| s.to_owned()))
        .unwrap_or(Uuid::new_v4().to_string());

    // TODO : refac memap
    let file = File::open(&args.file_path).await.unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    let doc = parse_document(
        &mmap,
        doc_name,
        None,
        true,
        page_range,
        layout_queue,
        native_queue,
        args.debug,
        Some(move |page_id| {
            pbc.set_message(format!("Page #{}", page_id + 1));
            pbc.inc(1u64);
        }),
    )
    .await
    .unwrap();

    pb.finish_with_message(format!(
        "Parsed document in {}ms",
        doc.metadata.parsing_duration.as_millis()
    ));
    save_parsed_document(&doc, args.output_dir.as_ref(), args.save_images).unwrap();
}
