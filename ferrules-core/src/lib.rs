//! # Ferrules
//!
//! A modern document processing library for extracting structured content from PDF documents.
//!
//! Ferrules provides robust capabilities for:
//!
//! - Document layout analysis using deep learning models
//! - Text extraction with native PDF parsing
//! - Structured content parsing including titles, lists, tables, and images
//! - Export to multiple formats (HTML, Markdown)
//! - Hardware-accelerated processing with configurable execution providers
//!
//! ## Key Features
//!
//! - **Layout Analysis**: Deep learning-based document region detection and classification
//! - **Text Extraction**: Native PDF text extraction with structural preservation
//! - **Structured Output**: Parses content into semantic blocks (titles, paragraphs, lists, etc.)
//! - **Hardware Acceleration**: Supports CPU, GPU (CUDA/TensorRT), and CoreML on macOS
//! - **Output Formats**: Export to HTML and Markdown with optional image extraction
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use ferrules_core::{
//!     layout::model::{ORTConfig, OrtExecutionProvider},
//!     FerrulesParser,
//! };
//!
//! async fn process_document() -> anyhow::Result<()> {
//!     // Configure hardware acceleration
//!     let ort_config = ORTConfig {
//!         execution_providers: vec![OrtExecutionProvider::CPU],
//!         intra_threads: 16,
//!         inter_threads: 4,
//!         opt_level: None,
//!     };
//!
//!     // Initialize parser
//!     let parser = FerrulesParser::new(ort_config);
//!
//!     // Parse document
//!     let doc_bytes = std::fs::read("document.pdf")?;
//!     let parsed_doc = parser.parse_document(
//!         &doc_bytes,
//!         "document".into(),
//!         Default::default(),
//!         None,          // No progress callback
//!     ).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The library consists of several key components:
//!
//! - **Layout Analysis**: Deep learning model for document structure detection
//! - **Text Extraction**: Native PDF parsing with structural preservation
//! - **Content Processing**: Merging and structuring detected elements
//! - **Rendering**: Converting structured content to HTML/Markdown
//!
//! ## Platform Support
//!
//! - Linux: Full support with CPU/GPU acceleration (CUDA, TensorRT)
//! - macOS: Native support with CoreML acceleration
//!
//! ## Performance
//!
//! The library is optimized for efficient document processing:
//!
//! - Configurable execution providers for hardware acceleration
//! - Parallel processing with adjustable thread counts
//! - Tunable optimization levels for inference
//!
//! ## License
//!
//! Licensed under the GPLv3 license.
#![feature(portable_simd)]

pub(crate) mod draw;

pub mod blocks;
pub mod entities;
pub mod error;
pub mod layout;
pub mod ocr;
pub mod render;
pub mod utils;

mod parse;
pub use parse::document::{FerrulesParseConfig, FerrulesParser};
