# Ferrules

<div align="center">
    <img src="./ferrules-logo.png" alt="Ferrules-logo" width="31%"  style="border-radius: 50%; padding-bottom: 20px"/>
</div>

Ferrules is an **opinionated high-performance document parsing library** designed to generate LLM-ready documents efficiently.
Unlike alternatives such as `unstructured` which are slow and Python-based, `ferrules` is written in Rust and aims to provide a seamless experience with robust deployment across various platforms.

| **NOTE** A ferrule is a corruption of Latin viriola on a pencil known as a Shoe, is any of a number of types of objects, generally used for fastening, joining, sealing, or reinforcement.

## Features

- **📄 PDF Parsing and Layout Extraction:**

  - Utilizes `pdfium2` to parse documents.
  - Supports OCR using Apple's Vision on macOS (using `objc2` Rust bindings and [`VNRecognizeTextRequest`](https://developer.apple.com/documentation/vision/vnrecognizetextrequest) functionality).
  - Extracts and analyzes **page layouts** with advanced preprocessing and postprocessing techniques.
  - Accelerate model inference on Apple Neural Engine (ANE)/GPU (using [`ort`](https://ort.pyke.io/) library).
  - Merges layout with PDF text lines for comprehensive document understanding.

- **🔄 Document Transformation:**

  - Groups captions, footers, and other elements intelligently.
  - Structures lists and merges blocks into cohesive sections.
  - Detects headings and titles using machine learning for logical document structuring.

- **🖨️ Rendering:**

  - Provides HTML, Markdown, and JSON rendering options for versatile use cases.

- **⚙️ Advanced Functionalities:**

  - Offers configurable inference parameters for optimized processing.
  - Batch inference on document pages. (COMING SOON)

- **🛠️ API and CLI:**

  - Provides both a CLI and API interface
  - Supports tracing

## Installation

// tocome

## Roadmap

- [x] Build pdfium statically for Macos

- [x] Parse document using pdfium

  - [x] Parse char
  - [x] Merge chars into CharSpans
  - [x] Merge spans into Lines

- [ ] Layout:

  - [x] Find Layout Model and run with ORT
  - [x] Accelerate Model on ANE/GPU
  - [x] Extract Page Layout
    - [x] Preprocess pdfium image
    - [x] Postprocess tensor -> nms
    - [x] Verify labels
  - [x] Determine pages needing OCR (coverage lines/blocks)
  - [x] OCR -> Use Apple vision on macOS target_os
  - [x] Merge Layout with pdfium lines
    - [x] Rescale / or / downscale line bbox/ layout bbox
    - [x] Merge intersection lines (from pdfium and OCR) with max bbox into blocks
    - [ ] Add lines to bbox based on distance
    - [ ] Add remaining layout blocks to blocks based on position

- [ ] Document IR merge:

  - [ ] Group caption/footer blocks with image blocks/tables using minimum gap
  - [ ] Group listItems into list : Find first and merge subsequent items
  - [ ] Merge Blocks into sections
  - [ ] Process SubHeader/Titles using kmeans on line heigths to get the title_level
  - [ ] Run processors (Text, List, PageHeader )
  - [ ] Get PDF Bookmarks (TOC) and reconcile detected titles with TOC

- [ ] Render Document

  - [ ] HTML renderer
  - [ ] Markdown renderer
  - [ ] JSON renderer
    - [ ] Crop images and save in directory if `--save_image` flag

- [x] Create CLI ferrules
- [ ] Add tracing
- [ ] `eyre` | `thiserror` for custom errosk
- [ ] Configurable inference params: ORTProviders/ batch_size, confidence_score, NMS ..

- [ ] OCR: Find good recognition model (onnxtr ??)

- [ ] Batch inference on pages (TODO -> )

  - [x] Export onnx with dynamic batch_size
  - [ ] Run layout on &[DynamicImage]
  - Explored this onnxruntime on coreml isn't faster for some weird reason
    - [ ] check on nvidia-gpu if batching is better

- [ ] API

  - [ ] Unify Config for env/CLI/API
  - [ ] Dynamic batching of document(pages) to process

- [ ] Open document mmap and share range of page between threads
- [ ] Build pdfium statically for Linux
- [ ] Determine page orientation + deskew
- [ ] Optimize layout model for ANE
- [ ] ORT inference in fp16/mixed precision
- [ ] Move to other yolo versions: yolov11s seems better with less params [yolo-doclaynet](https://github.com/ppaanngggg/yolo-doclaynet)

## Resources:

- Apple vision text detection:

  - https://github.com/straussmaximilian/ocrmac/blob/main/ocrmac/ocrmac.py
  - https://docs.rs/objc2-vision/latest/objc2_vision/index.html
  - https://developer.apple.com/documentation/vision/recognizing-text-in-images

- `ort` : https://ort.pyke.io/

## Credits

- This project uses models from the [yolo-doclaynet repository](https://github.com/ppaanngggg/yolo-doclaynet). We are grateful to the contributors of that project.
