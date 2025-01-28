use anyhow::Context;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use itertools::izip;
use memmap2::Mmap;
use pdfium_render::prelude::{PdfPage, PdfPageTextChar, PdfRenderConfig, Pdfium};
use rayon::prelude::*;
use std::{fmt::Write, fs::File, ops::Range, path::Path, time::Instant};
use uuid::Uuid;

use crate::{
    blocks::{Block, BlockType, ImageBlock, List, TextBlock, Title},
    chunk_docs_range,
    entities::{
        BBox, CharSpan, Document, Element, ElementType, Line, Page, PageID, StructuredPage,
    },
    layout::{
        draw::{draw_blocks, draw_layout_bboxes, draw_ocr_bboxes, draw_text_lines},
        model::{LayoutBBox, ORTLayoutParser},
    },
    ocr::parse_image_ocr,
    sanitize_doc_name,
};

/// This constant defines the minimum ratio between the area of text lines identified
/// by the pdfium2 and the area of text regions detected through layout analysis.
/// If this ratio falls below the threshold of 0.5 (or 50%), it indicates that the page
/// may not have enough __native__ lines, and therefore should
/// be considered for OCR to ensure accurate text extraction.
const MIN_LAYOUT_COVERAGE_THRESHOLD: f32 = 0.5;

/// This constant defines the minimum required intersection ratio between the bounding box of an
/// OCR-detected text line and a text block detected through layout analysis.
/// This approach ensures that only text lines significantly overlapping with a layout block are
/// paired, thus improving the accuracy of OCR-text and layout alignment.
const MIN_INTERSECTION_LAYOUT: f32 = 0.5;

/// Weights used for calculating distances between bounding boxes in layout analysis
/// X_WEIGHT is weighted higher (5.0) to prioritize horizontal alignment
/// Y_WEIGHT is weighted lower (1.0) to be more lenient with vertical spacing
const LAYOUT_DISTANCE_X_WEIGHT: f32 = 5.0;
const LAYOUT_DISTANCE_Y_WEIGHT: f32 = 1.0;

/// Maximum allowable distance between a text line and a layout block for assignment.
/// If the weighted distance (using X_WEIGHT and Y_WEIGHT) between a text line and
/// the nearest layout block exceeds this threshold, the line will not be assigned to any block.
/// This helps prevent incorrect assignments of text lines that are too far from layout blocks.
const MAXIMUM_ASSIGNMENT_DISTANCE: f32 = 20.0;

fn parse_text_spans<'a>(
    chars: impl Iterator<Item = PdfPageTextChar<'a>>,
    page_bbox: &BBox,
) -> Vec<CharSpan> {
    let mut spans: Vec<CharSpan> = Vec::new();

    for char in chars {
        if spans.is_empty() {
            let span = CharSpan::new_from_char(&char, page_bbox);
            spans.push(span);
        } else {
            let span = spans.last_mut().unwrap();
            match span.append(&char, page_bbox) {
                Some(_) => {}
                None => {
                    let span = CharSpan::new_from_char(&char, page_bbox);
                    spans.push(span);
                }
            };
        }
    }

    spans
}

fn parse_text_lines(spans: Vec<CharSpan>) -> Vec<Line> {
    let mut lines = Vec::new();
    for span in spans {
        if lines.is_empty() {
            let line = Line::new_from_span(span);
            lines.push(line);
        } else {
            let line = lines.last_mut().unwrap();
            if let Err(span) = line.append(span) {
                let line = Line::new_from_span(span);
                lines.push(line)
            }
        }
    }

    lines
}

pub fn parse_pages(
    pdf_pages: &mut [(PageID, PdfPage)],
    layout_model: &ORTLayoutParser,
    tmp_dir: &Path,
    flatten_pdf: bool,
    debug: bool,
    pb: &ProgressBar,
) -> anyhow::Result<Vec<StructuredPage>> {
    // TODO: deal with document embedded forms?
    for (_, page) in pdf_pages.iter_mut() {
        if flatten_pdf {
            page.flatten()?;
        }
    }
    let rescale_factors: Vec<f32> = pdf_pages
        .iter()
        .map(|(_, page)| {
            let scale_w = ORTLayoutParser::REQUIRED_WIDTH as f32 / page.width().value;
            let scale_h = ORTLayoutParser::REQUIRED_HEIGHT as f32 / page.height().value;
            f32::min(scale_h, scale_w)
        })
        .collect();

    let page_images: Result<Vec<_>, _> = pdf_pages
        .iter()
        .zip(rescale_factors.iter())
        .map(|((_, page), rescale_factor)| {
            page.render_with_config(
                &PdfRenderConfig::default().scale_page_by_factor(*rescale_factor),
            )
            .map(|bitmap| bitmap.as_image())
        })
        .collect();
    let page_images = page_images?;

    let downscale_factors = rescale_factors
        .iter()
        .map(|f| 1f32 / *f)
        .collect::<Vec<f32>>();

    let pages_layout = layout_model.parse_layout_batch(&page_images, &downscale_factors)?;

    let mut structured_pages = Vec::with_capacity(pdf_pages.len());

    for ((page_idx, page), page_layout, page_image, downscale_factor) in izip![
        pdf_pages.iter(),
        &pages_layout,
        page_images,
        &downscale_factors
    ] {
        let page_bbox = BBox {
            x0: 0f32,
            y0: 0f32,
            x1: page.width().value,
            y1: page.height().value,
        };

        // let page_rotation = page.rotation().unwrap_or(PdfPageRenderRotation::None);
        let text_spans = parse_text_spans(page.text()?.chars().iter(), &page_bbox);
        let text_lines = parse_text_lines(text_spans);

        let text_layout_box: Vec<&LayoutBBox> =
            page_layout.iter().filter(|b| b.is_text_block()).collect();
        let need_ocr = page_needs_ocr(&text_layout_box, &text_lines);

        let ocr_result = if need_ocr {
            parse_image_ocr(&page_image, *downscale_factor).ok()
        } else {
            None
        };

        let text_lines = if need_ocr && ocr_result.is_some() {
            let lines = ocr_result
                .as_ref()
                .unwrap()
                .iter()
                .map(|ocr_line| ocr_line.to_line())
                .collect::<Vec<_>>();
            lines
        } else {
            text_lines
        };

        // Merging elements with layout
        let mut elements = merge_lines_layout(page_layout, &text_lines, *page_idx)?;
        let merged_layout_blocks_ids = elements
            .iter()
            .map(|e| e.layout_block_id)
            .collect::<Vec<_>>();
        let unmerged_layout_boxes: Vec<&LayoutBBox> = page_layout
            .iter()
            .filter(|&b| !merged_layout_blocks_ids.contains(&b.id))
            .collect();

        merge_remaining(&mut elements, &unmerged_layout_boxes, *page_idx);

        let page_image = page
            .render_with_config(&PdfRenderConfig::default().scale_page_by_factor(1f32))
            .map(|bitmap| bitmap.as_image())?;
        if debug {
            // TODO: add feature compile debug
            let output_file = tmp_dir.join(format!("page_{}.png", page_idx));
            let final_output_file = tmp_dir.join(format!("page_blocks_{}.png", page_idx));
            let out_img = draw_text_lines(&text_lines, &page_image)?;
            let out_img = draw_layout_bboxes(page_layout, &out_img.into())?;
            // Draw the final prediction -
            let blocks = merge_elements_into_blocks(elements.clone())?;

            let final_img = draw_blocks(&blocks, &page_image)?;

            if let Some(ocr_result) = ocr_result {
                let out_img = draw_ocr_bboxes(&ocr_result, &out_img.into())?;
                out_img.save(output_file)?;
            } else {
                out_img.save(output_file)?;
            }

            final_img.save(final_output_file)?;
        };

        let structured_page = StructuredPage {
            id: *page_idx,
            width: page_bbox.width(),
            height: page_bbox.height(),
            image: page_image,
            elements,
            need_ocr,
        };

        structured_pages.push(structured_page);

        pb.set_message(format!("Page #{}", *page_idx + 1));
        pb.inc(1u64);
    }

    Ok(structured_pages)
}

fn get_doc_bytes<P: AsRef<Path>>(path: P) -> anyhow::Result<Mmap> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}

fn get_page_len(doc_bytes: &[u8], password: Option<&str>) -> usize {
    let pdfium = Pdfium::new(Pdfium::bind_to_statically_linked_library().unwrap());
    let document = pdfium
        .load_pdf_from_byte_slice(doc_bytes, password)
        .expect("can't open doc with pdfium");
    document.pages().len() as usize
}

#[allow(clippy::too_many_arguments)]
fn parse_document_range(
    doc_bytes: &[u8],
    page_range: Range<usize>,
    layout_model: &ORTLayoutParser,
    password: Option<&str>,
    flatten_pdf: bool,
    tmp_dir: &Path,
    debug: bool,
    // TODO: This should be a callback function
    pb: ProgressBar,
) -> anyhow::Result<Vec<StructuredPage>> {
    let pdfium = Pdfium::new(Pdfium::bind_to_statically_linked_library()?);
    let mut document = pdfium.load_pdf_from_byte_slice(doc_bytes, password)?;
    let mut pages: Vec<_> = document.pages_mut().iter().enumerate().collect();
    let mut pages = pages.drain(page_range).collect::<Vec<_>>();
    parse_pages(&mut pages, layout_model, tmp_dir, flatten_pdf, debug, &pb)
}

pub fn parse_document<P: AsRef<Path>>(
    path: P,
    layout_model: &ORTLayoutParser,
    password: Option<&str>,
    flatten_pdf: bool,
    page_range: Option<Range<usize>>,
    debug: bool,
) -> anyhow::Result<Document<P>> {
    let start_time = Instant::now();
    let doc_name = path
        .as_ref()
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.split('.').next().map(|s| s.to_owned()))
        .unwrap_or(Uuid::new_v4().to_string());

    let tmp_dir = std::env::temp_dir().join(format!("ferrules-{}", sanitize_doc_name(&doc_name)));
    if debug {
        std::fs::create_dir_all(&tmp_dir)?;
    }

    let n_workers = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(4usize);

    let doc_bytes = get_doc_bytes(&path)?;

    // Divide doc into chunks
    let n_pages = get_page_len(&doc_bytes, password);
    let chunks = chunk_docs_range(n_pages, n_workers, page_range);

    let pb = ProgressBar::new(chunks.iter().map(|c| c.len()).sum::<usize>() as u64);
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

    //  Process doc chunks in threadpool
    let parsed_pages: anyhow::Result<Vec<_>> = chunks
        .into_par_iter()
        .map(|page_range| {
            parse_document_range(
                &doc_bytes,
                page_range,
                layout_model,
                password,
                flatten_pdf,
                &tmp_dir,
                debug,
                pb.clone(),
            )
        })
        .collect();
    let parsed_pages: Vec<_> = parsed_pages
        .context("Error parsing pages")?
        .into_iter()
        .flatten()
        .collect();

    // TODO: clone might be huge here
    let all_elements = parsed_pages
        .iter()
        .flat_map(|p| p.elements.clone())
        .collect::<Vec<_>>();

    let pages = parsed_pages
        .into_iter()
        .map(|sp| Page {
            id: sp.id,
            width: sp.width,
            height: sp.height,
            need_ocr: sp.need_ocr,
            image: sp.image,
        })
        .collect();

    let blocks = merge_elements_into_blocks(all_elements)?;

    let duration = Instant::now().duration_since(start_time).as_millis();
    pb.finish_with_message(format!("Parsed document in {}ms", duration));

    Ok(Document {
        path,
        doc_name,
        pages,
        blocks,
        debug_path: if debug { Some(tmp_dir) } else { None },
    })
}

fn page_needs_ocr(text_boxes: &[&LayoutBBox], text_lines: &[Line]) -> bool {
    let line_area = text_lines.iter().map(|l| l.bbox.area()).sum::<f32>();
    let text_layoutbbox_area = text_boxes.iter().map(|l| l.bbox.area()).sum::<f32>();

    if text_layoutbbox_area > 0f32 {
        line_area / text_layoutbbox_area < MIN_LAYOUT_COVERAGE_THRESHOLD
    } else {
        true
    }
}

fn merge_or_create_elements(
    elements: &mut Vec<Element>,
    line: &Line,
    line_layout_block: &LayoutBBox,
    page_id: PageID,
) {
    if elements.is_empty() {
        let mut el = Element::from_layout_block(0, line_layout_block, page_id);
        el.push_line(line);
        elements.push(el);
    }

    // let last_el = elements.last_mut().unwrap();
    let matched_element = elements
        .iter_mut()
        .find(|e| e.layout_block_id == line_layout_block.id);

    match matched_element {
        Some(el) => {
            el.push_line(line);
        }
        None => {
            let mut element =
                Element::from_layout_block(elements.len() + 1, line_layout_block, page_id);
            element.push_line(line);
            elements.push(element);
        }
    }
}
/// Merges lines into blocks based on their layout, maintaining the order of lines.
///
/// This function takes a list of text boxes representing layout bounding boxes that contain text,
/// and a list of lines (which could be obtained from OCR or  PDF library pdfium2,
/// and merges these lines into blocks. The merging is done based on the intersection
/// of each line with the layout bounding boxes. The function prioritizes maintaining
/// the order of the lines, rather than the layout blocks.
fn merge_lines_layout(
    layout_boxes: &[LayoutBBox],
    lines: &[Line],
    page_id: usize,
) -> anyhow::Result<Vec<Element>> {
    let line_block_iterator = lines.iter().map(|line| {
        // TODO: the max here is sometimes very far away from the line.
        // ex: megatrends.pdf, header is categorized as text-block but the intersection  happens
        //
        // Get max intersection block for the line
        let max_intersection_bbox = layout_boxes.iter().max_by(|a, b| {
            let a_intersection = a.bbox.intersection(&line.bbox);
            let b_intersection = b.bbox.intersection(&line.bbox);

            a_intersection.partial_cmp(&b_intersection).unwrap()
        });
        // Get min distance block for the line
        let min_distance_block = layout_boxes.iter().min_by(|a, b| {
            let a_intersection = a.bbox.distance(
                &line.bbox,
                LAYOUT_DISTANCE_X_WEIGHT,
                LAYOUT_DISTANCE_Y_WEIGHT,
            );
            let b_intersection = b.bbox.distance(
                &line.bbox,
                LAYOUT_DISTANCE_X_WEIGHT,
                LAYOUT_DISTANCE_Y_WEIGHT,
            );
            a_intersection.partial_cmp(&b_intersection).unwrap()
        });
        let max_intersection_bbox = max_intersection_bbox.and_then(|b| {
            if line.bbox.intersection(&b.bbox) / line.bbox.area() > MIN_INTERSECTION_LAYOUT {
                Some(b)
            } else {
                None
            }
        });
        // Compare based on distance
        let matched_block = if max_intersection_bbox.is_none() {
            min_distance_block.and_then(|b| {
                if b.bbox.distance(
                    &line.bbox,
                    LAYOUT_DISTANCE_X_WEIGHT,
                    LAYOUT_DISTANCE_Y_WEIGHT,
                ) < MAXIMUM_ASSIGNMENT_DISTANCE
                {
                    Some(b)
                } else {
                    None
                }
            })
        } else {
            max_intersection_bbox
        };
        (line, matched_block)
    });

    let mut headers = Vec::new();
    let mut elements = Vec::new();
    let mut footers = Vec::new();
    for (line, layout_block) in line_block_iterator {
        match &layout_block.as_ref() {
            Some(&line_layout_block) => match line_layout_block.label {
                "Page-header" => {
                    merge_or_create_elements(&mut headers, line, line_layout_block, page_id);
                }
                "Page-footer" => {
                    merge_or_create_elements(&mut footers, line, line_layout_block, page_id);
                }
                _ => {
                    merge_or_create_elements(&mut elements, line, line_layout_block, page_id);
                }
            },
            // Line is detected but isn't assignable to some layout element
            None => {
                // TODO:
                // Check distance between line and the last element

                // let el = Element {
                //     id: 0,
                //     layout_block_id: -1,
                //     text_block: ElementText {
                //         text: line.text.to_owned(),
                //     },
                //     kind: ElementType::Text,
                //     page_id,
                //     bbox: line.bbox.clone(),
                // };
                // elements.push(el);
            }
        }
    }
    elements.append(&mut footers);
    headers.append(&mut elements);
    Ok(headers)
}

fn merge_remaining(elements: &mut Vec<Element>, remaining: &[&LayoutBBox], page_id: PageID) {
    for layout_box in remaining {
        let closest_block = elements
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let a_intersection = a.bbox.distance(
                    &layout_box.bbox,
                    LAYOUT_DISTANCE_X_WEIGHT,
                    LAYOUT_DISTANCE_Y_WEIGHT,
                );
                let b_intersection = b.bbox.distance(
                    &layout_box.bbox,
                    LAYOUT_DISTANCE_X_WEIGHT,
                    LAYOUT_DISTANCE_Y_WEIGHT,
                );
                a_intersection.partial_cmp(&b_intersection).unwrap()
            })
            .map(|(index, _)| index)
            .unwrap_or(elements.len());

        elements.insert(
            closest_block,
            Element::from_layout_block(elements.len(), layout_box, page_id),
        );
    }
}

fn merge_elements_into_blocks(elements: Vec<Element>) -> anyhow::Result<Vec<Block>> {
    let mut element_it = elements.into_iter().peekable();

    let mut blocks = Vec::new();
    let mut block_id = 0;
    while let Some(mut curr_el) = element_it.next() {
        match &mut curr_el.kind {
            crate::entities::ElementType::Text => {
                let text_block = Block {
                    id: block_id,
                    kind: crate::blocks::BlockType::TextBlock(TextBlock {
                        text: curr_el.text_block.text,
                    }),
                    pages_id: vec![curr_el.page_id],
                    bbox: curr_el.bbox,
                };
                // TODO : Change this to use some minimum gap
                // Check to see if we have another text block that is close
                // while let Some(next_el) = element_it.peek() {
                //     if matches!(next_el.kind, crate::entities::ElementType::Text(_))
                //         && (curr_el.bbox.distance(&next_el.bbox, 1.0, 1.0)
                //             < MAXIMUM_ASSIGNMENT_DISTANCE)
                //     {
                //         text_block.merge(next_el)?;
                //         element_it.next();
                //     } else {
                //         break;
                //     }
                // }
                block_id += 1;
                blocks.push(text_block);
            }
            crate::entities::ElementType::ListItem => {
                let mut list_block = Block {
                    id: block_id,
                    kind: BlockType::ListBlock(List {
                        items: vec![curr_el.text_block.text],
                    }),
                    pages_id: vec![curr_el.page_id],
                    bbox: curr_el.bbox,
                };

                while let Some(next_el) = element_it.peek() {
                    // TODO: add constraint on gap between bounding boxes on all dimensions (l,r,b,t)
                    if matches!(next_el.kind, crate::entities::ElementType::ListItem) {
                        let next_el = element_it.next().unwrap();
                        list_block.merge(next_el)?;
                    } else {
                        break;
                    }
                }
                block_id += 1;
                blocks.push(list_block);
            }
            ElementType::FootNote | ElementType::Caption => {
                // We find the closest image and create and image block
                loop {
                    match element_it.peek() {
                        None => {
                            // last element -> transform to txt block and break
                            let text_block = Block {
                                id: block_id,
                                kind: crate::blocks::BlockType::TextBlock(TextBlock {
                                    text: curr_el.text_block.text,
                                }),
                                pages_id: vec![curr_el.page_id],
                                bbox: curr_el.bbox,
                            };
                            element_it.next();
                            block_id += 1;
                            blocks.push(text_block);
                            break;
                        }
                        Some(next_el) => {
                            match &next_el.kind {
                                crate::entities::ElementType::FootNote
                                | crate::entities::ElementType::Caption => {
                                    // Merge this with a the caption
                                    curr_el.text_block.append_line(&next_el.text_block.text);
                                    element_it.next();
                                }
                                crate::entities::ElementType::Image => {
                                    curr_el.bbox.merge(&next_el.bbox);
                                    let img_block = Block {
                                        id: block_id,
                                        kind: BlockType::Image(ImageBlock {
                                            caption: Some(curr_el.text_block.text),
                                        }),
                                        pages_id: vec![next_el.page_id],
                                        bbox: curr_el.bbox,
                                    };
                                    block_id += 1;
                                    blocks.push(img_block);
                                    element_it.next();
                                    break;
                                }
                                _ => {
                                    // This caption isn't associated with Image/Table, transform to textblock
                                    let text_block = Block {
                                        id: block_id,
                                        kind: crate::blocks::BlockType::TextBlock(TextBlock {
                                            text: curr_el.text_block.text,
                                        }),
                                        pages_id: vec![curr_el.page_id],
                                        bbox: curr_el.bbox,
                                    };
                                    block_id += 1;
                                    blocks.push(text_block);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            crate::entities::ElementType::Image => {
                match element_it.peek() {
                    None => {
                        // last element -> transform to txt block and break
                        let block = Block {
                            id: block_id,
                            kind: crate::blocks::BlockType::Image(ImageBlock { caption: None }),
                            pages_id: vec![curr_el.page_id],
                            bbox: curr_el.bbox,
                        };
                        element_it.next();
                        block_id += 1;
                        blocks.push(block);
                    }
                    Some(next_el) => {
                        match &next_el.kind {
                            crate::entities::ElementType::FootNote
                            | crate::entities::ElementType::Caption => {
                                // TODO: check if there is a case where there is multiple caption associated with the same image
                                let next_el = element_it.next().unwrap();
                                curr_el.bbox.merge(&next_el.bbox);
                                let block = Block {
                                    id: block_id,
                                    kind: crate::blocks::BlockType::Image(ImageBlock {
                                        caption: Some(next_el.text_block.text),
                                    }),
                                    pages_id: vec![curr_el.page_id],
                                    bbox: curr_el.bbox,
                                };
                                block_id += 1;
                                blocks.push(block);
                            }
                            _ => {
                                let block = Block {
                                    id: block_id,
                                    kind: crate::blocks::BlockType::Image(ImageBlock {
                                        caption: None,
                                    }),
                                    pages_id: vec![curr_el.page_id],
                                    bbox: curr_el.bbox,
                                };
                                block_id += 1;
                                blocks.push(block);
                            }
                        }
                    }
                }
            }
            ElementType::Header => {
                let mut header_block = Block {
                    id: block_id,
                    kind: BlockType::Header(TextBlock {
                        text: curr_el.text_block.text,
                    }),
                    pages_id: vec![curr_el.page_id],
                    bbox: curr_el.bbox,
                };

                while let Some(next_el) = element_it.peek() {
                    if matches!(next_el.kind, crate::entities::ElementType::Header) {
                        let next_el = element_it.next().unwrap();
                        header_block.merge(next_el)?;
                    } else {
                        break;
                    }
                }
                block_id += 1;
                blocks.push(header_block);
            }
            ElementType::Footer => {
                let mut footer_block = Block {
                    id: block_id,
                    kind: BlockType::Footer(TextBlock {
                        text: curr_el.text_block.text,
                    }),
                    pages_id: vec![curr_el.page_id],
                    bbox: curr_el.bbox,
                };

                while let Some(next_el) = element_it.peek() {
                    if matches!(next_el.kind, ElementType::Footer) {
                        let next_el = element_it.next().unwrap();
                        footer_block.merge(next_el)?;
                    } else {
                        break;
                    }
                }
                block_id += 1;
                blocks.push(footer_block);
            }
            ElementType::Title | ElementType::Subtitle => {
                // TODO:
                // Handle title level via text font size (using kmeans)
                let title = Block {
                    id: block_id,
                    kind: BlockType::Title(Title {
                        level: 0,
                        text: curr_el.text_block.text,
                    }),
                    pages_id: vec![curr_el.page_id],
                    bbox: curr_el.bbox,
                };
                block_id += 1;
                blocks.push(title);
            }
            _ => {
                continue;
            }
        }
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {

    use crate::entities::ElementText;

    use super::*;
    use {BBox, Element, ElementType};

    fn create_text_element(id: usize, page_id: usize, text: &str, bbox: BBox) -> Element {
        Element {
            id,
            layout_block_id: 0,
            kind: ElementType::Text,
            text_block: ElementText {
                text: text.to_owned(),
            },
            page_id,
            bbox,
        }
    }

    fn create_list_element(id: usize, page_id: usize, text: &str, bbox: BBox) -> Element {
        Element {
            id,
            layout_block_id: 0,
            kind: ElementType::ListItem,
            text_block: ElementText {
                text: text.to_string(),
            },
            page_id,
            bbox,
        }
    }

    fn create_caption_element(id: usize, page_id: usize, text: &str, bbox: BBox) -> Element {
        Element {
            id,
            layout_block_id: 0,
            kind: ElementType::Caption,
            text_block: ElementText {
                text: text.to_string(),
            },
            page_id,
            bbox,
        }
    }

    fn create_footnote_element(id: usize, page_id: usize, text: &str, bbox: BBox) -> Element {
        Element {
            id,
            layout_block_id: 0,
            kind: ElementType::FootNote,
            text_block: ElementText {
                text: text.to_string(),
            },
            page_id,
            bbox,
        }
    }
    fn create_image_element(id: usize, page_id: usize, bbox: BBox) -> Element {
        Element {
            id,
            layout_block_id: 0,
            kind: ElementType::Image,
            text_block: ElementText::default(),
            page_id,
            bbox,
        }
    }

    #[test]
    fn test_merge_adjacent_text_blocks() -> anyhow::Result<()> {
        let bbox1 = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let bbox2 = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_text_element(0, 1, "First paragraph", bbox1),
            create_text_element(1, 1, "Second paragraph", bbox2),
        ];

        let blocks = merge_elements_into_blocks(elements)?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::TextBlock(text) = &blocks[0].kind {
            assert!(text.text.contains("First paragraph"));
            assert!(text.text.contains("Second paragraph"));
        } else {
            panic!("Expected TextBlock");
        }
        Ok(())
    }

    #[test]
    fn test_merge_list_items() -> anyhow::Result<()> {
        let bbox1 = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let bbox2 = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_list_element(0, 1, "First item", bbox1),
            create_list_element(1, 1, "Second item", bbox2.clone()),
            create_text_element(2, 1, "Random text", bbox2),
        ];

        let blocks = merge_elements_into_blocks(elements)?;

        assert_eq!(blocks.len(), 2);
        if let BlockType::ListBlock(list) = &blocks[0].kind {
            assert_eq!(list.items.len(), 2);
            assert_eq!(list.items[0], "First item");
            assert_eq!(list.items[1], "Second item");
        } else {
            panic!("Expected ListItem");
        }
        Ok(())
    }

    #[test]
    fn test_merge_caption_with_image() -> anyhow::Result<()> {
        let caption_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let image_bbox = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_caption_element(0, 1, "Image caption", caption_bbox),
            create_image_element(1, 1, image_bbox),
        ];

        let blocks = merge_elements_into_blocks(elements)?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::Image(image) = &blocks[0].kind {
            assert_eq!(image.caption, Some("Image caption".to_string()));
        } else {
            panic!("Expected Image");
        }
        Ok(())
    }

    #[test]
    fn test_merge_orphan_caption_becomes_text() -> anyhow::Result<()> {
        let caption_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };

        let elements = vec![create_caption_element(0, 1, "Orphan caption", caption_bbox)];

        let blocks = merge_elements_into_blocks(elements)?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::TextBlock(text) = &blocks[0].kind {
            assert_eq!(text.text, "Orphan caption");
        } else {
            panic!("Expected TextBlock");
        }
        Ok(())
    }

    #[test]
    fn test_merge_distant_text_blocks_not_merged() -> anyhow::Result<()> {
        let bbox1 = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let bbox2 = BBox {
            x0: 0.0,
            y0: 20.0, // Far away
            x1: 2.0,
            y1: 22.0,
        };

        let elements = vec![
            create_text_element(0, 1, "First paragraph", bbox1),
            create_text_element(1, 1, "Distant paragraph", bbox2),
        ];

        let blocks = merge_elements_into_blocks(elements)?;

        assert_eq!(blocks.len(), 2);
        Ok(())
    }

    #[test]
    fn test_merge_image_last_element() -> anyhow::Result<()> {
        let image_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };

        let elements = vec![create_image_element(0, 1, image_bbox)];

        let blocks = merge_elements_into_blocks(elements)?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::Image(image) = &blocks[0].kind {
            assert_eq!(image.caption, None);
        } else {
            panic!("Expected Image block");
        }
        Ok(())
    }

    #[test]
    fn test_merge_image_with_following_caption() -> anyhow::Result<()> {
        let image_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let caption_bbox = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_image_element(0, 1, image_bbox),
            create_caption_element(1, 1, "Image Description", caption_bbox),
        ];

        let blocks = merge_elements_into_blocks(elements)?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::Image(image) = &blocks[0].kind {
            assert_eq!(image.caption, Some("Image Description".to_string()));
        } else {
            panic!("Expected Image block with caption");
        }
        Ok(())
    }

    #[test]
    fn test_merge_image_with_following_non_caption() -> anyhow::Result<()> {
        let image_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let text_bbox = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_image_element(0, 1, image_bbox),
            create_text_element(1, 1, "Regular text", text_bbox),
        ];

        let blocks = merge_elements_into_blocks(elements)?;

        assert_eq!(blocks.len(), 2);
        if let BlockType::Image(image) = &blocks[0].kind {
            assert_eq!(image.caption, None);
        } else {
            panic!("Expected Image block without caption");
        }

        if let BlockType::TextBlock(text) = &blocks[1].kind {
            assert_eq!(text.text, "Regular text");
        } else {
            panic!("Expected Text block");
        }
        Ok(())
    }

    #[test]
    fn test_merge_image_with_footnote() -> anyhow::Result<()> {
        let image_bbox = BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 2.0,
            y1: 2.0,
        };
        let footnote_bbox = BBox {
            x0: 0.0,
            y0: 2.1,
            x1: 2.0,
            y1: 4.1,
        };

        let elements = vec![
            create_image_element(0, 1, image_bbox),
            create_footnote_element(1, 1, "Image Footnote", footnote_bbox),
        ];

        let blocks = merge_elements_into_blocks(elements)?;

        assert_eq!(blocks.len(), 1);
        if let BlockType::Image(image) = &blocks[0].kind {
            assert_eq!(image.caption, Some("Image Footnote".to_string()));
        } else {
            panic!("Expected Image block with footnote as caption");
        }
        Ok(())
    }
}
