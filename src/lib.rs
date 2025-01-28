use anyhow::Context;
use colored::*;
use std::{
    fs::File,
    io::{BufWriter, Write},
    ops::Range,
    path::Path,
};

use entities::Document;
use serde::Serialize;

pub mod parse;

pub mod blocks;
pub mod entities;
pub mod layout;

pub mod ocr;

const IMAGE_PADDING: u32 = 5;

fn sanitize_doc_name(doc_name: &str) -> String {
    doc_name
        .chars()
        .filter_map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                Some(c)
            } else if c.is_whitespace() {
                None
            } else {
                Some('-')
            }
        })
        .collect::<String>()
}

fn save_doc_images<P: AsRef<Path> + Serialize>(
    imgs_dir: &Path,
    doc: &Document<P>,
) -> anyhow::Result<()> {
    let mut image_id = 0;
    for block in doc.blocks.iter() {
        match &block.kind {
            blocks::BlockType::Image(_) => {
                let page_id = block.pages_id.first().unwrap();
                match doc.pages.iter().find(|&p| p.id == *page_id) {
                    Some(page) => {
                        assert!(page.height as u32 > 0);
                        assert!(page.width as u32 > 0);

                        let x = (block.bbox.x0 - IMAGE_PADDING as f32) as u32;
                        let y = (block.bbox.y0 - IMAGE_PADDING as f32) as u32;
                        let width = (block.bbox.width().max(1.0) as u32 + 2 * IMAGE_PADDING)
                            .min(page.width as u32);
                        let height = (block.bbox.height().max(1.0) as u32 + 2 * IMAGE_PADDING)
                            .min(page.height as u32);

                        let crop = page.image.clone().crop(x, y, width, height);

                        let output_file =
                            imgs_dir.join(format!("page_{}_img_{}.png", page_id, image_id));
                        image_id += 1;
                        crop.save(output_file)?;
                    }
                    None => continue,
                }
            }
            blocks::BlockType::Table => todo!(),
            _ => continue,
        }
    }
    Ok(())
}

pub fn save_parsed_document<P: AsRef<Path> + Serialize>(
    doc: &Document<P>,
    output_dir: Option<P>,
    save_imgs: bool,
) -> anyhow::Result<()> {
    let result_dir_name = format!("{}-results", sanitize_doc_name(&doc.doc_name));
    let res_dir_path = match output_dir {
        Some(p) => p.as_ref().to_owned().join(&result_dir_name),
        None => {
            if std::fs::create_dir(&result_dir_name).is_err() {
                std::fs::remove_dir_all(&result_dir_name)?;
                std::fs::create_dir(&result_dir_name)?;
            };
            format!("./{}", &result_dir_name).into()
        }
    };
    // Save json
    let file_out = res_dir_path.join("result.json");
    let file = File::create(&file_out)?;
    let mut writer = BufWriter::new(file);
    let doc_json = serde_json::to_string(&doc)?;
    writer.write_all(doc_json.as_bytes())?;

    if save_imgs {
        save_doc_images(&res_dir_path, doc).context("can't save the doc images")?;
    }

    if let Some(dbg_path) = &doc.debug_path {
        println!(
            "{} Debug output saved in: {}",
            "ℹ".yellow().bold(),
            dbg_path.display().to_string().yellow().underline()
        );
    }

    println!(
        "{} Results saved in: {}",
        "✓".green().bold(),
        res_dir_path.display().to_string().cyan().underline()
    );

    Ok(())
}

pub(crate) fn chunk_docs_range(
    n_pages: usize,
    n_workers: usize,
    page_range: Option<Range<usize>>,
) -> Vec<Range<usize>> {
    let page_range: Vec<usize> = match page_range {
        Some(range) => range.collect(),
        None => (0..n_pages).collect(),
    };

    if page_range.len() > n_workers {
        page_range
            .chunks(page_range.len() / n_workers)
            .map(|c| (*c.first().unwrap()..*c.last().unwrap()))
            .collect()
    } else {
        vec![(0..n_pages)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_chunk_docs_range_with_more_pages_than_workers() {
        let n_pages = 10;
        let n_workers = 3;
        let result = chunk_docs_range(n_pages, n_workers, None);

        assert_eq!(result.len(), 3); // Should split into 3 chunks
        assert_eq!(result[0], 0..3);
        assert_eq!(result[1], 3..6);
        assert_eq!(result[2], 6..9);
    }

    #[test]
    fn test_chunk_docs_range_with_fewer_pages_than_workers() {
        let n_pages = 3;
        let n_workers = 5;
        let result = chunk_docs_range(n_pages, n_workers, None);

        assert_eq!(result.len(), 1); // Should return single range
        assert_eq!(result[0], 0..3);
    }

    #[test]
    fn test_chunk_docs_range_with_custom_range() {
        let n_pages = 10;
        let n_workers = 2;
        let page_range = Some(2..8); // Pages 2 to 7
        let result = chunk_docs_range(n_pages, n_workers, page_range);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 2..4);
        assert_eq!(result[1], 4..7);
    }

    #[test]
    fn test_chunk_docs_range_with_single_page() {
        let n_pages = 1;
        let n_workers = 1;
        let result = chunk_docs_range(n_pages, n_workers, None);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0..1);
    }

    #[test]
    fn test_chunk_docs_range_with_zero_pages() {
        let n_pages = 0;
        let n_workers = 1;
        let result = chunk_docs_range(n_pages, n_workers, None);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0..0);
    }

    #[test]
    fn test_chunk_docs_range_with_equal_pages_and_workers() {
        let n_pages = 4;
        let n_workers = 4;
        let result = chunk_docs_range(n_pages, n_workers, None);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0..4);
    }
}
