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

pub(crate) fn doc_chunks(
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
            .chunks(n_pages / n_workers)
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
    fn test_doc_chunks_with_more_pages_than_workers() {
        let chunks = doc_chunks(10, 2, None);
        assert_eq!(chunks, vec![0..4, 5..9]); // Should split into roughly equal chunks
    }

    #[test]
    fn test_doc_chunks_with_fewer_pages_than_workers() {
        let chunks = doc_chunks(3, 4, None);
        assert_eq!(chunks, vec![0..3]); // Should return single chunk
    }

    #[test]
    fn test_doc_chunks_with_custom_page_range() {
        let chunks = doc_chunks(10, 2, Some(2..8));
        assert_eq!(chunks, vec![2..4, 5..7]); // Should respect custom range
    }

    #[test]
    fn test_doc_chunks_with_single_worker() {
        let chunks = doc_chunks(5, 1, None);
        assert_eq!(chunks, vec![0..5]); // Should return single chunk
    }

    #[test]
    fn test_doc_chunks_with_empty_range() {
        let chunks = doc_chunks(0, 2, None);
        assert_eq!(chunks, vec![0..0]); // Should handle empty range
    }

    #[test]
    fn test_doc_chunks_equal_pages_and_workers() {
        let chunks = doc_chunks(4, 4, None);
        assert_eq!(chunks, vec![0..4]); // Should return single chunk when pages equals workers
    }
}
