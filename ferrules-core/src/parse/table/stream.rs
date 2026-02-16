use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::blocks::{TableAlgorithm, TableBlock};
use crate::entities::BBox;
use crate::error::FerrulesError;

#[tracing::instrument(skip(lines), fields(lines_count = lines.len()))]
pub fn parse_table_stream(
    table_id_counter: Arc<AtomicUsize>,
    lines: &[crate::entities::Line],
    table_bbox: &BBox,
) -> Result<TableBlock, FerrulesError> {
    // 1. Filter lines within table_bbox
    let mut table_lines: Vec<_> = lines
        .iter()
        .filter(|l| table_bbox.contains(&l.bbox))
        .collect();

    // 2. Sort lines by Y (vertical)
    table_lines.sort_by(|a, b| a.bbox.y0.partial_cmp(&b.bbox.y0).unwrap());

    // 3. Group lines into rows based on Y overlap
    let mut rows = Vec::new();
    if table_lines.is_empty() {
        let table_id = table_id_counter.fetch_add(1, Ordering::SeqCst);
        return Ok(TableBlock {
            id: table_id,
            caption: None,
            rows: vec![],
            has_borders: false,
            algorithm: TableAlgorithm::Unknown,
        });
    }

    const ROW_OVERLAP_THRESHOLD: f32 = 0.5;

    let mut current_row_lines = vec![table_lines[0]];
    for line in table_lines.iter().skip(1) {
        let last_line = current_row_lines.last().unwrap();
        // NOTE: If the next line significantly overlaps vertically or is very close, it's the same row
        if line.bbox.y0 < last_line.bbox.y1 - last_line.bbox.height() * ROW_OVERLAP_THRESHOLD {
            current_row_lines.push(line);
        } else {
            rows.push(process_row_lines(&current_row_lines));
            current_row_lines = vec![line];
        }
    }
    rows.push(process_row_lines(&current_row_lines));

    let table_id = table_id_counter.fetch_add(1, Ordering::SeqCst);
    Ok(TableBlock {
        id: table_id,
        caption: None,
        rows,
        has_borders: false,
        algorithm: TableAlgorithm::Stream,
    })
}

fn process_row_lines(row_lines: &[&crate::entities::Line]) -> crate::blocks::TableRow {
    if row_lines.is_empty() {
        return crate::blocks::TableRow::default();
    }

    let mut sorted_lines = row_lines.to_vec();
    sorted_lines.sort_by(|a, b| a.bbox.x0.partial_cmp(&b.bbox.x0).unwrap());

    let mut cells = Vec::new();
    let mut current_cell_text = sorted_lines[0].text.clone();
    let mut current_cell_bbox = sorted_lines[0].bbox.clone();

    for line in sorted_lines.iter().skip(1) {
        // Threshold for horizontal gap between words/cells in a table
        // Usually tables have larger gaps than normal text
        let horizontal_gap = line.bbox.x0 - current_cell_bbox.x1;
        if horizontal_gap < 10.0 {
            current_cell_text.push(' ');
            current_cell_text.push_str(&line.text);
            current_cell_bbox.merge(&line.bbox);
        } else {
            cells.push(crate::blocks::TableCell {
                content_ids: vec![],
                text: current_cell_text.trim().to_string(),
                bbox: current_cell_bbox,
                col_span: 1,
                row_span: 1,
            });
            current_cell_text = line.text.clone();
            current_cell_bbox = line.bbox.clone();
        }
    }

    cells.push(crate::blocks::TableCell {
        content_ids: vec![],
        text: current_cell_text.trim().to_string(),
        bbox: current_cell_bbox,
        col_span: 1,
        row_span: 1,
    });

    let mut row_bbox = cells[0].bbox.clone();
    for cell in &cells[1..] {
        row_bbox.merge(&cell.bbox);
    }

    crate::blocks::TableRow {
        cells,
        bbox: row_bbox,
        is_header: false,
    }
}
