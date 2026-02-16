use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::blocks::{TableAlgorithm, TableBlock};
use crate::entities::{BBox, PDFPath};

#[tracing::instrument(skip(lines, paths), fields(lines_count = lines.len(), paths_count = paths.len()))]
pub fn parse_table_lattice(
    table_id_counter: Arc<AtomicUsize>,
    lines: &[crate::entities::Line],
    paths: &[PDFPath],
    table_bbox: &BBox,
) -> Option<TableBlock> {
    let padding = 5.0;
    let mut h_lines = Vec::new();
    let mut v_lines = Vec::new();

    for path in paths {
        for segment in &path.segments {
            match segment {
                crate::entities::Segment::Line { start, end } => {
                    let (x1, y1) = *start;
                    let (x2, y2) = *end;

                    // Horizontal line
                    if (y1 - y2).abs() < 1.0 {
                        let y = (y1 + y2) / 2.0;
                        if y >= table_bbox.y0 - padding && y <= table_bbox.y1 + padding {
                            let x_min = x1.min(x2);
                            let x_max = x1.max(x2);
                            if x_min < table_bbox.x1 + padding && x_max > table_bbox.x0 - padding {
                                h_lines.push((y, x_min, x_max));
                            }
                        }
                    }
                    // Vertical line
                    else if (x1 - x2).abs() < 1.0 {
                        let x = (x1 + x2) / 2.0;
                        if x >= table_bbox.x0 - padding && x <= table_bbox.x1 + padding {
                            let y_min = y1.min(y2);
                            let y_max = y1.max(y2);
                            if y_min < table_bbox.y1 + padding && y_max > table_bbox.x0 - padding {
                                v_lines.push((x, y_min, y_max));
                            }
                        }
                    }
                }
                crate::entities::Segment::Rect { bbox } => {
                    if table_bbox.intersection(bbox) > 0.0 {
                        h_lines.push((bbox.y0, bbox.x0, bbox.x1));
                        h_lines.push((bbox.y1, bbox.x0, bbox.x1));
                        v_lines.push((bbox.x0, bbox.y0, bbox.y1));
                        v_lines.push((bbox.x1, bbox.y0, bbox.y1));
                    }
                }
            }
        }
    }

    tracing::debug!(
        "Lattice - BBox {:?} - segments: H={}, V={}",
        table_bbox,
        h_lines.len(),
        v_lines.len()
    );

    if h_lines.is_empty() || v_lines.is_empty() {
        return None;
    }

    // Simple clustering
    h_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    v_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut unique_h = Vec::new();
    if !h_lines.is_empty() {
        let mut curr = h_lines[0];
        for next in h_lines.iter().skip(1) {
            if (next.0 - curr.0).abs() < 2.0 {
                curr.1 = curr.1.min(next.1);
                curr.2 = curr.2.max(next.2);
            } else {
                unique_h.push(curr);
                curr = *next;
            }
        }
        unique_h.push(curr);
    }

    let mut unique_v = Vec::new();
    if !v_lines.is_empty() {
        let mut curr = v_lines[0];
        for next in v_lines.iter().skip(1) {
            if (next.0 - curr.0).abs() < 2.0 {
                curr.1 = curr.1.min(next.1);
                curr.2 = curr.2.max(next.2);
            } else {
                unique_v.push(curr);
                curr = *next;
            }
        }
        unique_v.push(curr);
    }

    let h_coords: Vec<f32> = unique_h.iter().map(|l| l.0).collect();
    let v_coords: Vec<f32> = unique_v.iter().map(|l| l.0).collect();

    if h_coords.len() < 2 || v_coords.len() < 2 {
        tracing::debug!(
            "Lattice failed: insufficient grid lines. H={:?}, V={:?}",
            h_coords,
            v_coords
        );
        return None;
    }
    tracing::debug!("Lattice Grid: H={:?}, V={:?}", h_coords, v_coords);

    // Pre-filter lines that intersect the table
    let table_lines: Vec<_> = lines
        .iter()
        .filter(|l| table_bbox.intersection(&l.bbox) > 0.0)
        .collect();

    let mut rows = Vec::new();

    // Matrix to track visited grid cells (for spanning)
    let num_rows = h_coords.len() - 1;
    let num_cols = v_coords.len() - 1;
    let mut visited = vec![vec![false; num_cols]; num_rows];

    for i in 0..num_rows {
        let y0 = h_coords[i];
        let y1 = h_coords[i + 1];
        let mut row_cells = Vec::new();

        for j in 0..num_cols {
            if visited[i][j] {
                continue;
            }

            // Determine horizontal span
            let mut col_span = 1;
            while j + col_span < num_cols {
                let next_x = v_coords[j + col_span];
                // Is there a vertical line at next_x covering this row?
                let has_boundary = v_lines.iter().any(|(sx, ymin, ymax)| {
                    if (sx - next_x).abs() > 1.5 {
                        return false;
                    }
                    let overlap_start = y0.max(*ymin);
                    let overlap_end = y1.min(*ymax);
                    let overlap = (overlap_end - overlap_start).max(0.0);
                    overlap > (y1 - y0) * 0.1
                });

                if has_boundary {
                    break;
                }
                col_span += 1;
            }

            // Determine vertical span
            let mut row_span = 1;
            while i + row_span < num_rows {
                let next_y = h_coords[i + row_span];
                // Is there a horizontal line at next_y covering this column range?
                let x0 = v_coords[j];
                let x1 = v_coords[j + col_span];
                let has_boundary = h_lines.iter().any(|(sy, xmin, xmax)| {
                    if (sy - next_y).abs() > 1.5 {
                        return false;
                    }
                    let overlap_start = x0.max(*xmin);
                    let overlap_end = x1.min(*xmax);
                    let overlap = (overlap_end - overlap_start).max(0.0);
                    overlap > (x1 - x0) * 0.1
                });

                if has_boundary {
                    break;
                }
                row_span += 1;
            }

            // Mark grid cells as visited
            for r in i..i + row_span {
                for c in j..j + col_span {
                    visited[r][c] = true;
                }
            }

            let x0 = v_coords[j];
            let y0 = h_coords[i];
            let x1 = v_coords[j + col_span];
            let y1 = h_coords[i + row_span];
            let cell_bbox = BBox { x0, y0, x1, y1 };

            let mut cell_text = String::new();
            for line in &table_lines {
                if cell_bbox.intersection(&line.bbox) > line.bbox.area() * 0.5 {
                    if !cell_text.is_empty() {
                        cell_text.push(' ');
                    }
                    cell_text.push_str(&line.text);
                }
            }

            row_cells.push(crate::blocks::TableCell {
                content_ids: vec![],
                text: cell_text,
                row_span: row_span as u8,
                col_span: col_span as u8,
                bbox: cell_bbox,
            });
        }

        if !row_cells.is_empty() {
            let mut row_bbox = row_cells[0].bbox.clone();
            for cell in &row_cells[1..] {
                row_bbox.merge(&cell.bbox);
            }
            rows.push(crate::blocks::TableRow {
                cells: row_cells,
                is_header: false,
                bbox: row_bbox,
            });
        }
    }

    if rows.is_empty() {
        return None;
    }

    let table_id = table_id_counter.fetch_add(1, Ordering::SeqCst);
    Some(TableBlock {
        id: table_id,
        caption: None,
        rows,
        has_borders: true,
        algorithm: TableAlgorithm::Lattice,
    })
}
