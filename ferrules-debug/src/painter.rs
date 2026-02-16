use crate::inspector::{
    InspectorBlock, InspectorCell, InspectorElement, InspectorItem, InspectorLayout,
    InspectorTableDetails,
};
use ferrules_core::blocks::ArchivedBlockType;
use ferrules_core::debug_info::ArchivedDebugPage;
use ferrules_core::entities::{ArchivedElementType, ArchivedSegment};
use iced::mouse::{self, Cursor};
use iced::widget::canvas::{self, Frame, Geometry, Image, Path, Program, Stroke, Text};
use iced::widget::image;
use iced::{Color, Font, Point, Rectangle, Renderer, Theme, Vector};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PainterMode {
    Image,
    Overlay,
}

pub struct PagePainter<'a> {
    pub page: &'a ArchivedDebugPage,
    pub image_handle: image::Handle,
    pub zoom: f32,
    pub offset: Vector,
    pub mode: PainterMode,

    // Config
    pub show_native: bool,
    pub show_layout: bool,
    pub show_ocr: bool,
    pub show_elements: bool,
    pub show_blocks: bool,
    pub show_paths: bool,
    pub show_tables: bool,

    // Selection
    pub selected_item: Option<InspectorItem>,
}

pub struct State {
    pub last_cursor_position: Point,
    pub is_panning: bool,
    pub start_drag_position: Option<Point>,
    pub hovered_info: Option<InspectorItem>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            last_cursor_position: Point::ORIGIN,
            is_panning: false,
            start_drag_position: None,
            hovered_info: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum CanvasMessage {
    Hovered(Option<InspectorItem>),
    Clicked(Option<InspectorItem>),
    ZoomChanged(f32, Point),
    OffsetChanged(Vector),
}

impl<'a> Program<CanvasMessage> for PagePainter<'a> {
    type State = State;

    fn update(
        &self,
        state: &mut Self::State,
        event: canvas::Event,
        bounds: Rectangle,
        cursor: Cursor,
    ) -> (canvas::event::Status, Option<CanvasMessage>) {
        if let PainterMode::Image = self.mode {
            return (canvas::event::Status::Ignored, None);
        }

        let cursor_position = cursor.position_in(bounds);

        match event {
            canvas::Event::Mouse(mouse_event) => match mouse_event {
                mouse::Event::WheelScrolled { delta } => {
                    if let Some(pos) = cursor_position {
                        let factor = match delta {
                            mouse::ScrollDelta::Lines { y, .. } => y * 0.1,
                            mouse::ScrollDelta::Pixels { y, .. } => y * 0.001,
                        };
                        return (
                            canvas::event::Status::Captured,
                            Some(CanvasMessage::ZoomChanged(factor, pos)),
                        );
                    }
                }
                mouse::Event::ButtonPressed(mouse::Button::Left) => {
                    if let Some(pos) = cursor_position {
                        state.is_panning = true;
                        state.last_cursor_position = pos;
                        state.start_drag_position = Some(pos);
                        return (canvas::event::Status::Captured, None);
                    }
                }
                mouse::Event::ButtonReleased(mouse::Button::Left) => {
                    let mut message = None;
                    if state.is_panning {
                        state.is_panning = false;
                        if let Some(start_pos) = state.start_drag_position {
                            if let Some(current_pos) = cursor_position {
                                let dist = start_pos.distance(current_pos);
                                if dist < 5.0 {
                                    // It's a click!
                                    let item = self.get_hover_detailed(current_pos, bounds);
                                    message = Some(CanvasMessage::Clicked(item));
                                }
                            }
                        }
                    }
                    state.start_drag_position = None;
                    return (canvas::event::Status::Captured, message);
                }
                mouse::Event::CursorMoved { .. } => {
                    if let Some(pos) = cursor_position {
                        if state.is_panning {
                            let delta = pos - state.last_cursor_position;
                            state.last_cursor_position = pos;
                            return (
                                canvas::event::Status::Captured,
                                Some(CanvasMessage::OffsetChanged(Vector::new(delta.x, delta.y))),
                            );
                        }

                        let new_hover = self.get_hover_detailed(pos, bounds);
                        if new_hover != state.hovered_info {
                            state.hovered_info = new_hover.clone();
                            return (
                                canvas::event::Status::Captured,
                                Some(CanvasMessage::Hovered(new_hover)),
                            );
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }

        (canvas::event::Status::Ignored, None)
    }

    fn draw(
        &self,
        state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: Cursor,
    ) -> Vec<Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());

        let fit_scale_x = bounds.width / self.page.width;
        let fit_scale_y = bounds.height / self.page.height;
        let fit_scale = fit_scale_x.min(fit_scale_y);

        let final_scale = fit_scale * self.zoom;
        let center_offset_x = (bounds.width - self.page.width * fit_scale) / 2.0;
        let center_offset_y = (bounds.height - self.page.height * fit_scale) / 2.0;
        let total_offset_x = center_offset_x + self.offset.x;
        let total_offset_y = center_offset_y + self.offset.y;

        match self.mode {
            PainterMode::Image => {
                let image_rect = self.to_rect(
                    0.0,
                    0.0,
                    self.page.width,
                    self.page.height,
                    final_scale,
                    total_offset_x,
                    total_offset_y,
                );
                let img = Image::new(self.image_handle.clone());
                frame.draw_image(image_rect, img);
            }
            PainterMode::Overlay => {
                // Catppuccin Mocha Palette (Now using theme.rs)
                if self.show_paths {
                    for path in self.page.paths.as_slice() {
                        for segment in path.segments.as_slice() {
                            match segment {
                                ArchivedSegment::Line { start, end } => {
                                    let p1 = Point::new(
                                        start.0 * final_scale + total_offset_x,
                                        start.1 * final_scale + total_offset_y,
                                    );
                                    let p2 = Point::new(
                                        end.0 * final_scale + total_offset_x,
                                        end.1 * final_scale + total_offset_y,
                                    );
                                    frame.stroke(
                                        &Path::line(p1, p2),
                                        Stroke::default()
                                            .with_color(crate::theme::OVERLAY0_FADED.into())
                                            .with_width(0.8),
                                    );
                                }
                                ArchivedSegment::Rect { bbox } => {
                                    let rect = self.to_rect(
                                        bbox.x0,
                                        bbox.y0,
                                        bbox.x1,
                                        bbox.y1,
                                        final_scale,
                                        total_offset_x,
                                        total_offset_y,
                                    );
                                    frame.stroke(
                                        &Path::rectangle(rect.position(), rect.size()),
                                        Stroke::default()
                                            .with_color(crate::theme::OVERLAY0_FADED.into())
                                            .with_width(0.8),
                                    );
                                }
                            }
                        }
                    }
                }

                if self.show_native {
                    for line in self.page.native_lines.as_slice() {
                        let rect = self.to_rect(
                            line.bbox.x0,
                            line.bbox.y0,
                            line.bbox.x1,
                            line.bbox.y1,
                            final_scale,
                            total_offset_x,
                            total_offset_y,
                        );
                        frame.stroke(
                            &Path::rectangle(rect.position(), rect.size()),
                            Stroke::default()
                                .with_color(crate::theme::RED)
                                .with_width(0.8),
                        );
                    }
                }

                if self.show_ocr {
                    for line in self.page.ocr_lines.as_slice() {
                        let rect = self.to_rect(
                            line.bbox.x0,
                            line.bbox.y0,
                            line.bbox.x1,
                            line.bbox.y1,
                            final_scale,
                            total_offset_x,
                            total_offset_y,
                        );
                        frame.stroke(
                            &Path::rectangle(rect.position(), rect.size()),
                            Stroke::default()
                                .with_color(crate::theme::BLUE)
                                .with_width(0.8),
                        );
                    }
                }

                if self.show_layout {
                    for bbox in self.page.layout_bboxes.as_slice() {
                        let rect = self.to_rect(
                            bbox.bbox.x0,
                            bbox.bbox.y0,
                            bbox.bbox.x1,
                            bbox.bbox.y1,
                            final_scale,
                            total_offset_x,
                            total_offset_y,
                        );
                        frame.stroke(
                            &Path::rectangle(rect.position(), rect.size()),
                            Stroke::default()
                                .with_color(crate::theme::GREEN)
                                .with_width(1.2),
                        );
                    }
                }

                if self.show_elements {
                    for element in self.page.elements.as_slice() {
                        let rect = self.to_rect(
                            element.bbox.x0,
                            element.bbox.y0,
                            element.bbox.x1,
                            element.bbox.y1,
                            final_scale,
                            total_offset_x,
                            total_offset_y,
                        );
                        frame.stroke(
                            &Path::rectangle(rect.position(), rect.size()),
                            Stroke::default()
                                .with_color(crate::theme::PEACH)
                                .with_width(1.5),
                        );
                    }
                }

                if self.show_blocks {
                    for block in self.page.blocks.as_slice() {
                        let rect = self.to_rect(
                            block.bbox.x0,
                            block.bbox.y0,
                            block.bbox.x1,
                            block.bbox.y1,
                            final_scale,
                            total_offset_x,
                            total_offset_y,
                        );
                        frame.stroke(
                            &Path::rectangle(rect.position(), rect.size()),
                            Stroke::default()
                                .with_color(crate::theme::MAUVE)
                                .with_width(2.0),
                        );
                    }
                }

                let active_selection = self.selected_item.as_ref().or(state.hovered_info.as_ref());

                if let Some(selection) = active_selection {
                    if let InspectorItem::Selection {
                        block,
                        element,
                        layout,
                        cell,
                    } = selection
                    {
                        // Choose a primary bbox for highlight and label (priority to Element -> Block -> Layout)
                        let primary_bbox = element
                            .as_ref()
                            .map(|e| e.bbox)
                            .or_else(|| block.as_ref().map(|b| b.bbox))
                            .or_else(|| layout.as_ref().map(|l| l.bbox));

                        if let Some(bbox) = primary_bbox {
                            let rect = self.to_rect(
                                bbox[0],
                                bbox[1],
                                bbox[2],
                                bbox[3],
                                final_scale,
                                total_offset_x,
                                total_offset_y,
                            );

                            frame.fill(
                                &Path::rectangle(rect.position(), rect.size()),
                                crate::theme::SELECTION_BG,
                            );
                            frame.stroke(
                                &Path::rectangle(rect.position(), rect.size()),
                                Stroke::default()
                                    .with_color(crate::theme::SELECTION_BORDER)
                                    .with_width(4.0),
                            );

                            // Label etiquette
                            let label_text = if let Some(e) = element {
                                e.kind.clone()
                            } else if let Some(b) = block {
                                b.kind.clone()
                            } else if let Some(l) = layout {
                                l.label.clone()
                            } else {
                                String::new()
                            };

                            if !label_text.is_empty() {
                                let label_bg_color = if element.is_some() {
                                    crate::theme::PEACH
                                } else if block.is_some() {
                                    crate::theme::MAUVE
                                } else if layout.is_some() {
                                    crate::theme::GREEN
                                } else {
                                    crate::theme::SURFACE0
                                };

                                let label_size = 12.0;
                                let padding = 6.0;
                                let text_width = label_text.len() as f32 * 7.5;

                                frame.fill(
                                    &Path::rectangle(
                                        Point::new(rect.x, rect.y - 26.0),
                                        [text_width + padding * 2.0, label_size + padding * 1.5]
                                            .into(),
                                    ),
                                    label_bg_color,
                                );

                                frame.fill_text(Text {
                                    content: label_text,
                                    position: Point::new(rect.x + padding, rect.y - 22.0),
                                    color: Color::WHITE,
                                    size: label_size.into(),
                                    font: Font {
                                        weight: iced::font::Weight::Bold,
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                });
                            }
                        }

                        if let Some(c) = cell {
                            let rect = self.to_rect(
                                c.bbox[0],
                                c.bbox[1],
                                c.bbox[2],
                                c.bbox[3],
                                final_scale,
                                total_offset_x,
                                total_offset_y,
                            );

                            frame.stroke(
                                &Path::rectangle(rect.position(), rect.size()),
                                Stroke::default()
                                    .with_color(crate::theme::TABLE_SELECTION)
                                    .with_width(2.0),
                            );
                        }
                    }
                }

                if self.show_tables {
                    for block in self.page.blocks.as_slice() {
                        if let ArchivedBlockType::Table(table) = &block.kind {
                            // Draw table background/overlay
                            let table_rect = self.to_rect(
                                block.bbox.x0,
                                block.bbox.y0,
                                block.bbox.x1,
                                block.bbox.y1,
                                final_scale,
                                total_offset_x,
                                total_offset_y,
                            );

                            // Draw cells
                            for row in table.rows.as_slice() {
                                for cell in row.cells.as_slice() {
                                    let cell_rect = self.to_rect(
                                        cell.bbox.x0,
                                        cell.bbox.y0,
                                        cell.bbox.x1,
                                        cell.bbox.y1,
                                        final_scale,
                                        total_offset_x,
                                        total_offset_y,
                                    );

                                    frame.fill(
                                        &Path::rectangle(cell_rect.position(), cell_rect.size()),
                                        Color {
                                            a: 0.1,
                                            ..crate::theme::PEACH
                                        },
                                    );

                                    frame.stroke(
                                        &Path::rectangle(cell_rect.position(), cell_rect.size()),
                                        Stroke::default()
                                            .with_color(Color {
                                                a: 0.3,
                                                ..crate::theme::PEACH
                                            })
                                            .with_width(0.5),
                                    );
                                }
                            }

                            // Algorithm hint label
                            let algo_text = match table.algorithm {
                                ferrules_core::blocks::ArchivedTableAlgorithm::Lattice => "Lattice",
                                ferrules_core::blocks::ArchivedTableAlgorithm::Stream => "Stream",
                                ferrules_core::blocks::ArchivedTableAlgorithm::Vision => "Vision",
                                _ => "Unknown",
                            };
                            let algo_color = match table.algorithm {
                                ferrules_core::blocks::ArchivedTableAlgorithm::Lattice => {
                                    crate::theme::GREEN
                                }
                                ferrules_core::blocks::ArchivedTableAlgorithm::Stream => {
                                    crate::theme::BLUE
                                }
                                ferrules_core::blocks::ArchivedTableAlgorithm::Vision => {
                                    crate::theme::MAUVE
                                }
                                _ => crate::theme::OVERLAY0,
                            };

                            frame.fill_text(Text {
                                content: algo_text.to_string(),
                                position: Point::new(table_rect.x, table_rect.y - 12.0),
                                color: algo_color,
                                size: 10.0.into(),
                                font: Font {
                                    weight: iced::font::Weight::Bold,
                                    ..Default::default()
                                },
                                ..Default::default()
                            });
                        }
                    }
                }
            }
        }

        vec![frame.into_geometry()]
    }
}

impl<'a> PagePainter<'a> {
    fn to_rect(
        &self,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        scale: f32,
        off_x: f32,
        off_y: f32,
    ) -> Rectangle {
        let x = x0.min(x1);
        let y = y0.min(y1);
        let w = (x1 - x0).abs();
        let h = (y1 - y0).abs();

        Rectangle {
            x: x * scale + off_x,
            y: y * scale + off_y,
            width: w.max(0.1) * scale,
            height: h.max(0.1) * scale,
        }
    }

    fn get_hover_detailed(&self, pos: Point, bounds: Rectangle) -> Option<InspectorItem> {
        let fit_scale_x = bounds.width / self.page.width;
        let fit_scale_y = bounds.height / self.page.height;
        let fit_scale = fit_scale_x.min(fit_scale_y);
        let final_scale = fit_scale * self.zoom;
        let center_offset_x = (bounds.width - self.page.width * fit_scale) / 2.0;
        let center_offset_y = (bounds.height - self.page.height * fit_scale) / 2.0;
        let total_offset_x = center_offset_x + self.offset.x;
        let total_offset_y = center_offset_y + self.offset.y;

        let px = (pos.x - total_offset_x) / final_scale;
        let py = (pos.y - total_offset_y) / final_scale;

        let mut hovered_block = None;
        let mut hovered_element = None;
        let mut hovered_layout = None;
        let mut hovered_cell = None;

        if self.show_blocks || self.show_tables {
            for block in self.page.blocks.as_slice() {
                let is_table = matches!(block.kind, ArchivedBlockType::Table(_));

                if !self.show_blocks && !is_table {
                    continue;
                }

                if px >= block.bbox.x0
                    && px <= block.bbox.x1
                    && py >= block.bbox.y0
                    && py <= block.bbox.y1
                {
                    let block_text = match &block.kind {
                        ArchivedBlockType::TextBlock(t) => t.text.to_string(),
                        ArchivedBlockType::Header(h) => h.text.to_string(),
                        ArchivedBlockType::Footer(f) => f.text.to_string(),
                        ArchivedBlockType::Title(t) => t.text.to_string(),
                        ArchivedBlockType::ListBlock(l) => l.items.join("\n"),
                        _ => String::new(),
                    };
                    let block_kind = match &block.kind {
                        ArchivedBlockType::Header(_) => "Header",
                        ArchivedBlockType::Footer(_) => "Footer",
                        ArchivedBlockType::Title(_) => "Title",
                        ArchivedBlockType::ListBlock(_) => "List",
                        ArchivedBlockType::TextBlock(_) => "Text",
                        ArchivedBlockType::Image(_) => "Image",
                        ArchivedBlockType::Table(_) => "Table",
                    };

                    let mut table_details = None;

                    if let ArchivedBlockType::Table(table) = &block.kind {
                        table_details = Some(InspectorTableDetails {
                            algorithm: match table.algorithm {
                                ferrules_core::blocks::ArchivedTableAlgorithm::Lattice => {
                                    "Lattice".to_string()
                                }
                                ferrules_core::blocks::ArchivedTableAlgorithm::Stream => {
                                    "Stream".to_string()
                                }
                                ferrules_core::blocks::ArchivedTableAlgorithm::Vision => {
                                    "Vision".to_string()
                                }
                                _ => "Unknown".to_string(),
                            },
                            rows: table.rows.len(),
                            cols: if table.rows.is_empty() {
                                0
                            } else {
                                table.rows[0].cells.len()
                            },
                        });

                        // Check for cell hover
                        for (r_idx, row) in table.rows.iter().enumerate() {
                            for (c_idx, cell) in row.cells.iter().enumerate() {
                                if px >= cell.bbox.x0
                                    && px <= cell.bbox.x1
                                    && py >= cell.bbox.y0
                                    && py <= cell.bbox.y1
                                {
                                    hovered_cell = Some(InspectorCell {
                                        row_idx: r_idx,
                                        col_idx: c_idx,
                                        row_span: cell.row_span,
                                        col_span: cell.col_span,
                                        text: cell.text.to_string(),
                                        bbox: [
                                            cell.bbox.x0,
                                            cell.bbox.y0,
                                            cell.bbox.x1,
                                            cell.bbox.y1,
                                        ],
                                    });
                                    break;
                                }
                            }
                            if hovered_cell.is_some() {
                                break;
                            }
                        }
                    }

                    hovered_block = Some(InspectorBlock {
                        id: block.id as usize,
                        kind: block_kind.to_string(),
                        bbox: [block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1],
                        pages: block.pages_id.iter().map(|&id| id as usize).collect(),
                        text: block_text,
                        table_details,
                    });
                }
            }
        }

        if self.show_elements || self.show_tables {
            for element in self.page.elements.as_slice() {
                let is_table = matches!(element.kind, ArchivedElementType::Table(_));
                if !self.show_elements && !is_table {
                    continue;
                }

                if px >= element.bbox.x0
                    && px <= element.bbox.x1
                    && py >= element.bbox.y0
                    && py <= element.bbox.y1
                {
                    let elem_type = match &element.kind {
                        ArchivedElementType::Header => "Header",
                        ArchivedElementType::FootNote => "FootNote",
                        ArchivedElementType::Footer => "Footer",
                        ArchivedElementType::Text => "Text",
                        ArchivedElementType::Title => "Title",
                        ArchivedElementType::Subtitle => "Subtitle",
                        ArchivedElementType::ListItem => "ListItem",
                        ArchivedElementType::Caption => "Caption",
                        ArchivedElementType::Image => "Image",
                        ArchivedElementType::Table(_) => "Table",
                    };
                    hovered_element = Some(InspectorElement {
                        id: element.id as usize,
                        kind: elem_type.to_string(),
                        bbox: [
                            element.bbox.x0,
                            element.bbox.y0,
                            element.bbox.x1,
                            element.bbox.y1,
                        ],
                        layout_ref: element.layout_block_id,
                        text: element.text_block.text.to_string(),
                    });
                    break;
                }
            }
        }

        if self.show_layout {
            for lay in self.page.layout_bboxes.as_slice() {
                if px >= lay.bbox.x0 && px <= lay.bbox.x1 && py >= lay.bbox.y0 && py <= lay.bbox.y1
                {
                    hovered_layout = Some(InspectorLayout {
                        id: lay.id,
                        label: lay.label.to_string(),
                        proba: lay.proba,
                        bbox: [lay.bbox.x0, lay.bbox.y0, lay.bbox.x1, lay.bbox.y1],
                    });
                    break;
                }
            }
        }

        if hovered_block.is_some() || hovered_element.is_some() || hovered_layout.is_some() {
            Some(InspectorItem::Selection {
                block: hovered_block,
                element: hovered_element,
                layout: hovered_layout,
                cell: hovered_cell,
            })
        } else {
            None
        }
    }
}
