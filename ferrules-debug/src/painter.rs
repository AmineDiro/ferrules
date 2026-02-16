use crate::inspector::InspectorItem;
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
}

pub struct State {
    pub last_cursor_position: Point,
    pub is_panning: bool,
    pub hovered_info: Option<InspectorItem>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            last_cursor_position: Point::ORIGIN,
            is_panning: false,
            hovered_info: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum CanvasMessage {
    Hovered(Option<InspectorItem>),
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
                        return (canvas::event::Status::Captured, None);
                    }
                }
                mouse::Event::ButtonReleased(mouse::Button::Left) => {
                    state.is_panning = false;
                    return (canvas::event::Status::Captured, None);
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
                // Vibrant Dracula Palette
                let red = Color::from_rgb(1.0, 0.33, 0.33); // #ff5555
                let green = Color::from_rgb(0.31, 1.0, 0.44); // #50fa7b
                let purple = Color::from_rgb(0.74, 0.57, 0.97); // #bd93f9
                let cyan = Color::from_rgb(0.54, 0.88, 1.0); // #8be9fd
                let orange = Color::from_rgb(1.0, 0.72, 0.42); // #ffb86c
                let path_col = Color::from_rgba(0.9, 0.9, 0.9, 0.2);

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
                                        Stroke::default().with_color(path_col).with_width(0.8),
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
                                        Stroke::default().with_color(path_col).with_width(0.8),
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
                            Stroke::default().with_color(red).with_width(0.8),
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
                            Stroke::default().with_color(cyan).with_width(0.8),
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
                            Stroke::default().with_color(green).with_width(1.2),
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
                            Stroke::default().with_color(orange).with_width(1.5),
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
                            Stroke::default().with_color(purple).with_width(2.0),
                        );
                    }
                }

                if let Some(hover) = &state.hovered_info {
                    if let Some(bbox) = self.get_bbox_from_item(hover) {
                        let rect = self.to_rect(
                            bbox[0],
                            bbox[1],
                            bbox[2],
                            bbox[3],
                            final_scale,
                            total_offset_x,
                            total_offset_y,
                        );

                        // Highlight fill
                        frame.fill(
                            &Path::rectangle(rect.position(), rect.size()),
                            Color::from_rgba(1.0, 1.0, 1.0, 0.1),
                        );
                        // Highlight stroke
                        frame.stroke(
                            &Path::rectangle(rect.position(), rect.size()),
                            Stroke::default().with_color(Color::WHITE).with_width(2.0),
                        );

                        // Label etiquette
                        let label_text = self.get_label_from_item(hover);
                        if !label_text.is_empty() {
                            let label_bg_color = match hover {
                                InspectorItem::Block { .. } => purple,
                                InspectorItem::Element { .. } => orange,
                                InspectorItem::Layout { .. } => green,
                                _ => Color::from_rgb(0.2, 0.2, 0.2),
                            };

                            let label_size = 12.0;
                            let padding = 6.0;
                            let text_width = label_text.len() as f32 * 7.5; // Approximation

                            // Draw etiquette box
                            frame.fill(
                                &Path::rectangle(
                                    Point::new(rect.x, rect.y - 26.0),
                                    [text_width + padding * 2.0, label_size + padding * 1.5].into(),
                                ),
                                label_bg_color,
                            );

                            // Draw text on etiquette
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

    fn get_bbox_from_item(&self, item: &InspectorItem) -> Option<[f32; 4]> {
        match item {
            InspectorItem::Block { bbox, .. } => Some(*bbox),
            InspectorItem::Element { bbox, .. } => Some(*bbox),
            InspectorItem::Layout { bbox, .. } => Some(*bbox),
            InspectorItem::None => None,
        }
    }

    fn get_label_from_item(&self, item: &InspectorItem) -> String {
        match item {
            InspectorItem::Block { kind, .. } => kind.clone(),
            InspectorItem::Element { kind, .. } => kind.clone(),
            InspectorItem::Layout { label, .. } => label.clone(),
            InspectorItem::None => String::new(),
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

        // Hit testing order: Blocks -> Elements -> Layout
        if self.show_blocks {
            for block in self.page.blocks.as_slice() {
                if px >= block.bbox.x0
                    && px <= block.bbox.x1
                    && py >= block.bbox.y0
                    && py <= block.bbox.y1
                {
                    let block_type = match &block.kind {
                        ArchivedBlockType::Header(_) => "Header",
                        ArchivedBlockType::Footer(_) => "Footer",
                        ArchivedBlockType::Title(_) => "Title",
                        ArchivedBlockType::ListBlock(_) => "List",
                        ArchivedBlockType::TextBlock(_) => "Text",
                        ArchivedBlockType::Image(_) => "Image",
                        ArchivedBlockType::Table(_) => "Table",
                    };
                    return Some(InspectorItem::Block {
                        id: block.id as usize,
                        kind: block_type.to_string(),
                        bbox: [block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1],
                        pages: block.pages_id.iter().map(|&id| id as usize).collect(),
                    });
                }
            }
        }

        if self.show_elements {
            for element in self.page.elements.as_slice() {
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
                    return Some(InspectorItem::Element {
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
                }
            }
        }

        if self.show_layout {
            for lay in self.page.layout_bboxes.as_slice() {
                if px >= lay.bbox.x0 && px <= lay.bbox.x1 && py >= lay.bbox.y0 && py <= lay.bbox.y1
                {
                    return Some(InspectorItem::Layout {
                        id: lay.id,
                        label: lay.label.to_string(),
                        proba: lay.proba,
                        bbox: [lay.bbox.x0, lay.bbox.y0, lay.bbox.x1, lay.bbox.y1],
                    });
                }
            }
        }

        None
    }
}
