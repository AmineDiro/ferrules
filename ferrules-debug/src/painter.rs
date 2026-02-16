use ferrules_core::debug_info::ArchivedDebugPage;
use ferrules_core::entities::ArchivedSegment;
use iced::mouse::{self, Cursor};
use iced::widget::canvas::{self, Frame, Geometry, Image, Path, Program, Stroke};
use iced::widget::image;
use iced::{Color, Point, Rectangle, Renderer, Theme, Vector};

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
    pub hovered_info: Option<HoverDetailed>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HoverDetailed {
    pub title: String,
    pub details: String,
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
    Hovered(Option<HoverDetailed>),
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
        // Only Overlay handles interactions
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
                // Colors
                let red = Color::from_rgb(1.0, 0.0, 0.0);
                let blue = Color::from_rgb(0.2, 0.6, 1.0);
                let green = Color::from_rgb(0.2, 1.0, 0.4);
                let yellow = Color::from_rgb(1.0, 1.0, 0.0);
                let purple = Color::from_rgb(0.8, 0.2, 1.0);
                let path_col = Color::from_rgb(0.5, 0.5, 0.5);

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
                                        Stroke::default().with_color(path_col).with_width(1.5),
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
                                        Stroke::default().with_color(path_col).with_width(1.5),
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
                            Stroke::default().with_color(red).with_width(1.5),
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
                            Stroke::default().with_color(blue).with_width(1.5),
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
                            Stroke::default().with_color(green).with_width(2.0),
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
                            Stroke::default().with_color(yellow).with_width(2.5),
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
                            Stroke::default().with_color(purple).with_width(3.0),
                        );
                    }
                }

                if let Some(hover) = &state.hovered_info {
                    let mut highlight_bbox = None;
                    if hover.title.starts_with("Element") {
                        if let Some(id_str) = hover.title.split('#').last() {
                            if let Ok(id) = id_str.parse::<usize>() {
                                if let Some(e) = self
                                    .page
                                    .elements
                                    .as_slice()
                                    .iter()
                                    .find(|e| (e.id as usize) == id)
                                {
                                    highlight_bbox = Some(&e.bbox);
                                }
                            }
                        }
                    } else if hover.title.starts_with("Block") {
                        if let Some(id_str) = hover.title.split('#').last() {
                            if let Ok(id) = id_str.parse::<usize>() {
                                if let Some(b) = self
                                    .page
                                    .blocks
                                    .as_slice()
                                    .iter()
                                    .find(|b| (b.id as usize) == id)
                                {
                                    highlight_bbox = Some(&b.bbox);
                                }
                            }
                        }
                    }

                    if let Some(bbox) = highlight_bbox {
                        let rect = self.to_rect(
                            bbox.x0,
                            bbox.y0,
                            bbox.x1,
                            bbox.y1,
                            final_scale,
                            total_offset_x,
                            total_offset_y,
                        );
                        frame.fill(
                            &Path::rectangle(rect.position(), rect.size()),
                            Color::from_rgba(1.0, 1.0, 1.0, 0.4),
                        );
                        frame.stroke(
                            &Path::rectangle(rect.position(), rect.size()),
                            Stroke::default().with_color(Color::WHITE).with_width(3.0),
                        );
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
        // Ensure x,y are top-left
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

    fn get_hover_detailed(&self, pos: Point, bounds: Rectangle) -> Option<HoverDetailed> {
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

        if self.show_blocks {
            for block in self.page.blocks.as_slice() {
                if px >= block.bbox.x0
                    && px <= block.bbox.x1
                    && py >= block.bbox.y0
                    && py <= block.bbox.y1
                {
                    return Some(HoverDetailed {
                        title: format!("Block #{}", block.id),
                        details: format!(
                            "BBox: [{:.1}, {:.1}, {:.1}, {:.1}]\nPages: {:?}",
                            block.bbox.x0,
                            block.bbox.y0,
                            block.bbox.x1,
                            block.bbox.y1,
                            block.pages_id
                        ),
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
                    return Some(HoverDetailed {
                        title: format!("Element #{}", element.id),
                        details: format!(
                            "BBox: [{:.1}, {:.1}, {:.1}, {:.1}]\nText: {}\nLayout ID: {}",
                            element.bbox.x0,
                            element.bbox.y0,
                            element.bbox.x1,
                            element.bbox.y1,
                            element.text_block.text.as_str(),
                            element.layout_block_id
                        ),
                    });
                }
            }
        }

        None
    }
}
