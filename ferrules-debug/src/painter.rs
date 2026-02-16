use ferrules_core::debug_info::ArchivedDebugPage;
use ferrules_core::entities::ArchivedSegment;
use iced::widget::canvas::{Frame, Geometry, Path, Program, Stroke};
use iced::{Color, Point, Rectangle, Renderer, Theme};

pub struct PagePainter<'a> {
    pub page: &'a ArchivedDebugPage,
    pub show_native: bool,
    pub show_layout: bool,
    pub show_ocr: bool,
    pub show_elements: bool,
    pub show_blocks: bool,
    pub show_paths: bool,
}

impl<'a, Message> Program<Message> for PagePainter<'a> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());

        // Scale factor to fit the page into the bounds
        let scale_x = bounds.width / self.page.width;
        let scale_y = bounds.height / self.page.height;
        let scale = scale_x.min(scale_y);

        let offset_x = (bounds.width - self.page.width * scale) / 2.0;
        let offset_y = (bounds.height - self.page.height * scale) / 2.0;

        let to_screen =
            |x: f32, y: f32| -> Point { Point::new(x * scale + offset_x, y * scale + offset_y) };

        // Draw Paths first (Background)
        if self.show_paths {
            for path in self.page.paths.as_slice() {
                for segment in path.segments.as_slice() {
                    match segment {
                        ArchivedSegment::Line { start, end } => {
                            let p1 = to_screen(start.0, start.1);
                            let p2 = to_screen(end.0, end.1);
                            frame.stroke(
                                &Path::line(p1, p2),
                                Stroke::default()
                                    .with_color(Color::from_rgb(0.5, 0.5, 0.5))
                                    .with_width(0.5),
                            );
                        }
                        ArchivedSegment::Rect { bbox } => {
                            let rect = Rectangle {
                                x: bbox.x0 * scale + offset_x,
                                y: bbox.y0 * scale + offset_y,
                                width: (bbox.x1 - bbox.x0) * scale,
                                height: (bbox.y1 - bbox.y0) * scale,
                            };
                            frame.stroke(
                                &Path::rectangle(rect.position(), rect.size()),
                                Stroke::default()
                                    .with_color(Color::from_rgb(0.5, 0.5, 0.5))
                                    .with_width(0.5),
                            );
                        }
                    }
                }
            }
        }

        // Draw Native Lines
        if self.show_native {
            for line in self.page.native_lines.as_slice() {
                let rect = Rectangle {
                    x: line.bbox.x0 * scale + offset_x,
                    y: line.bbox.y0 * scale + offset_y,
                    width: (line.bbox.x1 - line.bbox.x0) * scale,
                    height: (line.bbox.y1 - line.bbox.y0) * scale,
                };
                frame.stroke(
                    &Path::rectangle(rect.position(), rect.size()),
                    Stroke::default()
                        .with_color(Color::from_rgb(1.0, 0.0, 0.0))
                        .with_width(1.0),
                );
            }
        }

        // Draw OCR Lines
        if self.show_ocr {
            for line in self.page.ocr_lines.as_slice() {
                let rect = Rectangle {
                    x: line.bbox.x0 * scale + offset_x,
                    y: line.bbox.y0 * scale + offset_y,
                    width: (line.bbox.x1 - line.bbox.x0) * scale,
                    height: (line.bbox.y1 - line.bbox.y0) * scale,
                };
                frame.stroke(
                    &Path::rectangle(rect.position(), rect.size()),
                    Stroke::default()
                        .with_color(Color::from_rgb(0.0, 0.0, 1.0)) // Blue for OCR
                        .with_width(1.0),
                );
            }
        }

        // Draw Layout BBoxes
        if self.show_layout {
            for bbox in self.page.layout_bboxes.as_slice() {
                let rect = Rectangle {
                    x: bbox.bbox.x0 * scale + offset_x,
                    y: bbox.bbox.y0 * scale + offset_y,
                    width: (bbox.bbox.x1 - bbox.bbox.x0) * scale,
                    height: (bbox.bbox.y1 - bbox.bbox.y0) * scale,
                };
                frame.stroke(
                    &Path::rectangle(rect.position(), rect.size()),
                    Stroke::default()
                        .with_color(Color::from_rgb(0.0, 1.0, 0.0))
                        .with_width(1.5),
                );
            }
        }

        // Draw Elements
        if self.show_elements {
            for element in self.page.elements.as_slice() {
                let rect = Rectangle {
                    x: element.bbox.x0 * scale + offset_x,
                    y: element.bbox.y0 * scale + offset_y,
                    width: (element.bbox.x1 - element.bbox.x0) * scale,
                    height: (element.bbox.y1 - element.bbox.y0) * scale,
                };
                frame.stroke(
                    &Path::rectangle(rect.position(), rect.size()),
                    Stroke::default()
                        .with_color(Color::from_rgb(1.0, 1.0, 0.0))
                        .with_width(2.0),
                );
            }
        }

        // Draw Blocks
        if self.show_blocks {
            for block in self.page.blocks.as_slice() {
                let rect = Rectangle {
                    x: block.bbox.x0 * scale + offset_x,
                    y: block.bbox.y0 * scale + offset_y,
                    width: (block.bbox.x1 - block.bbox.x0) * scale,
                    height: (block.bbox.y1 - block.bbox.y0) * scale,
                };
                frame.stroke(
                    &Path::rectangle(rect.position(), rect.size()),
                    Stroke::default()
                        .with_color(Color::from_rgb(1.0, 0.0, 1.0))
                        .with_width(2.5),
                );
            }
        }

        vec![frame.into_geometry()]
    }
}
