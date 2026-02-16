use clap::Parser;
use ferrules_core::debug_info::DebugDocument;
use iced::widget::{
    button, canvas, checkbox, column, container, image, row, scrollable, slider, text,
};
use iced::{event, window, Alignment, Color, Element, Event, Length, Task, Theme, Vector};
use memmap2::Mmap;
use rkyv::archived_root;
use std::path::PathBuf;

mod painter;
use painter::{CanvasMessage, HoverDetailed, PagePainter, PainterMode};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .ferr debug file
    #[arg(short, long)]
    file: Option<PathBuf>,
}

pub fn main() -> iced::Result {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    iced::application("Ferrules Debug", FerrulesDebug::update, FerrulesDebug::view)
        .theme(|_| Theme::Dark)
        .subscription(FerrulesDebug::subscription)
        .run_with(move || FerrulesDebug::new(args.file))
}

struct FerrulesDebug {
    mmap: Option<Mmap>,
    current_page_idx: usize,
    cached_page_image_handle: Option<image::Handle>,
    last_page_idx_cached: Option<usize>,

    zoom: f32,
    offset: Vector,
    show_native: bool,
    show_layout: bool,
    show_ocr: bool,
    show_elements: bool,
    show_blocks: bool,
    show_paths: bool,
    hovered_info: Option<HoverDetailed>,
}

#[derive(Debug, Clone)]
enum Message {
    PageChanged(u32),
    ToggleLayer(Layer),
    OpenFile,
    FileSelected(Option<PathBuf>),
    FileDropped(PathBuf),
    CanvasEvent(CanvasMessage),
    ZoomSliderChanged(f32),
    ResetView,
}

#[derive(Debug, Clone)]
enum Layer {
    Native,
    Layout,
    OCR,
    Elements,
    Blocks,
    Paths,
}

impl FerrulesDebug {
    fn new(file_path: Option<PathBuf>) -> (Self, Task<Message>) {
        (
            Self {
                mmap: None,
                current_page_idx: 0,
                cached_page_image_handle: None,
                last_page_idx_cached: None,
                zoom: 1.0,
                offset: Vector::new(0.0, 0.0),
                show_native: true,
                show_layout: true,
                show_ocr: true,
                show_elements: true,
                show_blocks: true,
                show_paths: true,
                hovered_info: None,
            },
            if let Some(path) = file_path {
                Task::done(Message::FileSelected(Some(path)))
            } else {
                Task::none()
            },
        )
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::PageChanged(idx) => {
                self.current_page_idx = idx as usize;
                self.zoom = 1.0;
                self.offset = Vector::new(0.0, 0.0);
                Task::none()
            }
            Message::ToggleLayer(l) => {
                match l {
                    Layer::Native => self.show_native = !self.show_native,
                    Layer::Layout => self.show_layout = !self.show_layout,
                    Layer::OCR => self.show_ocr = !self.show_ocr,
                    Layer::Elements => self.show_elements = !self.show_elements,
                    Layer::Blocks => self.show_blocks = !self.show_blocks,
                    Layer::Paths => self.show_paths = !self.show_paths,
                }
                Task::none()
            }
            Message::OpenFile => Task::perform(
                async {
                    rfd::AsyncFileDialog::new()
                        .add_filter("Ferrules", &["ferr"])
                        .pick_file()
                        .await
                },
                |file| Message::FileSelected(file.map(|f| f.path().to_path_buf())),
            ),
            Message::FileSelected(Some(path)) | Message::FileDropped(path) => {
                if let Ok(file) = std::fs::File::open(&path) {
                    if let Ok(m) = unsafe { Mmap::map(&file) } {
                        self.mmap = Some(m);
                        self.current_page_idx = 0;
                        self.cached_page_image_handle = None;
                        self.last_page_idx_cached = None;
                        self.zoom = 1.0;
                        self.offset = Vector::new(0.0, 0.0);
                    }
                }
                Task::none()
            }
            Message::FileSelected(None) => Task::none(),
            Message::CanvasEvent(ev) => match ev {
                CanvasMessage::Hovered(info) => {
                    self.hovered_info = info;
                    Task::none()
                }
                CanvasMessage::ZoomChanged(factor, pos) => {
                    let prev_zoom = self.zoom;
                    self.zoom = (self.zoom + factor).max(0.1).min(10.0);
                    let ratio = self.zoom / prev_zoom;
                    self.offset =
                        self.offset + (self.offset - Vector::new(pos.x, pos.y)) * (ratio - 1.0);
                    Task::none()
                }
                CanvasMessage::OffsetChanged(delta) => {
                    self.offset = self.offset + delta;
                    Task::none()
                }
            },
            Message::ZoomSliderChanged(val) => {
                self.zoom = val;
                Task::none()
            }
            Message::ResetView => {
                self.zoom = 1.0;
                self.offset = Vector::new(0.0, 0.0);
                Task::none()
            }
        }
    }

    fn subscription(&self) -> iced::Subscription<Message> {
        event::listen_with(|event, _status, _window| match event {
            Event::Window(window::Event::FileDropped(path)) => Some(Message::FileDropped(path)),
            _ => None,
        })
    }

    fn view(&self) -> Element<'_, Message> {
        if let Some(mmap) = &self.mmap {
            let archived = unsafe { archived_root::<DebugDocument>(&mmap[..]) };
            let current_page_idx = self
                .current_page_idx
                .min(archived.pages.len().saturating_sub(1));
            let current_page = &archived.pages[current_page_idx];
            let total_pages = archived.pages.len() as u32;

            let image_vec: Vec<u8> = (&current_page.image_data[..]).to_vec();
            let page_image = image::Handle::from_bytes(image_vec);

            let sidebar = container(
                column![
                    text(archived.name.as_str()).size(22),
                    button(text("Open File").size(14))
                        .on_press(Message::OpenFile)
                        .padding([8, 16]),
                    column![
                        text(format!("Page {} / {}", current_page_idx + 1, total_pages)).size(14),
                        slider(
                            0..=(total_pages.saturating_sub(1)),
                            current_page_idx as u32,
                            Message::PageChanged
                        ),
                    ]
                    .spacing(5),
                    column![
                        text("LAYERS")
                            .size(12)
                            .color(Color::from_rgb(0.5, 0.5, 0.5)),
                        self.layer_checkbox(
                            "Native Lines",
                            self.show_native,
                            Layer::Native,
                            Color::from_rgb(0.9, 0.2, 0.2)
                        ),
                        self.layer_checkbox(
                            "Vector Paths",
                            self.show_paths,
                            Layer::Paths,
                            Color::from_rgb(0.5, 0.5, 0.5)
                        ),
                        self.layer_checkbox(
                            "Layout Analysis",
                            self.show_layout,
                            Layer::Layout,
                            Color::from_rgb(0.2, 0.8, 0.4)
                        ),
                        self.layer_checkbox(
                            "OCR Results",
                            self.show_ocr,
                            Layer::OCR,
                            Color::from_rgb(0.2, 0.4, 0.9)
                        ),
                        self.layer_checkbox(
                            "Elements",
                            self.show_elements,
                            Layer::Elements,
                            Color::from_rgb(0.9, 0.9, 0.2)
                        ),
                        self.layer_checkbox(
                            "Blocks",
                            self.show_blocks,
                            Layer::Blocks,
                            Color::from_rgb(0.7, 0.2, 0.9)
                        ),
                    ]
                    .spacing(10),
                    column![
                        text("INSPECTOR")
                            .size(12)
                            .color(Color::from_rgb(0.5, 0.5, 0.5)),
                        container(scrollable(match &self.hovered_info {
                            Some(info) => Element::from(
                                column![
                                    text(&info.title)
                                        .size(15)
                                        .color(Color::from_rgb(0.2, 0.8, 0.4)),
                                    text(&info.details).size(13)
                                ]
                                .spacing(5)
                            ),
                            None => Element::from(
                                text("Hover an element to explore")
                                    .size(13)
                                    .color(Color::from_rgb(0.4, 0.4, 0.4))
                            ),
                        }))
                        .padding(10)
                        .width(Length::Fill)
                        .height(Length::Fixed(250.0))
                        .style(|theme: &Theme| {
                            let palette = theme.extended_palette();
                            container::Style {
                                background: Some(palette.background.weak.color.into()),
                                border: iced::Border {
                                    radius: 4.0.into(),
                                    width: 1.0,
                                    color: Color::from_rgba(1.0, 1.0, 1.0, 0.1),
                                },
                                ..Default::default()
                            }
                        })
                    ]
                    .spacing(8),
                    column![
                        text("VIEW CONTROL")
                            .size(12)
                            .color(Color::from_rgb(0.5, 0.5, 0.5)),
                        row![
                            text(format!("{:3.0}%", self.zoom * 100.0))
                                .size(14)
                                .width(Length::Fixed(55.0)),
                            slider(0.1..=5.0, self.zoom, Message::ZoomSliderChanged).step(0.1),
                            button(text("Reset").size(12))
                                .on_press(Message::ResetView)
                                .padding(5),
                        ]
                        .spacing(10)
                        .align_y(Alignment::Center)
                    ]
                    .spacing(8),
                ]
                .width(Length::Fixed(280.0))
                .spacing(15)
                .padding(20),
            )
            .height(Length::Fill)
            .style(|_theme: &Theme| container::Style {
                background: Some(Color::from_rgba(0.1, 0.1, 0.1, 0.9).into()),
                ..Default::default()
            });

            let image_painter = PagePainter {
                page: current_page,
                image_handle: page_image.clone(),
                zoom: self.zoom,
                offset: self.offset,
                mode: PainterMode::Image,
                show_native: self.show_native,
                show_layout: self.show_layout,
                show_ocr: self.show_ocr,
                show_elements: self.show_elements,
                show_blocks: self.show_blocks,
                show_paths: self.show_paths,
            };

            let overlay_painter = PagePainter {
                page: current_page,
                image_handle: page_image.clone(),
                zoom: self.zoom,
                offset: self.offset,
                mode: PainterMode::Overlay,
                show_native: self.show_native,
                show_layout: self.show_layout,
                show_ocr: self.show_ocr,
                show_elements: self.show_elements,
                show_blocks: self.show_blocks,
                show_paths: self.show_paths,
            };

            let viewer = container(iced::widget::stack(vec![
                Element::from(
                    canvas(image_painter)
                        .width(Length::Fill)
                        .height(Length::Fill),
                )
                .map(|_| Message::ResetView), // Dummy mapping for compilation, will not emit events anyway
                Element::from(
                    canvas(overlay_painter)
                        .width(Length::Fill)
                        .height(Length::Fill),
                )
                .map(Message::CanvasEvent),
            ]))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_| container::Style {
                background: Some(Color::BLACK.into()),
                ..Default::default()
            });

            row![sidebar, viewer].into()
        } else {
            container(
                column![
                    text("FERRULES")
                        .size(60)
                        .color(Color::from_rgb(0.2, 0.8, 0.4)),
                    text("Structure Extraction Debugger")
                        .size(20)
                        .color(Color::from_rgb(0.6, 0.6, 0.6)),
                    button(text("Select .ferr File").size(18))
                        .on_press(Message::OpenFile)
                        .padding([15, 40]),
                    text("Or drag and drop a file here")
                        .size(14)
                        .color(Color::from_rgb(0.4, 0.4, 0.4)),
                ]
                .align_x(Alignment::Center)
                .spacing(30),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into()
        }
    }

    fn layer_checkbox(
        &self,
        label: &str,
        is_checked: bool,
        layer: Layer,
        color: Color,
    ) -> Element<'_, Message> {
        row![
            container(column![])
                .width(Length::Fixed(4.0))
                .height(Length::Fixed(16.0))
                .style(move |_| container::Style {
                    background: Some(color.into()),
                    ..Default::default()
                }),
            checkbox(label, is_checked)
                .on_toggle(move |_| Message::ToggleLayer(layer.clone()))
                .size(16)
        ]
        .spacing(10)
        .align_y(Alignment::Center)
        .into()
    }
}
