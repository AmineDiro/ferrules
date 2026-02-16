use clap::Parser;
use ferrules_core::debug_info::DebugDocument;
use iced::font::Weight;
use iced::widget::{
    button, canvas, checkbox, column, container, horizontal_space, image, row, slider, text, Space,
    Tooltip,
};
use iced::{event, window, Alignment, Color, Element, Event, Font, Length, Task, Theme, Vector};
use memmap2::Mmap;
use rkyv::archived_root;
use std::path::PathBuf;

mod inspector;
mod painter;
use inspector::{view_inspector, InspectorItem, InspectorSection};
use painter::{CanvasMessage, PagePainter, PainterMode};

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
        .theme(|_| Theme::CatppuccinMocha)
        .subscription(FerrulesDebug::subscription)
        .run_with(move || FerrulesDebug::new(args.file))
}

// Catppuccin Mocha Palette
const MOCHA_BASE: Color = Color::from_rgb(0.117, 0.117, 0.180); // #1e1e2e
const MOCHA_SURFACE0: Color = Color::from_rgb(0.192, 0.196, 0.266); // #313244
const MOCHA_SURFACE1: Color = Color::from_rgb(0.270, 0.278, 0.352); // #45475a
const MOCHA_LAVENDER: Color = Color::from_rgb(0.705, 0.745, 0.996); // #b4befe (Accent)
const MOCHA_TEXT: Color = Color::from_rgb(0.803, 0.839, 0.956); // #cdd6f4

struct FerrulesDebug {
    mmap: Option<Mmap>,
    current_page_idx: usize,

    zoom: f32,
    offset: Vector,
    show_native: bool,
    show_layout: bool,
    show_ocr: bool,
    show_elements: bool,
    show_blocks: bool,
    show_paths: bool,

    hovered_info: InspectorItem,
    sidebar_open: bool,

    // Inspector Fold States
    inspector_block_open: bool,
    inspector_element_open: bool,
    inspector_layout_open: bool,
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
    ToggleSidebar,
    ToggleInspectorSection(InspectorSection),
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
                zoom: 1.0,
                offset: Vector::new(0.0, 0.0),
                show_native: true,
                show_layout: true,
                show_ocr: true,
                show_elements: true,
                show_blocks: true,
                show_paths: true,
                hovered_info: InspectorItem::None,
                sidebar_open: true,
                inspector_block_open: true,
                inspector_element_open: true,
                inspector_layout_open: false,
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
                        self.zoom = 1.0;
                        self.offset = Vector::new(0.0, 0.0);
                    }
                }
                Task::none()
            }
            Message::FileSelected(None) => Task::none(),
            Message::CanvasEvent(ev) => match ev {
                CanvasMessage::Hovered(info) => {
                    self.hovered_info = info.unwrap_or(InspectorItem::None);
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
            Message::ToggleSidebar => {
                self.sidebar_open = !self.sidebar_open;
                Task::none()
            }
            Message::ToggleInspectorSection(section) => {
                match section {
                    InspectorSection::Block => {
                        self.inspector_block_open = !self.inspector_block_open
                    }
                    InspectorSection::Element => {
                        self.inspector_element_open = !self.inspector_element_open
                    }
                    InspectorSection::Layout => {
                        self.inspector_layout_open = !self.inspector_layout_open
                    }
                }
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
        let bold_font = Font {
            weight: Weight::Bold,
            ..Default::default()
        };
        let medium_font = Font {
            weight: Weight::Medium,
            ..Default::default()
        };

        let logo_path = "/Users/amine/coding/ferrules/imgs/ferrules-logo.png";

        if let Some(mmap) = &self.mmap {
            let archived = unsafe { archived_root::<DebugDocument>(&mmap[..]) };
            let current_page_idx = self
                .current_page_idx
                .min(archived.pages.len().saturating_sub(1));
            let current_page = &archived.pages[current_page_idx];
            let total_pages = archived.pages.len() as u32;

            let image_vec: Vec<u8> = (&current_page.image_data[..]).to_vec();
            let page_image = image::Handle::from_bytes(image_vec);

            // --- LEFT SIDEBAR ---
            let left_sidebar_content = if self.sidebar_open {
                column![
                    row![
                        button(
                            image(logo_path)
                                .width(Length::Fixed(32.0))
                                .height(Length::Fixed(32.0))
                        )
                        .on_press(Message::ToggleSidebar)
                        .padding(0)
                        .style(button::text),
                        horizontal_space(),
                        button(text("‹").size(16).font(bold_font))
                            .on_press(Message::ToggleSidebar)
                            .padding(5)
                            .style(button::text),
                    ]
                    .align_y(Alignment::Center)
                    .width(Length::Fill),
                    Space::with_height(10),
                    text(archived.name.as_str())
                        .size(13)
                        .color(MOCHA_TEXT)
                        .font(medium_font),
                    Space::with_height(30),
                    button(text("Open .ferr").size(14).font(medium_font))
                        .on_press(Message::OpenFile)
                        .padding(10)
                        .width(Length::Fill),
                    Space::with_height(30),
                    text("LAYERS")
                        .size(11)
                        .color(Color::from_rgb(0.5, 0.5, 0.6))
                        .font(bold_font),
                    column![
                        self.layer_checkbox(
                            "Native Lines",
                            self.show_native,
                            Layer::Native,
                            Color::from_rgb(0.95, 0.54, 0.65)
                        ), // Mocha Red
                        self.layer_checkbox(
                            "Vector Paths",
                            self.show_paths,
                            Layer::Paths,
                            Color::from_rgba(0.9, 0.9, 0.9, 0.5)
                        ),
                        self.layer_checkbox(
                            "Layout Analysis",
                            self.show_layout,
                            Layer::Layout,
                            Color::from_rgb(0.65, 0.89, 0.63)
                        ), // Mocha Green
                        self.layer_checkbox(
                            "OCR Text Lines",
                            self.show_ocr,
                            Layer::OCR,
                            Color::from_rgb(0.53, 0.70, 0.98)
                        ), // Mocha Blue
                        self.layer_checkbox(
                            "Logical Elements",
                            self.show_elements,
                            Layer::Elements,
                            Color::from_rgb(0.98, 0.70, 0.52)
                        ), // Mocha Peach
                        self.layer_checkbox(
                            "Structural Blocks",
                            self.show_blocks,
                            Layer::Blocks,
                            Color::from_rgb(0.79, 0.65, 0.96)
                        ), // Mocha Mauve
                    ]
                    .spacing(14),
                ]
                .spacing(10)
                .padding(20)
            } else {
                column![
                    button(
                        image(logo_path)
                            .width(Length::Fixed(32.0))
                            .height(Length::Fixed(32.0))
                    )
                    .on_press(Message::ToggleSidebar)
                    .padding(0)
                    .style(button::text),
                    Space::with_height(30),
                    Tooltip::new(
                        button(text("📂").size(18))
                            .on_press(Message::OpenFile)
                            .padding(8)
                            .style(button::secondary),
                        "Open File",
                        iced::widget::tooltip::Position::Right,
                    ),
                ]
                .spacing(20)
                .padding(10)
                .align_x(Alignment::Center)
            };

            let left_sidebar = container(left_sidebar_content)
                .width(if self.sidebar_open {
                    Length::Fixed(240.0)
                } else {
                    Length::Fixed(60.0)
                })
                .height(Length::Fill)
                .style(move |_| container::Style {
                    background: Some(MOCHA_SURFACE0.into()),
                    border: iced::Border {
                        width: 0.0,
                        color: Color::TRANSPARENT,
                        ..Default::default()
                    },
                    ..Default::default()
                });

            // --- TOP BAR ---
            let top_bar = container(
                row![
                    horizontal_space(),
                    container(
                        row![
                            text(format!("Page {} / {}", current_page_idx + 1, total_pages))
                                .size(14)
                                .font(bold_font),
                            slider(
                                0..=total_pages.saturating_sub(1),
                                current_page_idx as u32,
                                Message::PageChanged,
                            )
                            .width(Length::Fixed(300.0))
                            .step(1u32),
                        ]
                        .spacing(15)
                        .align_y(Alignment::Center)
                    )
                    .padding([0, 20])
                    .style(move |_| container::Style {
                        background: Some(MOCHA_SURFACE0.into()),
                        border: iced::Border {
                            radius: 20.0.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }),
                    horizontal_space(),
                    row![
                        text(format!("{:3.0}%", self.zoom * 100.0))
                            .size(13)
                            .font(bold_font)
                            .color(MOCHA_TEXT),
                        slider(0.1..=5.0, self.zoom, Message::ZoomSliderChanged)
                            .step(0.1)
                            .width(Length::Fixed(120.0)),
                        button(text("Reset").size(12).font(bold_font))
                            .on_press(Message::ResetView)
                            .padding([4, 12])
                            .style(button::secondary),
                    ]
                    .spacing(15)
                    .align_y(Alignment::Center)
                ]
                .width(Length::Fill)
                .align_y(Alignment::Center)
                .padding(12),
            )
            .style(move |_| container::Style {
                background: Some(MOCHA_SURFACE1.into()),
                ..Default::default()
            });

            // --- RIGHT SIDEBAR (Inspector) ---
            let right_sidebar = container(
                column![
                    text("INSPECTOR")
                        .size(11)
                        .color(Color::from_rgb(0.5, 0.5, 0.6))
                        .font(bold_font),
                    Space::with_height(10),
                    view_inspector(
                        &self.hovered_info,
                        self.inspector_block_open,
                        self.inspector_element_open,
                        self.inspector_layout_open,
                        Message::ToggleInspectorSection(InspectorSection::Block),
                        Message::ToggleInspectorSection(InspectorSection::Element),
                        Message::ToggleInspectorSection(InspectorSection::Layout),
                    )
                ]
                .padding(10),
            )
            .width(Length::Fixed(320.0))
            .height(Length::Fill)
            .style(move |_| container::Style {
                background: Some(MOCHA_SURFACE0.into()),
                ..Default::default()
            });

            // --- MAIN CANVAS ---
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

            let canvas_view = container(iced::widget::stack(vec![
                Element::from(
                    canvas(image_painter)
                        .width(Length::Fill)
                        .height(Length::Fill),
                )
                .map(|_| Message::ResetView),
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
                background: Some(MOCHA_BASE.into()),
                ..Default::default()
            });

            // Assemble Layout
            row![
                left_sidebar,
                column![top_bar, canvas_view].width(Length::Fill),
                right_sidebar
            ]
            .into()
        } else {
            container(
                column![
                    image(logo_path)
                        .width(Length::Fixed(120.0))
                        .height(Length::Fixed(120.0)),
                    text("DOCUMENT ANALYSIS TOOLKIT")
                        .size(18)
                        .color(MOCHA_LAVENDER)
                        .font(bold_font),
                    Space::with_height(40),
                    button(text("Select .ferr File").size(18).font(bold_font))
                        .on_press(Message::OpenFile)
                        .padding([15, 60])
                        .style(button::primary),
                    text("Or drop archive here")
                        .size(14)
                        .color(MOCHA_TEXT)
                        .font(medium_font),
                ]
                .align_x(Alignment::Center)
                .spacing(20),
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
        let bold_font = Font {
            weight: Weight::Bold,
            ..Default::default()
        };
        row![
            container(Space::with_width(3))
                .width(Length::Fixed(3.0))
                .height(Length::Fixed(12.0))
                .style(move |_| container::Style {
                    background: Some(color.into()),
                    border: iced::Border {
                        radius: 2.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
            checkbox(label, is_checked)
                .on_toggle(move |_| Message::ToggleLayer(layer.clone()))
                .size(14)
                .text_size(13)
                .font(bold_font),
        ]
        .spacing(12)
        .align_y(Alignment::Center)
        .into()
    }
}
