use clap::Parser;
use ferrules_core::debug_info::DebugDocument;
use iced::widget::{
    Space, Tooltip, button, canvas, checkbox, column, container, horizontal_space, image, row,
    slider, text,
};
use iced::{Alignment, Color, Element, Event, Length, Task, Theme, Vector, event, window};
use memmap2::Mmap;
use rkyv::archived_root;
use std::path::PathBuf;

mod inspector;
mod painter;
pub mod theme;
pub mod widgets;
use inspector::{InspectorItem, InspectorSection, view_inspector};
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
// Catppuccin Mocha Palette (Moved to theme.rs)

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
    show_tables: bool,

    hovered_info: InspectorItem,
    selected_info: Option<InspectorItem>,
    sidebar_open: bool,

    // Inspector Fold States
    inspector_block_open: bool,
    inspector_element_open: bool,
    inspector_layout_open: bool,
    inspector_cell_open: bool,
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
    Tables,
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
                show_tables: true,
                hovered_info: InspectorItem::None,
                selected_info: None,
                sidebar_open: true,
                inspector_block_open: true,
                inspector_element_open: true,
                inspector_layout_open: false,
                inspector_cell_open: true,
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
                    Layer::Tables => self.show_tables = !self.show_tables,
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
                CanvasMessage::Clicked(info) => {
                    if let Some(clicked_item) = info {
                        // If we clicked the same thing that is already selected, deselect it.
                        // Otherwise, select the new thing.
                        if let Some(current) = &self.selected_info {
                            // Simple equality check might depend on PartialEq implementation of InspectorItem
                            // which we derived.
                            if current == &clicked_item {
                                self.selected_info = None;
                            } else {
                                self.selected_info = Some(clicked_item);
                            }
                        } else {
                            self.selected_info = Some(clicked_item);
                        }
                    } else {
                        // Clicked on nothing -> deselect
                        self.selected_info = None;
                    }
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
                    InspectorSection::Cell => self.inspector_cell_open = !self.inspector_cell_open,
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
                        // Line 268 fix: use theme::FONT_BOLD not bold_font
                        button(text("‹").size(16).font(theme::FONT_BOLD))
                            .on_press(Message::ToggleSidebar)
                            .padding(5)
                            .style(button::text),
                    ]
                    .align_y(Alignment::Center)
                    .width(Length::Fill),
                    widgets::v_space(10.0),
                    text(archived.name.as_str())
                        .size(theme::TEXT_SIZE_MD)
                        .color(theme::TEXT)
                        .font(theme::FONT_MEDIUM),
                    widgets::v_space(30.0),
                    button(
                        text("Open .ferr")
                            .size(theme::TEXT_SIZE_LG)
                            .font(theme::FONT_MEDIUM)
                    )
                    .on_press(Message::OpenFile)
                    .padding(theme::PADDING_MD)
                    .width(Length::Fill),
                    widgets::v_space(30.0),
                    widgets::section_header("LAYERS"),
                    column![
                        self.layer_checkbox(
                            "Native Lines",
                            self.show_native,
                            Layer::Native,
                            theme::RED
                        ),
                        self.layer_checkbox(
                            "Vector Paths",
                            self.show_paths,
                            Layer::Paths,
                            theme::OVERLAY0_FADED
                        ),
                        self.layer_checkbox(
                            "Layout Analysis",
                            self.show_layout,
                            Layer::Layout,
                            theme::GREEN
                        ),
                        self.layer_checkbox(
                            "OCR Text Lines",
                            self.show_ocr,
                            Layer::OCR,
                            theme::BLUE
                        ),
                        self.layer_checkbox(
                            "Logical Elements",
                            self.show_elements,
                            Layer::Elements,
                            theme::PEACH
                        ),
                        self.layer_checkbox(
                            "Tables & Cells",
                            self.show_tables,
                            Layer::Tables,
                            theme::ORANGE
                        ),
                        self.layer_checkbox(
                            "Structural Blocks",
                            self.show_blocks,
                            Layer::Blocks,
                            theme::MAUVE
                        ),
                    ]
                    .spacing(theme::SPACING_LG),
                ]
                .spacing(theme::SPACING_MD)
                .padding(theme::PADDING_LG)
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
                .spacing(theme::SPACING_XL)
                .padding(theme::PADDING_MD)
                .align_x(Alignment::Center)
            };

            let left_sidebar = widgets::panel_container(left_sidebar_content)
                .width(if self.sidebar_open {
                    Length::Fixed(theme::SIDEBAR_WIDTH_OPEN)
                } else {
                    Length::Fixed(theme::SIDEBAR_WIDTH_CLOSED)
                })
                .height(Length::Fill);

            // --- TOP BAR ---
            let top_bar = container(
                row![
                    horizontal_space(),
                    container(
                        row![
                            text(format!("Page {} / {}", current_page_idx + 1, total_pages))
                                .size(theme::TEXT_SIZE_LG)
                                .font(theme::FONT_BOLD),
                            slider(
                                0..=total_pages.saturating_sub(1),
                                current_page_idx as u32,
                                Message::PageChanged,
                            )
                            .width(Length::Fixed(300.0))
                            .step(1u32),
                        ]
                        .spacing(theme::SPACING_LG)
                        .align_y(Alignment::Center)
                    )
                    .padding([0.0, theme::PADDING_LG])
                    .style(move |_| container::Style {
                        background: Some(theme::SURFACE0.into()),
                        border: iced::Border {
                            radius: theme::BORDER_RADIUS_LG.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }),
                    horizontal_space(),
                    row![
                        text(format!("{:3.0}%", self.zoom * 100.0))
                            .size(theme::TEXT_SIZE_MD)
                            .font(theme::FONT_BOLD)
                            .color(theme::TEXT),
                        slider(0.1..=5.0, self.zoom, Message::ZoomSliderChanged)
                            .step(0.1)
                            .width(Length::Fixed(120.0)),
                        button(text("Reset").size(12).font(theme::FONT_BOLD))
                            .on_press(Message::ResetView)
                            .padding([4, 12])
                            .style(button::secondary),
                    ]
                    .spacing(theme::SPACING_LG)
                    .align_y(Alignment::Center)
                ]
                .width(Length::Fill)
                .align_y(Alignment::Center)
                .padding(12),
            )
            .style(move |_| container::Style {
                background: Some(theme::SURFACE1.into()),
                ..Default::default()
            });

            // --- RIGHT SIDEBAR (Inspector) ---
            let right_sidebar = widgets::panel_container(
                column![
                    widgets::section_header("INSPECTOR"),
                    widgets::v_space(10.0),
                    view_inspector(
                        self.selected_info.as_ref().unwrap_or(&self.hovered_info),
                        self.inspector_block_open,
                        self.inspector_element_open,
                        self.inspector_layout_open,
                        self.inspector_cell_open,
                        Message::ToggleInspectorSection(InspectorSection::Block),
                        Message::ToggleInspectorSection(InspectorSection::Element),
                        Message::ToggleInspectorSection(InspectorSection::Layout),
                        Message::ToggleInspectorSection(InspectorSection::Cell),
                    )
                ]
                .padding(theme::PADDING_MD),
            )
            .width(Length::Fixed(theme::INSPECTOR_WIDTH))
            .height(Length::Fill);

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
                show_tables: self.show_tables,
                selected_item: self.selected_info.clone(),
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
                show_tables: self.show_tables,
                selected_item: self.selected_info.clone(),
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
                background: Some(theme::BASE.into()),
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
                        .size(theme::TEXT_SIZE_XL)
                        .color(theme::LAVENDER)
                        .font(theme::FONT_BOLD),
                    widgets::v_space(40.0),
                    button(
                        text("Select .ferr File")
                            .size(theme::TEXT_SIZE_XL)
                            .font(theme::FONT_BOLD)
                    )
                    .on_press(Message::OpenFile)
                    .padding([15, 60])
                    .style(button::primary),
                    text("Or drop archive here")
                        .size(theme::TEXT_SIZE_LG)
                        .color(theme::TEXT)
                        .font(theme::FONT_MEDIUM),
                ]
                .align_x(Alignment::Center)
                .spacing(theme::SPACING_XL),
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
                .text_size(theme::TEXT_SIZE_MD)
                .font(theme::FONT_BOLD),
        ]
        .spacing(12)
        .align_y(Alignment::Center)
        .into()
    }
}
