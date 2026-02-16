use clap::Parser;
use ferrules_core::debug_info::DebugDocument;
use iced::widget::{
    button, canvas, checkbox, column, container, image, row, scrollable, slider, stack, text,
};
use iced::{event, window, Alignment, Color, Element, Event, Length, Task, Theme};
use memmap2::Mmap;
use rkyv::archived_root;
use std::path::PathBuf;

mod painter;
use painter::PagePainter;

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

    // Layer visibility
    show_native: bool,
    show_layout: bool,
    show_ocr: bool,
    show_elements: bool,
    show_blocks: bool,
    show_paths: bool,
}

#[derive(Debug, Clone)]
enum Message {
    PageChanged(u32),
    ToggleLayer(Layer),
    OpenFile,
    FileSelected(Option<PathBuf>),
    FileDropped(PathBuf),
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
        let debug = Self {
            mmap: None,
            current_page_idx: 0,
            show_native: true,
            show_layout: true,
            show_ocr: true,
            show_elements: true,
            show_blocks: true,
            show_paths: true,
        };

        let task = if let Some(path) = file_path {
            Task::done(Message::FileSelected(Some(path)))
        } else {
            Task::none()
        };

        (debug, task)
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::PageChanged(idx) => {
                self.current_page_idx = idx as usize;
                Task::none()
            }
            Message::ToggleLayer(layer) => {
                match layer {
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
                    }
                }
                Task::none()
            }
            Message::FileSelected(None) => Task::none(),
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
            let total_pages = archived.pages.len() as u32;
            let safe_page_idx = (self.current_page_idx as u32).min(total_pages.saturating_sub(1));
            let current_page = &archived.pages[safe_page_idx as usize];

            let sidebar = column![
                text(archived.name.as_str()).size(20),
                button("Open File").on_press(Message::OpenFile).padding(10),
                text(format!("Page {} / {}", safe_page_idx + 1, total_pages)),
                slider(
                    0..=(total_pages.saturating_sub(1)),
                    safe_page_idx,
                    Message::PageChanged
                ),
                column![
                    text("Layers"),
                    checkbox("Native Lines", self.show_native)
                        .on_toggle(|_| Message::ToggleLayer(Layer::Native)),
                    checkbox("Paths", self.show_paths)
                        .on_toggle(|_| Message::ToggleLayer(Layer::Paths)),
                    checkbox("Layout BBoxes", self.show_layout)
                        .on_toggle(|_| Message::ToggleLayer(Layer::Layout)),
                    checkbox("OCR Lines", self.show_ocr)
                        .on_toggle(|_| Message::ToggleLayer(Layer::OCR)),
                    checkbox("Elements", self.show_elements)
                        .on_toggle(|_| Message::ToggleLayer(Layer::Elements)),
                    checkbox("Blocks", self.show_blocks)
                        .on_toggle(|_| Message::ToggleLayer(Layer::Blocks)),
                ]
                .spacing(10)
                .padding(10),
                text("Metadata"),
                scrollable(text(format!("Elements: {}", current_page.elements.len())))
            ]
            .width(Length::Fixed(250.0))
            .spacing(20)
            .padding(20);

            // Page Viewer (Center)
            let image_vec: Vec<u8> = (&current_page.image_data[..]).to_vec();
            let page_image = image::Handle::from_bytes(image_vec);

            let painter = PagePainter {
                page: current_page,
                show_native: self.show_native,
                show_layout: self.show_layout,
                show_ocr: self.show_ocr,
                show_elements: self.show_elements,
                show_blocks: self.show_blocks,
                show_paths: self.show_paths,
            };

            let viewer = container(stack![
                image(page_image),
                canvas(painter).width(Length::Fill).height(Length::Fill)
            ])
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .style(|_theme| container::Style {
                background: Some(iced::Background::Color(Color::BLACK)),
                ..Default::default()
            });

            row![sidebar, viewer].into()
        } else {
            container(
                column![
                    text("Ferrules Debugger").size(40),
                    text("Visualize the PDF parsing pipeline").size(20),
                    button("Select .ferr File")
                        .on_press(Message::OpenFile)
                        .padding([15, 30]),
                    text("Or drag and drop a file anywhere"),
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
}
