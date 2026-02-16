use iced::font::Weight;
use iced::widget::{button, column, container, row, scrollable, text, Space};
use iced::{Alignment, Color, Element, Font, Length};

#[derive(Debug, Clone, PartialEq)]
pub struct InspectorBlock {
    pub id: usize,
    pub kind: String,
    pub bbox: [f32; 4],
    pub pages: Vec<usize>,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InspectorElement {
    pub id: usize,
    pub kind: String,
    pub bbox: [f32; 4],
    pub layout_ref: i32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InspectorLayout {
    pub id: i32,
    pub label: String,
    pub proba: f32,
    pub bbox: [f32; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub enum InspectorItem {
    Selection {
        block: Option<InspectorBlock>,
        element: Option<InspectorElement>,
        layout: Option<InspectorLayout>,
    },
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectorSection {
    Block,
    Element,
    Layout,
}

pub fn view_inspector<'a, Message>(
    item: &'a InspectorItem,
    block_open: bool,
    element_open: bool,
    layout_open: bool,
    on_toggle_block: Message,
    on_toggle_element: Message,
    on_toggle_layout: Message,
) -> Element<'a, Message>
where
    Message: 'a + Clone,
{
    match item {
        InspectorItem::Selection {
            block,
            element,
            layout,
        } => {
            let mut sections = column![].spacing(10);

            if let Some(e) = element {
                sections = sections.push(render_foldable_section(
                    "ELEMENT",
                    &e.kind,
                    element_open,
                    || render_element_details(e),
                    on_toggle_element.clone(),
                ));
            }

            if let Some(b) = block {
                sections = sections.push(render_foldable_section(
                    "BLOCK",
                    &b.kind,
                    block_open,
                    || render_block_details(b),
                    on_toggle_block.clone(),
                ));
            }

            if let Some(l) = layout {
                sections = sections.push(render_foldable_section(
                    "LAYOUT",
                    &l.label,
                    layout_open,
                    || render_layout_details(l),
                    on_toggle_layout.clone(),
                ));
            }

            scrollable(container(sections).width(Length::Fill).padding(10)).into()
        }
        InspectorItem::None => container(
            text("No selection")
                .size(14)
                .color(Color::from_rgb(0.5, 0.5, 0.6)),
        )
        .width(Length::Fill)
        .center_x(Length::Fill)
        .padding(20)
        .into(),
    }
}

fn render_foldable_section<'a, Message>(
    title: &'a str,
    subtitle: &'a str,
    is_open: bool,
    content_fn: impl Fn() -> Element<'a, Message>,
    on_toggle: Message,
) -> Element<'a, Message>
where
    Message: 'a + Clone,
{
    let header = button(
        row![
            text(if is_open { "▼" } else { "▶" })
                .size(10)
                .font(Font::MONOSPACE),
            text(title)
                .size(11)
                .font(Font {
                    weight: Weight::Bold,
                    ..Default::default()
                })
                .color(Color::from_rgb(0.7, 0.7, 0.9)),
            text(subtitle)
                .size(11)
                .color(Color::from_rgb(0.5, 0.5, 0.6)),
        ]
        .spacing(10)
        .align_y(Alignment::Center),
    )
    .on_press(on_toggle)
    .style(button::text)
    .padding(5)
    .width(Length::Fill);

    let mut col = column![header].spacing(5);
    if is_open {
        col = col.push(
            container(content_fn())
                .padding([5, 15])
                .style(|_| container::Style {
                    border: iced::Border {
                        width: 0.5,
                        color: Color::from_rgba(1.0, 1.0, 1.0, 0.1),
                        radius: 4.0.into(),
                    },
                    ..Default::default()
                }),
        );
    }

    col.into()
}

fn render_block_details<'a, Message: 'a>(b: &InspectorBlock) -> Element<'a, Message> {
    let mut col = column![
        field::<Message>("Type", b.kind.clone()),
        bbox_field::<Message>(&b.bbox),
        field::<Message>("Pages", format!("{:?}", b.pages)),
    ]
    .spacing(8);

    if !b.text.is_empty() {
        col = col.push(
            column![
                Space::with_height(5),
                section_header::<Message>("Text Content"),
                render_text_box::<Message>(&b.text)
            ]
            .spacing(5),
        );
    }

    col.into()
}

fn render_element_details<'a, Message: 'a>(e: &InspectorElement) -> Element<'a, Message> {
    let mut col = column![
        field::<Message>("Type", e.kind.clone()),
        field::<Message>("Layout Ref", e.layout_ref.to_string()),
        bbox_field::<Message>(&e.bbox),
    ]
    .spacing(8);

    if !e.text.is_empty() {
        col = col.push(
            column![
                Space::with_height(5),
                section_header::<Message>("Text Content"),
                render_text_box::<Message>(&e.text)
            ]
            .spacing(5),
        );
    }

    col.into()
}

fn render_layout_details<'a, Message: 'a>(l: &InspectorLayout) -> Element<'a, Message> {
    column![
        field::<Message>("Label", l.label.clone()),
        field::<Message>("Confidence", format!("{:.2}%", l.proba * 100.0)),
        bbox_field::<Message>(&l.bbox),
    ]
    .spacing(8)
    .into()
}

fn render_text_box<'a, Message: 'a>(content: &str) -> Element<'a, Message> {
    container(
        text(content.to_string())
            .size(12)
            .line_height(iced::widget::text::LineHeight::Relative(1.4)),
    )
    .padding(8)
    .style(|_| container::Style {
        background: Some(Color::from_rgba(1.0, 1.0, 1.0, 0.03).into()),
        border: iced::Border {
            color: Color::from_rgba(1.0, 1.0, 1.0, 0.1),
            width: 1.0,
            radius: 4.0.into(),
        },
        ..Default::default()
    })
    .into()
}

fn field<'a, Message: 'a>(key: &'a str, value: String) -> Element<'a, Message> {
    row![
        text(format!("{}:", key))
            .size(10)
            .color(Color::from_rgb(0.5, 0.5, 0.6))
            .font(Font {
                weight: Weight::Bold,
                ..Default::default()
            }),
        text(value).size(11),
    ]
    .spacing(10)
    .into()
}

fn bbox_field<'a, Message: 'a>(bbox: &[f32; 4]) -> Element<'a, Message> {
    let [x0, y0, x1, y1] = bbox;
    let w = (x1 - x0).abs();
    let h = (y1 - y0).abs();

    column![
        text("Geometry")
            .size(10)
            .color(Color::from_rgb(0.5, 0.5, 0.6))
            .font(Font {
                weight: Weight::Bold,
                ..Default::default()
            }),
        row![
            text(format!("X:{:.1} Y:{:.1} W:{:.1} H:{:.1}", x0, y0, w, h))
                .size(11)
                .font(Font::MONOSPACE)
        ]
    ]
    .spacing(2)
    .into()
}

fn section_header<'a, Message: 'a>(title: &'a str) -> Element<'a, Message> {
    text(title)
        .size(10)
        .font(Font {
            weight: Weight::Bold,
            ..Default::default()
        })
        .color(Color::from_rgb(0.6, 0.6, 0.7))
        .into()
}
