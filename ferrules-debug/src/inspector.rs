use iced::font::Weight;
use iced::widget::{column, container, row, scrollable, text, Rule, Space};
use iced::{Color, Element, Font, Length, Theme};

#[derive(Debug, Clone, PartialEq)]
pub enum InspectorItem {
    Block {
        id: usize,
        kind: String,
        bbox: [f32; 4], // x0, y0, x1, y1
        pages: Vec<usize>,
    },
    Element {
        id: usize,
        kind: String,
        bbox: [f32; 4],
        layout_ref: i32,
        text: String,
    },
    Layout {
        id: i32,
        label: String,
        proba: f32,
        bbox: [f32; 4],
    },
    None,
}

impl InspectorItem {
    pub fn title(&self) -> String {
        match self {
            InspectorItem::Block { id, .. } => format!("Block #{}", id),
            InspectorItem::Element { id, .. } => format!("Element #{}", id),
            InspectorItem::Layout { id, .. } => format!("Layout #{}", id),
            InspectorItem::None => "No Selection".to_string(),
        }
    }

    pub fn bbox(&self) -> Option<[f32; 4]> {
        match self {
            InspectorItem::Block { bbox, .. } => Some(*bbox),
            InspectorItem::Element { bbox, .. } => Some(*bbox),
            InspectorItem::Layout { bbox, .. } => Some(*bbox),
            InspectorItem::None => None,
        }
    }

    pub fn kind_label(&self) -> String {
        match self {
            InspectorItem::Block { kind, .. } => kind.clone(),
            InspectorItem::Element { kind, .. } => kind.clone(),
            InspectorItem::Layout { label, .. } => label.clone(),
            InspectorItem::None => "".to_string(),
        }
    }
}

pub fn view_inspector<'a, Message>(item: &'a InspectorItem) -> Element<'a, Message>
where
    Message: 'a + Clone + std::fmt::Debug,
{
    let content: Element<'a, Message> = match item {
        InspectorItem::Block {
            id,
            kind,
            bbox,
            pages,
        } => column![
            header("BLOCK", *id, kind.clone()),
            field("Type", kind.clone()),
            bbox_field(bbox),
            field("Pages", format!("{:?}", pages)),
        ]
        .into(),
        InspectorItem::Element {
            id,
            kind,
            bbox,
            layout_ref,
            text,
        } => {
            column![
                header("ELEMENT", *id, kind.clone()),
                field("Type", kind.clone()),
                field("Layout Ref", layout_ref.to_string()),
                bbox_field(bbox),
                Space::with_height(10),
                section_header("Content"),
                container(
                    iced::widget::text(text.clone())
                        .size(13)
                        .line_height(iced::widget::text::LineHeight::Relative(1.4))
                )
                .padding(8) // Apply padding to container widget
                .style(|theme: &Theme| {
                    let palette = theme.extended_palette();
                    container::Style {
                        background: Some(palette.background.weak.color.into()),
                        border: iced::Border {
                            color: palette.background.strong.color,
                            width: 1.0,
                            radius: 4.0.into(),
                        },
                        ..Default::default()
                    }
                })
            ]
            .into()
        }
        InspectorItem::Layout {
            id,
            label,
            proba,
            bbox,
        } => column![
            header("LAYOUT", *id as usize, label.clone()),
            field("Label", label.clone()),
            field("Confidence", format!("{:.2}%", proba * 100.0)),
            bbox_field(bbox),
        ]
        .into(),
        InspectorItem::None => column![text("No Selection")
            .size(14)
            .color(Color::from_rgb(0.6, 0.6, 0.6))]
        .align_x(iced::alignment::Horizontal::Center)
        .into(),
    };

    scrollable(container(content).width(Length::Fill).padding(15)).into()
}

fn header<'a, Message>(type_name: &'a str, id: usize, subtitle: String) -> Element<'a, Message>
where
    Message: 'a + Clone + std::fmt::Debug,
{
    column![
        row![
            text(type_name)
                .size(12)
                .font(Font {
                    weight: Weight::Bold,
                    ..Default::default()
                })
                .color(Color::from_rgb(0.4, 0.4, 1.0)),
            text(format!("#{}", id))
                .size(12)
                .color(Color::from_rgb(0.6, 0.6, 0.6)),
        ]
        .spacing(5),
        text(subtitle).size(20).font(Font {
            weight: Weight::Semibold,
            ..Default::default()
        }),
        Rule::horizontal(1),
    ]
    .spacing(5)
    .into()
}

fn field<'a, Message>(key: &'a str, value: String) -> Element<'a, Message>
where
    Message: 'a + Clone + std::fmt::Debug,
{
    column![
        text(key).size(11).color(Color::from_rgb(0.5, 0.5, 0.5)),
        text(value).size(14),
    ]
    .into()
}

fn bbox_field<'a, Message>(bbox: &[f32; 4]) -> Element<'a, Message>
where
    Message: 'a + Clone + std::fmt::Debug,
{
    let [x0, y0, x1, y1] = bbox;
    let w = x1 - x0;
    let h = y1 - y0;

    column![
        text("Geometry")
            .size(11)
            .color(Color::from_rgb(0.5, 0.5, 0.5)),
        row![kv_small("X", *x0), kv_small("Y", *y0),].spacing(15),
        row![kv_small("W", w), kv_small("H", h),].spacing(15),
    ]
    .spacing(5)
    .into()
}

fn kv_small<'a, Message>(k: &'a str, v: f32) -> Element<'a, Message>
where
    Message: 'a + Clone + std::fmt::Debug,
{
    row![
        text(k).size(11).color(Color::from_rgb(0.5, 0.5, 0.5)),
        text(format!("{:.1}", v)).size(12).font(Font::MONOSPACE),
    ]
    .spacing(5)
    .into()
}

fn section_header<'a, Message>(title: &'a str) -> Element<'a, Message>
where
    Message: 'a + Clone + std::fmt::Debug,
{
    text(title)
        .size(12)
        .font(Font {
            weight: Weight::Bold,
            ..Default::default()
        })
        .color(Color::from_rgb(0.7, 0.7, 0.7))
        .into()
}
