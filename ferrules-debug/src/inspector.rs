use crate::theme;
use crate::widgets;
use iced::widget::{button, column, container, row, scrollable, text};
use iced::{Alignment, Color, Element, Length};

#[derive(Debug, Clone, PartialEq)]
pub struct InspectorTableDetails {
    pub algorithm: String,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InspectorCell {
    pub row_idx: usize,
    pub col_idx: usize,
    pub row_span: u8,
    pub col_span: u8,
    pub text: String,
    pub bbox: [f32; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct InspectorBlock {
    pub id: usize,
    pub kind: String,
    pub bbox: [f32; 4],
    pub pages: Vec<usize>,
    pub text: String,
    pub table_details: Option<InspectorTableDetails>,
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
        cell: Option<InspectorCell>,
    },
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectorSection {
    Block,
    Element,
    Layout,
    Cell,
}

pub fn view_inspector<'a, Message>(
    item: &'a InspectorItem,
    block_open: bool,
    element_open: bool,
    layout_open: bool,
    cell_open: bool,
    on_toggle_block: Message,
    on_toggle_element: Message,
    on_toggle_layout: Message,
    on_toggle_cell: Message,
) -> Element<'a, Message>
where
    Message: 'a + Clone,
{
    match item {
        InspectorItem::Selection {
            block,
            element,
            layout,
            cell,
        } => {
            let mut sections = column![].spacing(theme::SPACING_MD);

            if let Some(c) = cell {
                sections = sections.push(render_foldable_section(
                    "CELL".to_string(),
                    format!("R{} C{}", c.row_idx, c.col_idx),
                    cell_open,
                    || render_cell_details(c),
                    on_toggle_cell.clone(),
                ));
            }

            if let Some(e) = element {
                sections = sections.push(render_foldable_section(
                    "ELEMENT".to_string(),
                    e.kind.clone(),
                    element_open,
                    || render_element_details(e),
                    on_toggle_element.clone(),
                ));
            }

            if let Some(b) = block {
                sections = sections.push(render_foldable_section(
                    "BLOCK".to_string(),
                    b.kind.clone(),
                    block_open,
                    || render_block_details(b),
                    on_toggle_block.clone(),
                ));
            }

            if let Some(l) = layout {
                sections = sections.push(render_foldable_section(
                    "LAYOUT".to_string(),
                    l.label.clone(),
                    layout_open,
                    || render_layout_details(l),
                    on_toggle_layout.clone(),
                ));
            }

            scrollable(
                container(sections)
                    .width(Length::Fill)
                    .padding(theme::PADDING_MD),
            )
            .into()
        }
        InspectorItem::None => container(
            text("No selection")
                .size(theme::TEXT_SIZE_LG)
                .color(theme::SUBTEXT0),
        )
        .width(Length::Fill)
        .center_x(Length::Fill)
        .padding(theme::PADDING_LG)
        .into(),
    }
}

fn render_foldable_section<'a, Message>(
    title: String,
    subtitle: String,
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
                .size(theme::TEXT_SIZE_SM)
                .font(theme::FONT_MONOSPACE),
            text(title)
                .size(theme::TEXT_SIZE_SM)
                .font(theme::FONT_BOLD)
                .color(theme::LAVENDER),
            text(subtitle)
                .size(theme::TEXT_SIZE_SM)
                .color(theme::SUBTEXT0),
        ]
        .spacing(theme::SPACING_MD)
        .align_y(Alignment::Center),
    )
    .on_press(on_toggle)
    .style(button::text)
    .padding(theme::PADDING_SM)
    .width(Length::Fill);

    let mut col = column![header].spacing(theme::SPACING_SM);
    if is_open {
        col = col.push(
            container(content_fn())
                .padding([theme::PADDING_SM, theme::PADDING_LG])
                .style(|_| container::Style {
                    border: iced::Border {
                        width: 0.5,
                        color: theme::SURFACE1,
                        radius: theme::BORDER_RADIUS_SM.into(),
                    },
                    ..Default::default()
                }),
        );
    }

    col.into()
}

fn render_block_details<'a, Message: 'a>(b: &InspectorBlock) -> Element<'a, Message> {
    let mut col = column![
        widgets::field::<Message>("Type", b.kind.clone()),
        bbox_field::<Message>(&b.bbox),
        widgets::field::<Message>("Pages", format!("{:?}", b.pages)),
    ]
    .spacing(theme::SPACING_MD);

    if let Some(td) = &b.table_details {
        col = col.push(
            column![
                widgets::v_space(theme::SPACING_SM),
                widgets::section_header("Table Info"),
                widgets::field::<Message>("Algorithm", td.algorithm.clone()),
                widgets::field::<Message>("Rows", td.rows.to_string()),
                widgets::field::<Message>("Columns", td.cols.to_string()),
            ]
            .spacing(theme::SPACING_SM),
        );
    }

    if !b.text.is_empty() {
        col = col.push(
            column![
                widgets::v_space(theme::SPACING_SM),
                widgets::section_header("Text Content"),
                render_text_box::<Message>(&b.text)
            ]
            .spacing(theme::SPACING_SM),
        );
    }

    col.into()
}

fn render_element_details<'a, Message: 'a>(e: &InspectorElement) -> Element<'a, Message> {
    let mut col = column![
        widgets::field::<Message>("Type", e.kind.clone()),
        widgets::field::<Message>("Layout Ref", e.layout_ref.to_string()),
        bbox_field::<Message>(&e.bbox),
    ]
    .spacing(theme::SPACING_SM);

    if !e.text.is_empty() {
        col = col.push(
            column![
                widgets::v_space(theme::SPACING_SM),
                widgets::section_header("Text Content"),
                render_text_box::<Message>(&e.text)
            ]
            .spacing(theme::SPACING_SM),
        );
    }

    col.into()
}

fn render_layout_details<'a, Message: 'a>(l: &InspectorLayout) -> Element<'a, Message> {
    column![
        widgets::field::<Message>("Label", l.label.clone()),
        widgets::field::<Message>("Confidence", format!("{:.2}%", l.proba * 100.0)),
        bbox_field::<Message>(&l.bbox),
    ]
    .spacing(theme::SPACING_SM)
    .into()
}

fn render_text_box<'a, Message: 'a>(content: &str) -> Element<'a, Message> {
    container(
        text(content.to_string())
            .size(theme::TEXT_SIZE_MD)
            .line_height(iced::widget::text::LineHeight::Relative(1.4)),
    )
    .padding(theme::PADDING_MD)
    .style(|_| container::Style {
        background: Some(Color::from_rgba(1.0, 1.0, 1.0, 0.03).into()),
        border: iced::Border {
            color: theme::SURFACE1,
            width: 1.0,
            radius: theme::BORDER_RADIUS_SM.into(),
        },
        ..Default::default()
    })
    .into()
}

fn bbox_field<'a, Message: 'a>(bbox: &[f32; 4]) -> Element<'a, Message> {
    let [x0, y0, x1, y1] = bbox;
    let w = (x1 - x0).abs();
    let h = (y1 - y0).abs();

    column![
        text("Geometry")
            .size(theme::TEXT_SIZE_SM)
            .color(theme::SUBTEXT0)
            .font(theme::FONT_BOLD),
        row![
            text(format!("X:{:.1} Y:{:.1} W:{:.1} H:{:.1}", x0, y0, w, h))
                .size(theme::TEXT_SIZE_SM)
                .font(theme::FONT_MONOSPACE)
        ]
    ]
    .spacing(2)
    .into()
}

fn render_cell_details<'a, Message: 'a>(c: &InspectorCell) -> Element<'a, Message> {
    let mut col = column![
        widgets::field::<Message>(
            "Position",
            format!("Row: {}, Col: {}", c.row_idx, c.col_idx)
        ),
        widgets::field::<Message>("Span", format!("R: {}, C: {}", c.row_span, c.col_span)),
        bbox_field::<Message>(&c.bbox),
    ]
    .spacing(theme::SPACING_SM);

    if !c.text.is_empty() {
        col = col.push(
            column![
                widgets::v_space(theme::SPACING_SM),
                widgets::section_header("Cell Text"),
                render_text_box::<Message>(&c.text)
            ]
            .spacing(theme::SPACING_SM),
        );
    }

    col.into()
}
