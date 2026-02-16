use crate::theme;
use iced::{
    widget::{button, container, row, text, Space},
    Element,
};

// --- TEXT HELPER ---
pub fn header_text<'a>(content: impl Into<String>) -> text::Text<'a> {
    let s: String = content.into();
    text(s)
        .size(theme::TEXT_SIZE_SM)
        .color(theme::SUBTEXT0)
        .font(theme::FONT_BOLD)
}

pub fn body_text<'a>(content: impl Into<String>) -> text::Text<'a> {
    let s: String = content.into();
    text(s).size(theme::TEXT_SIZE_MD).color(theme::TEXT)
}

pub fn section_header<'a>(content: impl Into<String>) -> text::Text<'a> {
    let s: String = content.into();
    text(s)
        .size(theme::TEXT_SIZE_SM)
        .font(theme::FONT_BOLD)
        .color(theme::OVERLAY0)
}

// --- BUTTONS ---
pub fn icon_button<'a, Message>(
    icon: text::Text<'a>,
    on_press: Message,
) -> button::Button<'a, Message>
where
    Message: Clone + 'a,
{
    button(icon)
        .on_press(on_press)
        .padding(theme::PADDING_SM)
        .style(button::text)
}

// --- CONTAINERS ---
pub fn panel_container<'a, Message>(
    content: impl Into<Element<'a, Message>>,
) -> container::Container<'a, Message> {
    container(content).style(|_| container::Style {
        background: Some(theme::SURFACE0.into()),
        ..Default::default()
    })
}

// --- FIELDS ---
pub fn field<'a, Message: 'a>(key: &'a str, value: String) -> Element<'a, Message> {
    row![
        text(format!("{}:", key))
            .size(theme::TEXT_SIZE_SM)
            .color(theme::SUBTEXT0)
            .font(theme::FONT_BOLD),
        text(value).size(theme::TEXT_SIZE_SM),
    ]
    .spacing(theme::SPACING_MD)
    .into()
}

// --- SPACERS ---
pub fn v_space<'a>(height: f32) -> Space {
    Space::with_height(height)
}

pub fn h_space<'a>(width: f32) -> Space {
    Space::with_width(width)
}
