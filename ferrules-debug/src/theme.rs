use iced::{Color, Font, font::Weight};

// --- PALETTE: Catppuccin Mocha ---
pub const BASE: Color = Color::from_rgb(0.117, 0.117, 0.180); // #1e1e2e
pub const SURFACE0: Color = Color::from_rgb(0.192, 0.196, 0.266); // #313244
pub const SURFACE1: Color = Color::from_rgb(0.270, 0.278, 0.352); // #45475a
pub const OVERLAY0: Color = Color::from_rgb(0.424, 0.447, 0.533); // #6c7086 (Adjusted for visibility)
pub const OVERLAY0_FADED: Color = Color { a: 0.3, ..OVERLAY0 };
pub const TEXT: Color = Color::from_rgb(0.803, 0.839, 0.956); // #cdd6f4
pub const SUBTEXT0: Color = Color::from_rgb(0.659, 0.694, 0.808); // #a6adc8

// Accents
pub const RED: Color = Color::from_rgb(0.953, 0.545, 0.659); // #f38ba8
pub const GREEN: Color = Color::from_rgb(0.651, 0.890, 0.631); // #a6e3a1
pub const BLUE: Color = Color::from_rgb(0.537, 0.706, 0.980); // #89b4fa
pub const PEACH: Color = Color::from_rgb(0.980, 0.702, 0.522); // #fab387
pub const MAUVE: Color = Color::from_rgb(0.796, 0.651, 0.969); // #cba6f7
pub const LAVENDER: Color = Color::from_rgb(0.706, 0.745, 0.996); // #b4befe
pub const YELLOW: Color = Color::from_rgb(0.976, 0.890, 0.686); // #f9e2af
pub const ORANGE: Color = Color::from_rgb(0.980, 0.545, 0.447); // #fa8b72

pub const SELECTION_BG: Color = Color {
    r: 0.976,
    g: 0.890,
    b: 0.686,
    a: 0.2,
}; // Yellow with alpha
pub const SELECTION_BORDER: Color = YELLOW;
pub const TABLE_SELECTION: Color = ORANGE;

// --- TYPOGRAPHY ---
pub const FONT_BOLD: Font = Font {
    weight: Weight::Bold,
    ..Font::DEFAULT
};

pub const FONT_MEDIUM: Font = Font {
    weight: Weight::Medium,
    ..Font::DEFAULT
};

pub const FONT_MONOSPACE: Font = Font::MONOSPACE;

// --- SIZES ---
pub const TEXT_SIZE_SM: f32 = 11.0;
pub const TEXT_SIZE_MD: f32 = 13.0;
pub const TEXT_SIZE_LG: f32 = 14.0;
pub const TEXT_SIZE_XL: f32 = 18.0;

pub const ICON_SIZE_SM: f32 = 16.0;
pub const ICON_SIZE_MD: f32 = 24.0;

// --- LAYOUT ---
pub const SPACING_SM: f32 = 5.0;
pub const SPACING_MD: f32 = 10.0;
pub const SPACING_LG: f32 = 15.0;
pub const SPACING_XL: f32 = 20.0;

pub const PADDING_SM: f32 = 5.0;
pub const PADDING_MD: f32 = 10.0;
pub const PADDING_LG: f32 = 20.0;

pub const BORDER_RADIUS_SM: f32 = 4.0;
pub const BORDER_RADIUS_MD: f32 = 8.0;
pub const BORDER_RADIUS_LG: f32 = 20.0;

pub const SIDEBAR_WIDTH_OPEN: f32 = 240.0;
pub const SIDEBAR_WIDTH_CLOSED: f32 = 60.0;
pub const INSPECTOR_WIDTH: f32 = 320.0;
