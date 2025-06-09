// ANSI color codes
pub const RED: &str = "\x1b[31m";
pub const YELLOW: &str = "\x1b[33m";
pub const CYAN: &str = "\x1b[36m";
pub const WHITE: &str = "\x1b[37m";
pub const BOLD: &str = "\x1b[1m";
pub const RESET: &str = "\x1b[0m";
pub const DIM: &str = "\x1b[2m";

pub fn format_error(error_type: &str, message: &str, details: Vec<(&str, String)>) {
    // Print error header with border
    eprintln!("\n{RED}{BOLD}╭─────────────────────────────────────────────────────────────────╮{RESET}");
    eprintln!("{RED}{BOLD}│ ✖ ERROR: {:<54}│{RESET}", error_type);
    eprintln!("{RED}{BOLD}╰─────────────────────────────────────────────────────────────────╯{RESET}");
    
    // Print main message
    eprintln!("\n{WHITE}{message}{RESET}");
    
    // Print details if any
    if !details.is_empty() {
        eprintln!("\n{CYAN}{BOLD}Details:{RESET}");
        for (label, value) in details {
            eprintln!("  {DIM}•{RESET} {YELLOW}{label}:{RESET} {value}");
        }
    }
    
    // Print footer with suggestion
    eprintln!("\n{DIM}For more information, try running with --debug flag{RESET}");
    eprintln!("{RED}{BOLD}═══════════════════════════════════════════════════════════════════{RESET}\n");
}