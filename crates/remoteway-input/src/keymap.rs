//! XKB keymap handling and memfd creation for the virtual keyboard Wayland protocol.

use std::io::Write;
use std::os::fd::OwnedFd;

use nix::sys::memfd::{MemFdCreateFlag, memfd_create};

use crate::error::InputError;

/// Minimal XKB keymap for en-US QWERTY layout.
///
/// This is used as a default when the client hasn't yet transmitted its keymap.
/// The virtual keyboard protocol requires `keymap()` before any `key()` request.
pub const DEFAULT_KEYMAP: &str = r#"xkb_keymap {
    xkb_keycodes "evdev" {
        minimum = 8;
        maximum = 255;
        <ESC>  = 9;
        <AE01> = 10; <AE02> = 11; <AE03> = 12; <AE04> = 13;
        <AE05> = 14; <AE06> = 15; <AE07> = 16; <AE08> = 17;
        <AE09> = 18; <AE10> = 19; <AE11> = 20; <AE12> = 21;
        <BKSP> = 22;
        <TAB>  = 23;
        <AD01> = 24; <AD02> = 25; <AD03> = 26; <AD04> = 27;
        <AD05> = 28; <AD06> = 29; <AD07> = 30; <AD08> = 31;
        <AD09> = 32; <AD10> = 33; <AD11> = 34; <AD12> = 35;
        <RTRN> = 36;
        <LCTL> = 37;
        <AC01> = 38; <AC02> = 39; <AC03> = 40; <AC04> = 41;
        <AC05> = 42; <AC06> = 43; <AC07> = 44; <AC08> = 45;
        <AC09> = 46; <AC10> = 47; <AC11> = 48;
        <TLDE> = 49;
        <LFSH> = 50;
        <BKSL> = 51;
        <AB01> = 52; <AB02> = 53; <AB03> = 54; <AB04> = 55;
        <AB05> = 56; <AB06> = 57; <AB07> = 58; <AB08> = 59;
        <AB09> = 60; <AB10> = 61;
        <RTSH> = 62;
        <LALT> = 64;
        <SPCE> = 65;
        <CAPS> = 66;
        <FK01> = 67; <FK02> = 68; <FK03> = 69; <FK04> = 70;
        <FK05> = 71; <FK06> = 72; <FK07> = 73; <FK08> = 74;
        <FK09> = 75; <FK10> = 76;
        <FK11> = 95; <FK12> = 96;
        <RCTL> = 105;
        <RALT> = 108;
        <UP>   = 111; <LEFT> = 113; <DOWN> = 116; <RGHT> = 114;
        <HOME> = 110; <END>  = 115; <PGUP> = 112; <PGDN> = 117;
        <INS>  = 118; <DELE> = 119;
        <LWIN> = 133; <RWIN> = 134;
        <MENU> = 135;
    };
    xkb_types "default" {
        type "ONE_LEVEL" { modifiers = None; map[None] = Level1; };
        type "TWO_LEVEL" {
            modifiers = Shift;
            map[None]  = Level1;
            map[Shift] = Level2;
        };
        type "ALPHABETIC" {
            modifiers = Shift+Lock;
            map[None]       = Level1;
            map[Shift]      = Level2;
            map[Lock]       = Level2;
            map[Shift+Lock] = Level1;
        };
    };
    xkb_compatibility "default" {
        interpret Any+AnyOf(all) { action = SetMods(modifiers=modMapMods,clearLocks); };
        interpret Shift_L+AnyOf(all) { action = SetMods(modifiers=Shift,clearLocks); };
        interpret Caps_Lock+AnyOf(all) { action = LockMods(modifiers=Lock); };
        interpret Control_L+AnyOf(all) { action = SetMods(modifiers=Control,clearLocks); };
        interpret Alt_L+AnyOf(all) { action = SetMods(modifiers=Mod1,clearLocks); };
    };
    xkb_symbols "us" {
        key <ESC>  { [ Escape ] };
        key <AE01> { [ 1, exclam ] };
        key <AE02> { [ 2, at ] };
        key <AE03> { [ 3, numbersign ] };
        key <AE04> { [ 4, dollar ] };
        key <AE05> { [ 5, percent ] };
        key <AE06> { [ 6, asciicircum ] };
        key <AE07> { [ 7, ampersand ] };
        key <AE08> { [ 8, asterisk ] };
        key <AE09> { [ 9, parenleft ] };
        key <AE10> { [ 0, parenright ] };
        key <AE11> { [ minus, underscore ] };
        key <AE12> { [ equal, plus ] };
        key <BKSP> { [ BackSpace ] };
        key <TAB>  { [ Tab, ISO_Left_Tab ] };
        key <AD01> { [ q, Q ] }; key <AD02> { [ w, W ] };
        key <AD03> { [ e, E ] }; key <AD04> { [ r, R ] };
        key <AD05> { [ t, T ] }; key <AD06> { [ y, Y ] };
        key <AD07> { [ u, U ] }; key <AD08> { [ i, I ] };
        key <AD09> { [ o, O ] }; key <AD10> { [ p, P ] };
        key <AD11> { [ bracketleft, braceleft ] };
        key <AD12> { [ bracketright, braceright ] };
        key <RTRN> { [ Return ] };
        key <LCTL> { [ Control_L ] };
        key <AC01> { [ a, A ] }; key <AC02> { [ s, S ] };
        key <AC03> { [ d, D ] }; key <AC04> { [ f, F ] };
        key <AC05> { [ g, G ] }; key <AC06> { [ h, H ] };
        key <AC07> { [ j, J ] }; key <AC08> { [ k, K ] };
        key <AC09> { [ l, L ] }; key <AC10> { [ semicolon, colon ] };
        key <AC11> { [ apostrophe, quotedbl ] };
        key <TLDE> { [ grave, asciitilde ] };
        key <LFSH> { [ Shift_L ] };
        key <BKSL> { [ backslash, bar ] };
        key <AB01> { [ z, Z ] }; key <AB02> { [ x, X ] };
        key <AB03> { [ c, C ] }; key <AB04> { [ v, V ] };
        key <AB05> { [ b, B ] }; key <AB06> { [ n, N ] };
        key <AB07> { [ m, M ] }; key <AB08> { [ comma, less ] };
        key <AB09> { [ period, greater ] }; key <AB10> { [ slash, question ] };
        key <RTSH> { [ Shift_R ] };
        key <LALT> { [ Alt_L ] };
        key <SPCE> { [ space ] };
        key <CAPS> { [ Caps_Lock ] };
        key <FK01> { [ F1 ] }; key <FK02> { [ F2 ] };
        key <FK03> { [ F3 ] }; key <FK04> { [ F4 ] };
        key <FK05> { [ F5 ] }; key <FK06> { [ F6 ] };
        key <FK07> { [ F7 ] }; key <FK08> { [ F8 ] };
        key <FK09> { [ F9 ] }; key <FK10> { [ F10 ] };
        key <FK11> { [ F11 ] }; key <FK12> { [ F12 ] };
        key <RCTL> { [ Control_R ] };
        key <RALT> { [ Alt_R ] };
        key <UP>   { [ Up ] }; key <DOWN> { [ Down ] };
        key <LEFT> { [ Left ] }; key <RGHT> { [ Right ] };
        key <HOME> { [ Home ] }; key <END>  { [ End ] };
        key <PGUP> { [ Prior ] }; key <PGDN> { [ Next ] };
        key <INS>  { [ Insert ] }; key <DELE> { [ Delete ] };
        key <LWIN> { [ Super_L ] }; key <RWIN> { [ Super_R ] };
        key <MENU> { [ Menu ] };
        modifier_map Shift { <LFSH>, <RTSH> };
        modifier_map Lock  { <CAPS> };
        modifier_map Control { <LCTL>, <RCTL> };
        modifier_map Mod1 { <LALT>, <RALT> };
        modifier_map Mod4 { <LWIN>, <RWIN> };
    };
};
"#;

/// Create a memfd containing the given XKB keymap string.
///
/// Returns the owned fd and the size in bytes (including NUL terminator).
/// The virtual keyboard protocol requires `XKB_KEYMAP_FORMAT_TEXT_V1` format,
/// which is a NUL-terminated string passed via a file descriptor.
pub fn create_keymap_fd(keymap: &str) -> Result<(OwnedFd, u32), InputError> {
    let fd = memfd_create(
        c"remoteway-keymap",
        MemFdCreateFlag::MFD_CLOEXEC | MemFdCreateFlag::MFD_ALLOW_SEALING,
    )
    .map_err(|e| InputError::Keymap(format!("memfd_create: {e}")))?;

    let mut file = std::fs::File::from(fd);
    // Write the keymap string with a NUL terminator.
    file.write_all(keymap.as_bytes())
        .map_err(|e| InputError::Keymap(format!("write: {e}")))?;
    file.write_all(b"\0")
        .map_err(|e| InputError::Keymap(format!("write NUL: {e}")))?;

    let size = keymap.len() as u32 + 1; // +1 for NUL

    let fd = OwnedFd::from(file);
    Ok((fd, size))
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::os::fd::AsFd;

    use super::*;

    #[test]
    fn default_keymap_is_not_empty() {
        assert!(DEFAULT_KEYMAP.len() > 100);
        assert!(DEFAULT_KEYMAP.contains("xkb_keymap"));
        assert!(DEFAULT_KEYMAP.contains("xkb_keycodes"));
        assert!(DEFAULT_KEYMAP.contains("xkb_symbols"));
    }

    #[test]
    fn create_keymap_fd_returns_valid_fd() {
        let (fd, size) = create_keymap_fd(DEFAULT_KEYMAP).unwrap();
        assert_eq!(size, DEFAULT_KEYMAP.len() as u32 + 1);
        // Verify the fd is valid by reading from it.
        let mut file = std::fs::File::from(fd);
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();
        assert_eq!(buf.len(), size as usize);
        assert_eq!(&buf[..buf.len() - 1], DEFAULT_KEYMAP.as_bytes());
        assert_eq!(buf[buf.len() - 1], 0); // NUL terminator
    }

    #[test]
    fn create_keymap_fd_small_string() {
        let small = "test keymap";
        let (fd, size) = create_keymap_fd(small).unwrap();
        assert_eq!(size, small.len() as u32 + 1);
        // Verify fd is valid.
        fd.as_fd();
    }

    #[test]
    fn create_keymap_fd_empty_string() {
        let (_, size) = create_keymap_fd("").unwrap();
        assert_eq!(size, 1); // just the NUL terminator
    }

    #[test]
    fn create_keymap_fd_large_keymap() {
        // Test with a keymap string large enough to exercise write buffering.
        let large = "x".repeat(65536);
        let (fd, size) = create_keymap_fd(&large).unwrap();
        assert_eq!(size, 65536 + 1);

        let mut file = std::fs::File::from(fd);
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();
        assert_eq!(buf.len(), 65537);
        assert!(buf[..65536].iter().all(|&b| b == b'x'));
        assert_eq!(buf[65536], 0); // NUL terminator
    }

    #[test]
    fn create_keymap_fd_binary_content() {
        // Keymap can contain any valid UTF-8 string.
        let content = "xkb_keymap {\n  special chars: \t\n};\n";
        let (fd, size) = create_keymap_fd(content).unwrap();
        assert_eq!(size, content.len() as u32 + 1);

        let mut file = std::fs::File::from(fd);
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();
        assert_eq!(&buf[..buf.len() - 1], content.as_bytes());
        assert_eq!(buf[buf.len() - 1], 0);
    }

    #[test]
    fn create_keymap_fd_nul_terminator_present() {
        // Verify the NUL terminator is exactly at the expected position.
        let s = "hello";
        let (fd, size) = create_keymap_fd(s).unwrap();
        assert_eq!(size, 6);

        let mut file = std::fs::File::from(fd);
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"hello\0");
    }

    #[test]
    fn default_keymap_has_all_key_sections() {
        // Verify specific important keycodes are present.
        assert!(DEFAULT_KEYMAP.contains("<ESC>"));
        assert!(DEFAULT_KEYMAP.contains("<SPCE>"));
        assert!(DEFAULT_KEYMAP.contains("<RTRN>"));
        assert!(DEFAULT_KEYMAP.contains("Escape"));
        assert!(DEFAULT_KEYMAP.contains("space"));
        assert!(DEFAULT_KEYMAP.contains("Return"));
        assert!(DEFAULT_KEYMAP.contains("Shift_L"));
        assert!(DEFAULT_KEYMAP.contains("Control_L"));
        assert!(DEFAULT_KEYMAP.contains("Alt_L"));
    }

    #[test]
    fn default_keymap_has_modifier_maps() {
        assert!(DEFAULT_KEYMAP.contains("modifier_map Shift"));
        assert!(DEFAULT_KEYMAP.contains("modifier_map Lock"));
        assert!(DEFAULT_KEYMAP.contains("modifier_map Control"));
        assert!(DEFAULT_KEYMAP.contains("modifier_map Mod1"));
        assert!(DEFAULT_KEYMAP.contains("modifier_map Mod4"));
    }

    #[test]
    fn keymap_error_variant_display() {
        use crate::error::InputError;
        let err = InputError::Keymap("memfd_create: EMFILE".to_string());
        let display = err.to_string();
        assert!(display.contains("keymap creation failed"));
        assert!(display.contains("memfd_create: EMFILE"));
    }

    #[test]
    fn default_keymap_minimum_maximum_keycodes() {
        assert!(DEFAULT_KEYMAP.contains("minimum = 8"));
        assert!(DEFAULT_KEYMAP.contains("maximum = 255"));
    }

    #[test]
    fn default_keymap_type_definitions() {
        assert!(DEFAULT_KEYMAP.contains("ONE_LEVEL"));
        assert!(DEFAULT_KEYMAP.contains("TWO_LEVEL"));
        assert!(DEFAULT_KEYMAP.contains("ALPHABETIC"));
    }
}
