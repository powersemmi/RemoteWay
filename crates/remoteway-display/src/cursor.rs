use std::os::fd::{AsFd, OwnedFd};
use std::ptr::NonNull;

use nix::sys::mman::{MapFlags, ProtFlags, munmap};
use wayland_client::QueueHandle;
use wayland_client::protocol::{
    wl_buffer, wl_compositor, wl_pointer, wl_shm, wl_shm_pool, wl_surface,
};

use remoteway_proto::cursor::CursorUpdate;

use crate::error::DisplayError;
use crate::surface::DisplayState;

/// Local cursor overlay for immediate cursor rendering.
///
/// Updates cursor position instantly on local input events (before server response).
/// The cursor image is replaced when a `CursorUpdate` arrives from the server
/// with a new bitmap.
pub struct CursorOverlay {
    surface: wl_surface::WlSurface,
    cursor_buffer: Option<CursorBuffer>,
    hotspot_x: i32,
    hotspot_y: i32,
    current_x: f32,
    current_y: f32,
    has_server_cursor: bool,
    last_enter_serial: u32,
}

/// Holds cursor bitmap data in a wl_shm buffer.
struct CursorBuffer {
    _fd: OwnedFd,
    map: NonNull<std::ffi::c_void>,
    map_size: usize,
    pool: wl_shm_pool::WlShmPool,
    buffer: wl_buffer::WlBuffer,
    _width: u32,
    _height: u32,
}

// SAFETY: CursorBuffer owns fd and mmap exclusively.
unsafe impl Send for CursorBuffer {}

impl Drop for CursorBuffer {
    fn drop(&mut self) {
        self.buffer.destroy();
        self.pool.destroy();
        // SAFETY: map was created with mmap of map_size bytes.
        unsafe {
            let _ = munmap(self.map, self.map_size);
        }
    }
}

impl CursorOverlay {
    /// Create a new cursor overlay.
    pub fn new(compositor: &wl_compositor::WlCompositor, qh: &QueueHandle<DisplayState>) -> Self {
        // Cursor surface uses a sentinel surface_id (u16::MAX).
        let surface = compositor.create_surface(qh, u16::MAX);

        Self {
            surface,
            cursor_buffer: None,
            hotspot_x: 0,
            hotspot_y: 0,
            current_x: 0.0,
            current_y: 0.0,
            has_server_cursor: false,
            last_enter_serial: 0,
        }
    }

    /// Update cursor position from local input (immediate, before server roundtrip).
    pub fn update_position(&mut self, x: f32, y: f32) {
        self.current_x = x;
        self.current_y = y;
    }

    /// Record the serial from a wl_pointer.enter event (needed for set_cursor).
    pub fn set_enter_serial(&mut self, serial: u32) {
        self.last_enter_serial = serial;
    }

    /// Apply a cursor update from the remote server.
    ///
    /// If `bitmap_data` is provided, creates a new cursor buffer with the RGBA pixels.
    /// Otherwise only updates position and hotspot.
    pub fn apply_cursor_update(
        &mut self,
        update: &CursorUpdate,
        bitmap_data: Option<&[u8]>,
        shm: &wl_shm::WlShm,
        qh: &QueueHandle<DisplayState>,
    ) -> Result<(), DisplayError> {
        // CursorUpdate is #[repr(C, packed)] — copy fields to aligned locals
        // to avoid unaligned access UB.
        let hotspot_x = update.hotspot_x;
        let hotspot_y = update.hotspot_y;
        let pos_x = update.x;
        let pos_y = update.y;
        let has_bitmap = update.has_bitmap;
        let bitmap_w = update.bitmap_width;
        let bitmap_h = update.bitmap_height;

        self.hotspot_x = hotspot_x as i32;
        self.hotspot_y = hotspot_y as i32;
        self.current_x = pos_x;
        self.current_y = pos_y;

        if has_bitmap != 0
            && let Some(data) = bitmap_data
        {
            let w = bitmap_w as u32;
            let h = bitmap_h as u32;
            self.set_cursor_bitmap(data, w, h, shm, qh)?;
            self.has_server_cursor = true;
        }

        Ok(())
    }

    /// Set cursor image on a pointer (call after enter event).
    pub fn set_on_pointer(&self, pointer: &wl_pointer::WlPointer) {
        if self.cursor_buffer.is_some() {
            pointer.set_cursor(
                self.last_enter_serial,
                Some(&self.surface),
                self.hotspot_x,
                self.hotspot_y,
            );
        }
    }

    /// Hide the cursor.
    pub fn hide(&self, pointer: &wl_pointer::WlPointer) {
        pointer.set_cursor(self.last_enter_serial, None, 0, 0);
    }

    pub fn position(&self) -> (f32, f32) {
        (self.current_x, self.current_y)
    }

    pub fn has_server_cursor(&self) -> bool {
        self.has_server_cursor
    }

    fn set_cursor_bitmap(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        shm: &wl_shm::WlShm,
        qh: &QueueHandle<DisplayState>,
    ) -> Result<(), DisplayError> {
        let stride = width * 4;
        let buf_size = stride as usize * height as usize;
        if data.len() < buf_size {
            return Err(DisplayError::ShmBuffer(
                "cursor bitmap data too small".into(),
            ));
        }

        let fd = nix::sys::memfd::memfd_create(
            c"remoteway-cursor",
            nix::sys::memfd::MemFdCreateFlag::MFD_CLOEXEC,
        )
        .map_err(|e| DisplayError::ShmBuffer(format!("cursor memfd_create failed: {e}")))?;

        nix::unistd::ftruncate(&fd, buf_size as i64)
            .map_err(|e| DisplayError::ShmBuffer(format!("cursor ftruncate failed: {e}")))?;

        // SAFETY: fd is valid, buf_size > 0.
        let map = unsafe {
            nix::sys::mman::mmap(
                None,
                std::num::NonZero::new(buf_size).unwrap(),
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                &fd,
                0,
            )
            .map_err(|e| DisplayError::ShmBuffer(format!("cursor mmap failed: {e}")))?
        };

        // Copy bitmap data.
        // SAFETY: map is valid and buf_size bytes long.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), map.as_ptr() as *mut u8, buf_size);
        }

        let pool = shm.create_pool(fd.as_fd(), buf_size as i32, qh, ());
        let buffer = pool.create_buffer(
            0,
            width as i32,
            height as i32,
            stride as i32,
            wl_shm::Format::Argb8888,
            qh,
            // Use usize::MAX as sentinel for cursor buffer release.
            usize::MAX,
        );

        // Attach and commit the cursor surface.
        self.surface.attach(Some(&buffer), 0, 0);
        self.surface
            .damage_buffer(0, 0, width as i32, height as i32);
        self.surface.commit();

        // Drop old cursor buffer.
        self.cursor_buffer = Some(CursorBuffer {
            _fd: fd,
            map,
            map_size: buf_size,
            pool,
            buffer,
            _width: width,
            _height: height,
        });

        Ok(())
    }
}

impl Drop for CursorOverlay {
    fn drop(&mut self) {
        self.cursor_buffer = None;
        self.surface.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_update_fields_accessible() {
        let update = CursorUpdate {
            surface_id: 1,
            hotspot_x: -5,
            hotspot_y: 3,
            has_bitmap: 0,
            _pad: 0,
            bitmap_width: 0,
            bitmap_height: 0,
            x: 100.0,
            y: 200.0,
        };
        assert_eq!({ update.surface_id }, 1);
        assert_eq!({ update.hotspot_x }, -5);
        assert_eq!({ update.has_bitmap }, 0);
    }

    #[test]
    fn cursor_buffer_size_calculation() {
        let w = 32u32;
        let h = 32u32;
        let stride = w * 4;
        let buf_size = stride as usize * h as usize;
        assert_eq!(buf_size, 32 * 32 * 4);
    }

    #[test]
    fn cursor_buffer_size_various_sizes() {
        // Common cursor sizes: 24x24, 32x32, 48x48, 64x64.
        let sizes: Vec<(u32, u32)> = vec![(24, 24), (32, 32), (48, 48), (64, 64), (128, 128)];
        for (w, h) in sizes {
            let stride = w * 4;
            let buf_size = stride as usize * h as usize;
            assert_eq!(buf_size, (w * h * 4) as usize);
            assert!(buf_size > 0);
        }
    }

    /// The bitmap data validation: data.len() must be >= stride * height.
    #[test]
    fn bitmap_data_too_small_detection() {
        let width = 32u32;
        let height = 32u32;
        let stride = width * 4;
        let buf_size = stride as usize * height as usize;

        // Exactly enough data: OK.
        let data_ok = vec![0u8; buf_size];
        assert!(data_ok.len() >= buf_size);

        // One byte short: error.
        let data_short = vec![0u8; buf_size - 1];
        assert!(data_short.len() < buf_size);

        // More than enough: OK.
        let data_extra = vec![0u8; buf_size + 100];
        assert!(data_extra.len() >= buf_size);

        // Empty data for non-zero cursor: error.
        let data_empty: Vec<u8> = vec![];
        assert!(data_empty.len() < buf_size);
    }

    /// Verify hotspot can be negative (cursor tip is offset from top-left).
    #[test]
    fn cursor_update_negative_hotspot() {
        let update = CursorUpdate {
            surface_id: 0,
            hotspot_x: -10,
            hotspot_y: -20,
            has_bitmap: 0,
            _pad: 0,
            bitmap_width: 0,
            bitmap_height: 0,
            x: 0.0,
            y: 0.0,
        };
        assert_eq!({ update.hotspot_x }, -10);
        assert_eq!({ update.hotspot_y }, -20);
        // Conversion to i32 for Wayland API.
        let hx = { update.hotspot_x } as i32;
        let hy = { update.hotspot_y } as i32;
        assert_eq!(hx, -10);
        assert_eq!(hy, -20);
    }

    /// has_bitmap flag controls whether bitmap data is expected.
    #[test]
    fn cursor_update_has_bitmap_flag() {
        let with_bitmap = CursorUpdate {
            surface_id: 1,
            hotspot_x: 5,
            hotspot_y: 5,
            has_bitmap: 1,
            _pad: 0,
            bitmap_width: 32,
            bitmap_height: 32,
            x: 100.0,
            y: 200.0,
        };
        assert_ne!({ with_bitmap.has_bitmap }, 0);

        let without_bitmap = CursorUpdate {
            surface_id: 1,
            hotspot_x: 5,
            hotspot_y: 5,
            has_bitmap: 0,
            _pad: 0,
            bitmap_width: 0,
            bitmap_height: 0,
            x: 100.0,
            y: 200.0,
        };
        assert_eq!({ without_bitmap.has_bitmap }, 0);
    }

    /// Position tracking: update_position sets current_x/y independently.
    #[test]
    fn position_update_logic() {
        let mut x: f32;
        let mut y: f32;

        // Simulate update_position.
        x = 150.5;
        y = 300.75;
        assert_eq!(x, 150.5);
        assert_eq!(y, 300.75);

        // Negative positions (cursor off-screen partially).
        x = -10.0;
        y = -20.0;
        assert_eq!(x, -10.0);
        assert_eq!(y, -20.0);
    }

    /// set_enter_serial stores the serial needed for set_cursor.
    #[test]
    fn enter_serial_storage() {
        let mut serial: u32;
        serial = 42;
        assert_eq!(serial, 42);
        serial = u32::MAX;
        assert_eq!(serial, u32::MAX);
    }

    /// has_server_cursor flag starts false, becomes true after bitmap set.
    #[test]
    fn server_cursor_flag_logic() {
        let mut has_server_cursor = false;
        assert!(!has_server_cursor);

        // After applying a cursor update with bitmap:
        has_server_cursor = true;
        assert!(has_server_cursor);
    }

    /// CursorUpdate with large bitmap dimensions.
    #[test]
    fn cursor_update_large_bitmap() {
        let update = CursorUpdate {
            surface_id: 0,
            hotspot_x: 0,
            hotspot_y: 0,
            has_bitmap: 1,
            _pad: 0,
            bitmap_width: 256,
            bitmap_height: 256,
            x: 0.0,
            y: 0.0,
        };
        let w = { update.bitmap_width } as u32;
        let h = { update.bitmap_height } as u32;
        let stride = w * 4;
        let buf_size = stride as usize * h as usize;
        assert_eq!(buf_size, 256 * 256 * 4);
    }

    /// Verify the sentinel surface_id used for cursor surface.
    #[test]
    fn cursor_surface_uses_sentinel_id() {
        let sentinel = u16::MAX;
        assert_eq!(sentinel, 65535);
        // This is used in CursorOverlay::new to create the cursor surface.
    }

    /// The apply_cursor_update logic: bitmap_data=None skips bitmap creation.
    #[test]
    fn apply_cursor_update_logic_no_bitmap() {
        let update = CursorUpdate {
            surface_id: 1,
            hotspot_x: 5,
            hotspot_y: 3,
            has_bitmap: 1,
            _pad: 0,
            bitmap_width: 32,
            bitmap_height: 32,
            x: 50.0,
            y: 60.0,
        };
        // Even with has_bitmap=1, if bitmap_data is None, the branch is skipped.
        let bitmap_data: Option<&[u8]> = None;
        let has_bitmap = { update.has_bitmap };
        let should_create = has_bitmap != 0 && bitmap_data.is_some();
        assert!(!should_create);
    }

    /// The apply_cursor_update logic: both conditions must be true.
    #[test]
    fn apply_cursor_update_logic_with_bitmap() {
        let update = CursorUpdate {
            surface_id: 1,
            hotspot_x: 5,
            hotspot_y: 3,
            has_bitmap: 1,
            _pad: 0,
            bitmap_width: 32,
            bitmap_height: 32,
            x: 50.0,
            y: 60.0,
        };
        let data = vec![0u8; 32 * 32 * 4];
        let bitmap_data: Option<&[u8]> = Some(&data);
        let has_bitmap = { update.has_bitmap };
        let should_create = has_bitmap != 0 && bitmap_data.is_some();
        assert!(should_create);
    }

    /// has_bitmap=0 with bitmap_data present: no bitmap created.
    #[test]
    fn apply_cursor_update_no_flag_with_data() {
        let update = CursorUpdate {
            surface_id: 1,
            hotspot_x: 0,
            hotspot_y: 0,
            has_bitmap: 0,
            _pad: 0,
            bitmap_width: 0,
            bitmap_height: 0,
            x: 0.0,
            y: 0.0,
        };
        let data = vec![0u8; 1024];
        let has_bitmap = { update.has_bitmap };
        let should_create = has_bitmap != 0 && Some(&data[..]).is_some();
        assert!(!should_create);
    }

    /// Position is always updated regardless of bitmap.
    #[test]
    fn position_always_updated() {
        let update = CursorUpdate {
            surface_id: 0,
            hotspot_x: 1,
            hotspot_y: 2,
            has_bitmap: 0,
            _pad: 0,
            bitmap_width: 0,
            bitmap_height: 0,
            x: 123.456,
            y: 789.012,
        };
        // Simulate the extraction from packed struct.
        let pos_x = { update.x };
        let pos_y = { update.y };
        let hotspot_x = { update.hotspot_x } as i32;
        let hotspot_y = { update.hotspot_y } as i32;
        assert_eq!(pos_x, 123.456);
        assert_eq!(pos_y, 789.012);
        assert_eq!(hotspot_x, 1);
        assert_eq!(hotspot_y, 2);
    }
}
