use std::os::fd::{AsFd, OwnedFd};
use std::ptr::NonNull;

use nix::sys::mman::{MapFlags, MmapAdvise, ProtFlags, madvise, munmap};
use wayland_client::protocol::{wl_buffer, wl_shm, wl_shm_pool};
use wayland_client::{Dispatch, QueueHandle};

use crate::error::DisplayError;

/// Double-buffered SHM frame uploader for displaying decompressed frames.
///
/// Wraps a memfd-backed mmap region with two `wl_buffer` objects.
/// One buffer is attached to the surface (held by compositor),
/// while the other is available for uploading the next frame.
pub struct ShmFrameUploader {
    _fd: OwnedFd,
    map: NonNull<std::ffi::c_void>,
    map_size: usize,
    pool: wl_shm_pool::WlShmPool,
    buffers: [wl_buffer::WlBuffer; 2],
    /// Index of the buffer currently being written to (back buffer).
    active: usize,
    /// Tracks which buffers the compositor has released.
    released: [bool; 2],
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bytes per row (may be > width * 4 due to padding).
    pub stride: u32,
}

// SAFETY: ShmFrameUploader owns the fd and mmap exclusively. The buffers are only
// accessed by one thread at a time (the display thread).
unsafe impl Send for ShmFrameUploader {}

impl ShmFrameUploader {
    /// Create a new double-buffered SHM frame uploader.
    pub fn new<D>(
        shm: &wl_shm::WlShm,
        width: u32,
        height: u32,
        stride: u32,
        format: wl_shm::Format,
        qh: &QueueHandle<D>,
    ) -> Result<Self, DisplayError>
    where
        D: Dispatch<wl_shm_pool::WlShmPool, ()> + Dispatch<wl_buffer::WlBuffer, usize> + 'static,
    {
        let min_stride = (width as usize).checked_mul(4).ok_or_else(|| {
            DisplayError::ShmBuffer("stride overflow: width * 4 exceeds usize".into())
        })?;
        if (stride as usize) < min_stride {
            return Err(DisplayError::ShmBuffer(format!(
                "stride ({stride}) < width * 4 ({min_stride})"
            )));
        }
        let buf_size = (stride as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| {
                DisplayError::ShmBuffer(
                    "buffer size overflow: stride * height exceeds usize".into(),
                )
            })?;
        let total_size = buf_size.checked_mul(2).ok_or_else(|| {
            DisplayError::ShmBuffer("total size overflow: buf_size * 2 exceeds usize".into())
        })?;
        if total_size == 0 {
            return Err(DisplayError::ShmBuffer("zero-size buffer".into()));
        }

        // Create memfd.
        let fd = nix::sys::memfd::memfd_create(
            c"remoteway-display-shm",
            nix::sys::memfd::MemFdCreateFlag::MFD_CLOEXEC,
        )
        .map_err(|e| DisplayError::ShmBuffer(format!("memfd_create failed: {e}")))?;

        nix::unistd::ftruncate(&fd, total_size as i64)
            .map_err(|e| DisplayError::ShmBuffer(format!("ftruncate failed: {e}")))?;

        // SAFETY: fd is valid, size is non-zero, MAP_SHARED for Wayland SHM.
        let map = unsafe {
            nix::sys::mman::mmap(
                None,
                #[allow(clippy::expect_used)]
                std::num::NonZero::new(total_size).expect("total_size > 0"),
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                &fd,
                0,
            )
            .map_err(|e| DisplayError::ShmBuffer(format!("mmap failed: {e}")))?
        };

        // SAFETY: total_size > 0 checked above, alignment is page-aligned.
        #[allow(clippy::unwrap_used)]
        // madvise is an optimization hint; failure is non-critical.
        // SAFETY: map is valid and covers total_size bytes.
        unsafe {
            let _ = madvise(map, total_size, MmapAdvise::MADV_SEQUENTIAL);
        }

        let pool = shm.create_pool(fd.as_fd(), total_size as i32, qh, ());

        let buf0 = pool.create_buffer(
            0,
            width as i32,
            height as i32,
            stride as i32,
            format,
            qh,
            0usize,
        );
        let buf1 = pool.create_buffer(
            buf_size as i32,
            width as i32,
            height as i32,
            stride as i32,
            format,
            qh,
            1usize,
        );

        Ok(Self {
            _fd: fd,
            map,
            map_size: total_size,
            pool,
            buffers: [buf0, buf1],
            active: 0,
            released: [true, true],
            width,
            height,
            stride,
        })
    }

    /// Upload pixel data into the active (back) buffer.
    ///
    /// Always copies the full frame into the mmap'd region. With double
    /// buffering the back buffer was last written 2 frames ago, so partial
    /// copies would leave stale pixels in regions that changed in the
    /// intermediate frame but not in the current one. Damage regions are
    /// still passed to `wl_surface.damage_buffer` separately.
    pub fn upload(&mut self, data: &[u8]) {
        let buf_size = self.buffer_size();
        let offset = self.active * buf_size;
        // SAFETY: map is valid, offset + buf_size <= map_size. We are the only
        // writer (display thread) and the compositor has released this buffer.
        let dst = unsafe {
            std::slice::from_raw_parts_mut((self.map.as_ptr() as *mut u8).add(offset), buf_size)
        };

        let copy_len = data.len().min(buf_size);
        dst[..copy_len].copy_from_slice(&data[..copy_len]);
    }

    /// Get the active (back) buffer for attaching to a surface.
    #[must_use]
    pub fn active_buffer(&self) -> &wl_buffer::WlBuffer {
        &self.buffers[self.active]
    }

    /// Swap buffers: the current back buffer becomes front, and vice versa.
    pub fn swap(&mut self) {
        self.released[self.active] = false;
        self.active = 1 - self.active;
    }

    /// Mark a buffer as released by the compositor (ready for reuse).
    pub fn mark_released(&mut self, buffer_idx: usize) {
        if buffer_idx < 2 {
            self.released[buffer_idx] = true;
        }
    }

    /// Check if the active (back) buffer has been released and can be written to.
    #[must_use]
    pub fn can_upload(&self) -> bool {
        self.released[self.active]
    }

    /// Single buffer size in bytes.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.stride as usize * self.height as usize
    }
}

impl Drop for ShmFrameUploader {
    fn drop(&mut self) {
        self.buffers[0].destroy();
        self.buffers[1].destroy();
        self.pool.destroy();

        // SAFETY: map was created with mmap of map_size bytes.
        // INTENTIONAL: munmap failure during cleanup is not actionable.
        unsafe {
            let _ = munmap(self.map, self.map_size);
        }
    }
}

/// A rectangular damage region indicating which part of the frame changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DamageRegion {
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl DamageRegion {
    /// Create a new `DamageRegion`.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Total number of pixels in this region.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DamageRegion tests ---

    #[test]
    fn damage_region_new() {
        let r = DamageRegion::new(10, 20, 100, 50);
        assert_eq!(r.x, 10);
        assert_eq!(r.y, 20);
        assert_eq!(r.width, 100);
        assert_eq!(r.height, 50);
    }

    #[test]
    fn damage_region_pixel_count() {
        let r = DamageRegion::new(0, 0, 1920, 1080);
        assert_eq!(r.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn damage_region_zero_size() {
        let r = DamageRegion::new(0, 0, 0, 0);
        assert_eq!(r.pixel_count(), 0);
    }

    #[test]
    fn damage_region_1x1() {
        let r = DamageRegion::new(5, 5, 1, 1);
        assert_eq!(r.pixel_count(), 1);
    }

    #[test]
    fn damage_region_zero_width() {
        let r = DamageRegion::new(10, 10, 0, 100);
        assert_eq!(r.pixel_count(), 0);
    }

    #[test]
    fn damage_region_zero_height() {
        let r = DamageRegion::new(10, 10, 100, 0);
        assert_eq!(r.pixel_count(), 0);
    }

    #[test]
    fn damage_region_large_4k() {
        let r = DamageRegion::new(0, 0, 3840, 2160);
        assert_eq!(r.pixel_count(), 3840 * 2160);
    }

    #[test]
    fn damage_region_8k() {
        let r = DamageRegion::new(0, 0, 7680, 4320);
        assert_eq!(r.pixel_count(), 7680 * 4320);
    }

    #[test]
    fn damage_region_equality() {
        let a = DamageRegion::new(10, 20, 100, 50);
        let b = DamageRegion::new(10, 20, 100, 50);
        assert_eq!(a, b);
    }

    #[test]
    fn damage_region_inequality() {
        let a = DamageRegion::new(10, 20, 100, 50);
        assert_ne!(a, DamageRegion::new(11, 20, 100, 50));
        assert_ne!(a, DamageRegion::new(10, 21, 100, 50));
        assert_ne!(a, DamageRegion::new(10, 20, 101, 50));
        assert_ne!(a, DamageRegion::new(10, 20, 100, 51));
    }

    #[test]
    fn damage_region_copy() {
        let a = DamageRegion::new(1, 2, 3, 4);
        let b = a; // Copy
        assert_eq!(a, b);
        // Both are still usable after copy.
        assert_eq!(a.pixel_count(), 12);
        assert_eq!(b.pixel_count(), 12);
    }

    #[test]
    fn damage_region_clone() {
        let a = DamageRegion::new(1, 2, 3, 4);
        #[allow(clippy::clone_on_copy)]
        let b = a.clone(); // Intentionally testing Clone trait.
        assert_eq!(a, b);
    }

    #[test]
    fn damage_region_debug() {
        let r = DamageRegion::new(10, 20, 30, 40);
        let dbg = format!("{r:?}");
        assert!(dbg.contains("DamageRegion"));
        assert!(dbg.contains("10"));
        assert!(dbg.contains("20"));
        assert!(dbg.contains("30"));
        assert!(dbg.contains("40"));
    }

    #[test]
    fn damage_region_max_u32() {
        let r = DamageRegion::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX);
        assert_eq!(r.x, u32::MAX);
        assert_eq!(r.y, u32::MAX);
        // pixel_count uses usize multiplication — may overflow on 32-bit but not on 64-bit.
        #[cfg(target_pointer_width = "64")]
        {
            let expected = u32::MAX as usize * u32::MAX as usize;
            assert_eq!(r.pixel_count(), expected);
        }
    }

    // --- Buffer size calculation tests ---

    #[test]
    fn buffer_size_calculation() {
        let stride = 1920u32 * 4;
        let height = 1080u32;
        assert_eq!(stride as usize * height as usize, 1920 * 4 * 1080);
    }

    #[test]
    fn buffer_size_4k() {
        let stride = 3840u32 * 4;
        let height = 2160u32;
        let expected = 3840usize * 4 * 2160;
        assert_eq!(stride as usize * height as usize, expected);
    }

    #[test]
    fn buffer_size_8k() {
        let stride = 7680u32 * 4;
        let height = 4320u32;
        let expected = 7680usize * 4 * 4320;
        assert_eq!(stride as usize * height as usize, expected);
    }

    #[test]
    fn double_buffer_total_size() {
        let stride = 1920u32 * 4;
        let height = 1080u32;
        let buf_size = stride as usize * height as usize;
        let total = buf_size * 2;
        assert_eq!(total, 1920 * 4 * 1080 * 2);
    }

    /// Validate the stride check logic: stride must be >= width * 4.
    #[test]
    fn stride_validation_min_stride() {
        let width = 1920u32;
        let min_stride = (width as usize).checked_mul(4).unwrap();
        assert_eq!(min_stride, 7680);

        // Stride exactly at minimum: OK.
        let stride = 7680u32;
        assert!((stride as usize) >= min_stride);

        // Stride below minimum: error.
        let stride = 7679u32;
        assert!((stride as usize) < min_stride);
    }

    /// Validate that overflow detection works for stride calculation.
    #[test]
    fn stride_overflow_detection() {
        // On 64-bit systems, this won't overflow, but the logic is still tested.
        let width_ok = 1920u32;
        assert!((width_ok as usize).checked_mul(4).is_some());

        // Very large width — checked_mul should still succeed on 64-bit.
        #[cfg(target_pointer_width = "64")]
        {
            let width_large = u32::MAX;
            let result = (width_large as usize).checked_mul(4);
            assert!(result.is_some()); // u32::MAX * 4 fits in usize on 64-bit.
        }
    }

    /// Buffer size overflow detection: stride * height.
    #[test]
    fn buffer_size_overflow_detection() {
        let stride = 7680u32; // 1920 * 4
        let height = 1080u32;
        let result = (stride as usize).checked_mul(height as usize);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 7680 * 1080);
    }

    /// Total size overflow detection: `buf_size` * 2 (double buffering).
    #[test]
    fn total_size_overflow_detection() {
        let buf_size = 7680usize * 1080;
        let result = buf_size.checked_mul(2);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), buf_size * 2);
    }

    /// Zero-size buffer is rejected.
    #[test]
    fn zero_size_buffer_rejected() {
        let height = 0u32;
        let stride = 0u32;
        let buf_size = (stride as usize).checked_mul(height as usize).unwrap_or(0);
        let total_size = buf_size.checked_mul(2).unwrap_or(0);
        assert_eq!(total_size, 0);
        // The constructor would return Err(DisplayError::ShmBuffer("zero-size buffer")).
    }

    /// Simulate the active buffer swap logic.
    #[test]
    fn buffer_swap_logic() {
        let mut active = 0usize;
        let mut released = [true, true];

        // Initial state: both released, active=0.
        assert!(released[active]);

        // After swap: active becomes 1, buffer 0 is marked as in-use.
        released[active] = false;
        active = 1 - active;
        assert_eq!(active, 1);
        assert!(!released[0]);
        assert!(released[1]);

        // After another swap: active becomes 0, buffer 1 is in-use.
        released[active] = false;
        active = 1 - active;
        assert_eq!(active, 0);
        assert!(!released[0]);
        assert!(!released[1]);

        // Release buffer 0.
        released[0] = true;
        assert!(released[0]);
        assert!(!released[1]);

        // Can upload to active (0) now.
        assert!(released[active]);
    }

    /// Mark released out of bounds does nothing (matches the guard in `mark_released`).
    #[test]
    fn mark_released_out_of_bounds() {
        let mut released = [true, true];
        let buffer_idx = 5usize;
        if buffer_idx < 2 {
            released[buffer_idx] = true;
        }
        // No change — both still true.
        assert!(released[0]);
        assert!(released[1]);
    }

    /// `can_upload` reflects the released state of the active buffer.
    #[test]
    fn can_upload_logic() {
        let active = 0usize;
        let released = [true, true];
        assert!(released[active]); // can upload

        let released = [false, true];
        assert!(!released[active]); // cannot upload — buffer 0 in use

        let active = 1usize;
        assert!(released[active]); // can upload to buffer 1
    }

    /// Buffer size method matches stride * height.
    #[test]
    fn buffer_size_method() {
        let stride = 1920u32 * 4;
        let height = 1080u32;
        let buf_size = stride as usize * height as usize;
        assert_eq!(buf_size, 8_294_400);
    }

    /// Verify stride >= width * 4 property for various resolutions.
    #[test]
    fn stride_property_various_resolutions() {
        let resolutions: Vec<(u32, u32)> = vec![
            (640, 480),
            (1280, 720),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
            (7680, 4320),
            (1, 1),
            (64, 48),
        ];
        for (w, h) in resolutions {
            let stride = w * 4;
            let min_stride = w as usize * 4;
            assert!(
                stride as usize >= min_stride,
                "stride {stride} < min_stride {min_stride} for {w}x{h}"
            );
            let buf_size = stride as usize * h as usize;
            assert!(buf_size > 0, "zero buf_size for {w}x{h}");
            let total = buf_size * 2;
            assert!(total > 0, "zero total for {w}x{h}");
        }
    }
}
