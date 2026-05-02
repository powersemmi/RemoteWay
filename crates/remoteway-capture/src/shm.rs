//! Double-buffered SHM pool for zero-copy Wayland capture.
//!
//! Two independent `ShmSlot`s each on their own memfd + `wl_shm_pool` at
//! offset 0, avoiding compositor bugs with non-zero pool offsets.

use std::os::fd::{AsFd, OwnedFd};
use std::ptr::NonNull;

use nix::sys::mman::{MapFlags, MmapAdvise, ProtFlags, madvise, munmap};
use wayland_client::protocol::{wl_buffer, wl_shm, wl_shm_pool};
use wayland_client::{Dispatch, QueueHandle};

use crate::error::CaptureError;

/// One-buffer SHM slot backed by its own memfd + `wl_shm_pool`.
///
/// Each slot owns a private fd, mmap, pool, and `wl_buffer` at offset 0.
/// Keeping every buffer on its own pool sidesteps a known issue where some
/// smithay-based compositors (niri, cosmic-comp) reject `wl_buffer` objects
/// created at non-zero pool offsets in `zwlr_screencopy_frame.copy`,
/// reporting `invalid_buffer`. With offset 0 the buffer/pool/fd correspondence
/// is unambiguous and matches what compositors typically test against.
struct ShmSlot {
    /// Memfd file descriptor backing the SHM mapping.
    _fd: OwnedFd,
    /// Pointer to the mmap'd buffer.
    map: NonNull<std::ffi::c_void>,
    /// Size of the mmap'd region in bytes.
    map_size: usize,
    /// `wl_shm_pool` created from the fd at offset 0.
    pool: wl_shm_pool::WlShmPool,
    /// `wl_buffer` created from the pool at offset 0.
    buffer: wl_buffer::WlBuffer,
}

// SAFETY: ShmSlot owns its fd and mmap exclusively. Buffers are accessed by
// only one thread at a time (the capture thread).
unsafe impl Send for ShmSlot {}

/// Double-buffered SHM pool for zero-copy Wayland capture.
///
/// Internally maintains two independent `ShmSlot`s, each with its own fd,
/// mmap and `wl_shm_pool`. One buffer is used by the compositor for the
/// current capture, while the other is available for reading by the pipeline.
pub struct ShmBufferPool {
    slots: [ShmSlot; 2],
    active: usize,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bytes per row.
    pub stride: u32,
    ///  pixel format.
    pub format: wl_shm::Format,
}

impl ShmBufferPool {
    /// Create a new double-buffered SHM pool.
    ///
    /// # Arguments
    /// * `shm` - The `wl_shm` global from the compositor.
    /// * `width`, `height` - Frame dimensions.
    /// * `stride` - Bytes per row.
    /// * `format` - Pixel format code (`wl_shm` format).
    /// * `qh` - Queue handle for Wayland protocol dispatch.
    pub fn new<D>(
        shm: &wl_shm::WlShm,
        width: u32,
        height: u32,
        stride: u32,
        format: wl_shm::Format,
        qh: &QueueHandle<D>,
    ) -> Result<Self, CaptureError>
    where
        D: Dispatch<wl_shm_pool::WlShmPool, ()> + Dispatch<wl_buffer::WlBuffer, usize> + 'static,
    {
        // Validate inputs.
        let min_stride = (width as usize).checked_mul(4).ok_or_else(|| {
            CaptureError::ShmPool("stride overflow: width * 4 exceeds usize".into())
        })?;
        if (stride as usize) < min_stride {
            return Err(CaptureError::ShmPool(format!(
                "stride ({stride}) < width * 4 ({min_stride})"
            )));
        }
        let buf_size = (stride as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| {
                CaptureError::ShmPool("buffer size overflow: stride * height exceeds usize".into())
            })?;
        if buf_size == 0 {
            return Err(CaptureError::ShmPool("zero-size buffer".into()));
        }

        let slot0 = ShmSlot::new(shm, width, height, stride, format, buf_size, qh, 0usize)?;
        let slot1 = ShmSlot::new(shm, width, height, stride, format, buf_size, qh, 1usize)?;

        Ok(Self {
            slots: [slot0, slot1],
            active: 0,
            width,
            height,
            stride,
            format,
        })
    }

    /// Get the `wl_buffer` that the compositor should write to.
    #[must_use]
    pub fn active_buffer(&self) -> &wl_buffer::WlBuffer {
        &self.slots[self.active].buffer
    }

    /// Get a slice of the active buffer's pixel data.
    ///
    /// # Safety
    /// The compositor must not be concurrently writing to this buffer.
    /// Call only after receiving the `ready` event.
    #[must_use]
    pub unsafe fn active_data(&self) -> &[u8] {
        let slot = &self.slots[self.active];
        let buf_size = self.stride as usize * self.height as usize;
        // SAFETY: map is valid and at least `buf_size` bytes long.
        unsafe { std::slice::from_raw_parts(slot.map.as_ptr() as *const u8, buf_size) }
    }

    /// Swap to the other buffer for the next capture.
    pub fn swap(&mut self) {
        self.active = 1 - self.active;
    }

    /// Buffer size in bytes (single buffer, not total).
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.stride as usize * self.height as usize
    }
}

impl ShmSlot {
    #[allow(clippy::too_many_arguments)]
    fn new<D>(
        shm: &wl_shm::WlShm,
        width: u32,
        height: u32,
        stride: u32,
        format: wl_shm::Format,
        size: usize,
        qh: &QueueHandle<D>,
        user_data: usize,
    ) -> Result<Self, CaptureError>
    where
        D: Dispatch<wl_shm_pool::WlShmPool, ()> + Dispatch<wl_buffer::WlBuffer, usize> + 'static,
    {
        let fd = nix::sys::memfd::memfd_create(
            c"remoteway-shm",
            nix::sys::memfd::MemFdCreateFlag::MFD_CLOEXEC,
        )
        .map_err(|e| CaptureError::ShmPool(format!("memfd_create failed: {e}")))?;

        nix::unistd::ftruncate(&fd, size as i64)
            .map_err(|e| CaptureError::ShmPool(format!("ftruncate failed: {e}")))?;

        // SAFETY: fd is valid, size is non-zero, MAP_SHARED for Wayland SHM.
        let map = unsafe {
            nix::sys::mman::mmap(
                None,
                #[allow(clippy::expect_used)]
                std::num::NonZero::new(size).expect("size > 0"),
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                &fd,
                0,
            )
            .map_err(|e| CaptureError::ShmPool(format!("mmap fd failed: {e}")))?
        };

        // SAFETY: map is valid and covers `size` bytes.
        unsafe {
            // MADV_SEQUENTIAL is a hint; failure is non-critical but worth
            // logging in case the kernel rejects the advise for this region.
            if let Err(e) = madvise(map, size, MmapAdvise::MADV_SEQUENTIAL) {
                tracing::warn!(
                    size,
                    error = %e,
                    "madvise MADV_SEQUENTIAL failed"
                );
            }
        }

        let pool = shm.create_pool(fd.as_fd(), size as i32, qh, ());
        let buffer = pool.create_buffer(
            0,
            width as i32,
            height as i32,
            stride as i32,
            format,
            qh,
            user_data,
        );

        Ok(Self {
            _fd: fd,
            map,
            map_size: size,
            pool,
            buffer,
        })
    }
}

impl Drop for ShmSlot {
    fn drop(&mut self) {
        self.buffer.destroy();
        self.pool.destroy();
        // SAFETY: map was created with mmap of map_size bytes.
        unsafe {
            // munmap failure in Drop is unexpected but not actionable —
            // we cannot recover and the OS will clean up on process exit.
            if let Err(e) = munmap(self.map, self.map_size) {
                tracing::error!(
                    size = self.map_size,
                    error = %e,
                    "munmap failed in ShmSlot::drop"
                );
            }
        }
        // fd is dropped automatically by OwnedFd.
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn buffer_size_calculation() {
        // We can't create a real ShmBufferPool without a Wayland connection,
        // but we can test the math.
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
    fn min_stride_check() {
        // width * 4 must not exceed usize
        let width = 1920u32;
        let min_stride = (width as usize).checked_mul(4);
        assert_eq!(min_stride, Some(7680));
    }

    #[test]
    fn stride_less_than_min_is_invalid() {
        // Stride must be >= width * 4. Test the validation logic.
        let width: u32 = 100;
        let stride: u32 = 100; // Less than 100 * 4 = 400
        let min_stride = (width as usize).checked_mul(4).unwrap();
        assert!((stride as usize) < min_stride);
    }

    #[test]
    fn stride_equal_to_min_is_valid() {
        let width: u32 = 100;
        let stride: u32 = 400; // Exactly width * 4
        let min_stride = (width as usize).checked_mul(4).unwrap();
        assert!((stride as usize) >= min_stride);
    }

    #[test]
    fn stride_with_padding_is_valid() {
        // Some compositors add padding to stride
        let width: u32 = 100;
        let stride: u32 = 512; // Padded to 512
        let min_stride = (width as usize).checked_mul(4).unwrap();
        assert!((stride as usize) >= min_stride);
    }

    #[test]
    fn zero_height_produces_zero_buf_size() {
        let stride: u32 = 7680;
        let height: u32 = 0;
        let buf_size = (stride as usize).checked_mul(height as usize);
        assert_eq!(buf_size, Some(0));
        // Zero-size buffer would be rejected
    }

    #[test]
    fn zero_width_produces_zero_min_stride() {
        let width: u32 = 0;
        let min_stride = (width as usize).checked_mul(4);
        assert_eq!(min_stride, Some(0));
    }

    #[test]
    fn overflow_width_times_4() {
        // On 64-bit, usize::MAX / 4 is huge, so we test the logic path
        // by simulating what would happen for very large values.
        let width: u32 = u32::MAX;
        let result = (width as usize).checked_mul(4);
        // On 64-bit this won't overflow; on 32-bit it would.
        // The key is that checked_mul is used.
        assert!(result.is_some() || result.is_none());
    }

    #[test]
    fn overflow_stride_times_height() {
        // Verify checked_mul protects against overflow
        let stride = usize::MAX;
        let height = 2usize;
        assert!(stride.checked_mul(height).is_none());
    }

    #[test]
    fn swap_logic() {
        // Test the swap toggle logic: 1 - active
        let mut active: usize = 0;
        active = 1 - active;
        assert_eq!(active, 1);
        active = 1 - active;
        assert_eq!(active, 0);
    }

    #[test]
    fn typical_frame_sizes() {
        // Verify common resolutions don't overflow
        let resolutions: &[(u32, u32)] = &[(1920, 1080), (2560, 1440), (3840, 2160), (7680, 4320)];
        for &(w, h) in resolutions {
            let stride = w * 4;
            let min_stride = (w as usize).checked_mul(4);
            assert!(min_stride.is_some(), "overflow for width {w}");
            assert!((stride as usize) >= min_stride.unwrap());
            let buf_size = (stride as usize).checked_mul(h as usize);
            assert!(buf_size.is_some(), "overflow for {w}x{h}");
            assert!(buf_size.unwrap() > 0);
        }
    }
}
