use remoteway_compress::delta::DamageRect;

use crate::error::CaptureError;

/// Pixel format of captured frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Argb8888,
    Xrgb8888,
    Abgr8888,
    Xbgr8888,
}

impl PixelFormat {
    /// Bytes per pixel — always 4 for supported formats.
    pub const fn bytes_per_pixel(&self) -> usize {
        4
    }

    /// Convert from wl_shm format code (DRM fourcc values).
    pub fn from_wl_shm(format: u32) -> Option<Self> {
        match format {
            0 => Some(Self::Argb8888),
            1 => Some(Self::Xrgb8888),
            0x34324241 => Some(Self::Abgr8888),
            0x34324258 => Some(Self::Xbgr8888),
            _ => None,
        }
    }
}

/// A captured frame with pixel data and damage information.
#[derive(Debug)]
#[must_use]
pub struct CapturedFrame {
    /// Raw pixel data (owned, copied from SHM buffer).
    pub data: Vec<u8>,
    /// Regions that changed since the previous frame.
    pub damage: Vec<DamageRect>,
    /// Pixel format.
    pub format: PixelFormat,
    /// Frame dimensions.
    pub width: u32,
    pub height: u32,
    /// Bytes per row (may be > width * 4 due to padding).
    pub stride: u32,
    /// Compositor timestamp in nanoseconds.
    pub timestamp_ns: u64,
}

/// Abstraction over different Wayland capture protocols.
///
/// Implementors capture screen content and report damage regions.
pub trait CaptureBackend: Send {
    /// Block until the next frame is available, then return it.
    fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError>;

    /// Human-readable backend name.
    fn name(&self) -> &'static str;

    /// Signal the backend to stop capturing.
    fn stop(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_format_bytes_per_pixel() {
        assert_eq!(PixelFormat::Argb8888.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Xrgb8888.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Abgr8888.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Xbgr8888.bytes_per_pixel(), 4);
    }

    #[test]
    fn pixel_format_from_wl_shm() {
        assert_eq!(PixelFormat::from_wl_shm(0), Some(PixelFormat::Argb8888));
        assert_eq!(PixelFormat::from_wl_shm(1), Some(PixelFormat::Xrgb8888));
        assert_eq!(
            PixelFormat::from_wl_shm(0x34324241),
            Some(PixelFormat::Abgr8888)
        );
        assert_eq!(
            PixelFormat::from_wl_shm(0x34324258),
            Some(PixelFormat::Xbgr8888)
        );
        assert_eq!(PixelFormat::from_wl_shm(999), None);
    }

    #[test]
    fn pixel_format_debug_clone_copy_eq() {
        let f = PixelFormat::Xrgb8888;
        let f2 = f; // Copy
        assert_eq!(f, f2);
        let f3: PixelFormat = f;
        assert_eq!(f, f3);
        assert_ne!(PixelFormat::Argb8888, PixelFormat::Xrgb8888);
        let dbg = format!("{:?}", f);
        assert!(dbg.contains("Xrgb8888"));
    }

    #[test]
    fn captured_frame_construction() {
        let frame = CapturedFrame {
            data: vec![0u8; 16],
            damage: vec![DamageRect::new(0, 0, 2, 2)],
            format: PixelFormat::Xrgb8888,
            width: 2,
            height: 2,
            stride: 8,
            timestamp_ns: 12345,
        };
        assert_eq!(frame.width, 2);
        assert_eq!(frame.damage.len(), 1);
        assert_eq!(frame.data.len(), 16);
    }

    #[test]
    fn capture_backend_is_send() {
        fn assert_send<T: Send>() {}
        // Trait requires Send; this compiles only if CaptureBackend: Send.
        assert_send::<Box<dyn CaptureBackend>>();
    }
}
