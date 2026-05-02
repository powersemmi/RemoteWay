use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Monitor information sent during handshake (fixed-size header, 24 bytes).
///
/// Each connected output on the server is described by one of these structures.
/// The `stream_id` maps this monitor to frames in the pipeline.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct MonitorInfo {
    /// Stream ID used in `FrameHeader` for this monitor's frames.
    pub stream_id: u16,
    /// Horizontal position in the compositor coordinate space.
    pub x: i16,
    /// Vertical position in the compositor coordinate space.
    pub y: i16,
    /// Alignment padding; always zero.
    pub _pad: u16,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Refresh rate in millihertz (e.g. 60000 for 60 Hz).
    pub refresh_mhz: u32,
    /// Integer scale factor (1 or 2 typically).
    pub scale: u32,
}

impl MonitorInfo {
    /// Size of `MonitorInfo` in bytes (always 24).
    pub const SIZE: usize = size_of::<Self>();

    /// Create a new `MonitorInfo` descriptor.
    #[must_use]
    pub fn new(
        stream_id: u16,
        x: i16,
        y: i16,
        width: u32,
        height: u32,
        refresh_mhz: u32,
        scale: u32,
    ) -> Self {
        Self {
            stream_id,
            x,
            y,
            _pad: 0,
            width,
            height,
            refresh_mhz,
            scale,
        }
    }
}

const _: () = assert!(size_of::<MonitorInfo>() == 24);

/// Fractional scale descriptor (4 bytes).
///
/// Sent as part of the handshake when the server detects `wp-fractional-scale-v1`.
/// The scale is in 1/120 units (e.g. 120 = 1.0x, 180 = 1.5x, 240 = 2.0x).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct FractionalScale {
    /// Stream ID this scale applies to.
    pub stream_id: u16,
    /// Scale in 1/120 units. 120 = 1.0x, 180 = 1.5x, 240 = 2.0x.
    pub scale_120: u16,
}

impl FractionalScale {
    /// Size of `FractionalScale` in bytes (always 4).
    pub const SIZE: usize = size_of::<Self>();

    /// Create a new `FractionalScale` descriptor.
    #[must_use]
    pub fn new(stream_id: u16, scale_120: u16) -> Self {
        Self {
            stream_id,
            scale_120,
        }
    }

    /// Get the floating-point scale value.
    #[must_use]
    pub fn scale_f64(&self) -> f64 {
        self.scale_120 as f64 / 120.0
    }
}

const _: () = assert!(size_of::<FractionalScale>() == 4);

#[cfg(test)]
mod tests {
    use zerocopy::{FromBytes, IntoBytes};

    use super::*;

    #[test]
    fn monitor_info_size() {
        assert_eq!(MonitorInfo::SIZE, 24);
    }

    #[test]
    fn monitor_info_round_trip() {
        let m = MonitorInfo::new(1, 0, 0, 1920, 1080, 60000, 1);
        let bytes = m.as_bytes();
        let decoded = MonitorInfo::ref_from_bytes(bytes).unwrap();
        assert_eq!({ decoded.stream_id }, 1);
        assert_eq!({ decoded.width }, 1920);
        assert_eq!({ decoded.height }, 1080);
        assert_eq!({ decoded.refresh_mhz }, 60000);
        assert_eq!({ decoded.scale }, 1);
    }

    #[test]
    fn fractional_scale_size() {
        assert_eq!(FractionalScale::SIZE, 4);
    }

    #[test]
    fn fractional_scale_values() {
        let s1 = FractionalScale::new(1, 120);
        assert!((s1.scale_f64() - 1.0).abs() < f64::EPSILON);

        let s15 = FractionalScale::new(1, 180);
        assert!((s15.scale_f64() - 1.5).abs() < f64::EPSILON);

        let s2 = FractionalScale::new(1, 240);
        assert!((s2.scale_f64() - 2.0).abs() < f64::EPSILON);
    }
}
