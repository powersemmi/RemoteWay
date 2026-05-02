//! Error types for frame interpolation.

/// Errors from interpolation backends.
#[derive(Debug, thiserror::Error)]
pub enum InterpolateError {
    /// No interpolation backend is available on this system.
    #[error("no interpolation backend available")]
    NoBackend,
    /// The GPU device was lost (driver crash, hot-unplug, etc.).
    #[error("GPU device lost")]
    DeviceLost,
    /// The two input frames have different dimensions.
    /// Fields: `a.width`, `a.height`, `b.width`, `b.height`.
    #[error("frame dimensions mismatch: {0}x{1} vs {2}x{3}")]
    DimensionMismatch(u32, u32, u32, u32),
    /// The interpolation factor `t` is outside the valid range `0.0..=1.0`.
    #[error("invalid interpolation factor: {0} (must be 0.0..=1.0)")]
    InvalidFactor(f32),
    /// Backend-specific initialization failure.
    #[error("backend initialization failed: {0}")]
    InitFailed(String),
    /// Backend-specific interpolation failure.
    #[error("interpolation failed: {0}")]
    InterpolateFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_all_variants() {
        let errors: Vec<InterpolateError> = vec![
            InterpolateError::NoBackend,
            InterpolateError::DeviceLost,
            InterpolateError::DimensionMismatch(1920, 1080, 3840, 2160),
            InterpolateError::InvalidFactor(1.5),
            InterpolateError::InitFailed("test".into()),
            InterpolateError::InterpolateFailed("test".into()),
        ];
        for e in &errors {
            let s = format!("{e}");
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn error_debug() {
        let e = InterpolateError::DimensionMismatch(100, 200, 300, 400);
        let dbg = format!("{e:?}");
        assert!(dbg.contains("DimensionMismatch"));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InterpolateError>();
    }
}
