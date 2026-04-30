/// Errors from interpolation backends.
#[derive(Debug, thiserror::Error)]
pub enum InterpolateError {
    #[error("no interpolation backend available")]
    NoBackend,
    #[error("GPU device lost")]
    DeviceLost,
    #[error("frame dimensions mismatch: {0}x{1} vs {2}x{3}")]
    DimensionMismatch(u32, u32, u32, u32),
    #[error("invalid interpolation factor: {0} (must be 0.0..=1.0)")]
    InvalidFactor(f32),
    #[error("backend initialization failed: {0}")]
    InitFailed(String),
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
