//! Backend detection and enumeration.
//!
//! Defines `BackendKind` variants and `BackendDetector` for probing available
//! interpolation backends at runtime.

use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, LinearBlendInterpolator};

/// Available interpolation backend types, ordered by quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Per-pixel linear blend (CPU, universal fallback).
    LinearBlend,
    /// AMD FSR 2.x temporal interpolation (requires `fsr2` feature + Vulkan).
    Fsr2,
    /// AMD FSR 3 Frame Generation + FSR SDK upscale (requires `fsr3` feature).
    Fsr3,
    /// FSR2 upscaling + RIFE neural interpolation (requires `fsr2-rife` feature).
    Fsr2Rife,
}

impl BackendKind {
    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::LinearBlend => "linear-blend",
            Self::Fsr2 => "fsr2",
            Self::Fsr3 => "fsr3",
            Self::Fsr2Rife => "fsr2-rife",
        }
    }
}

impl std::str::FromStr for BackendKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "linear-blend" => Ok(Self::LinearBlend),
            "fsr2" => Ok(Self::Fsr2),
            "fsr3" => Ok(Self::Fsr3),
            "fsr2-rife" => Ok(Self::Fsr2Rife),
            other => Err(format!(
                "unknown backend '{other}'. Available: fsr3, fsr2, fsr2-rife, linear-blend"
            )),
        }
    }
}

/// Detects available interpolation backends and selects the best one.
pub struct BackendDetector;

impl BackendDetector {
    /// Detect all available backends on this system.
    ///
    /// Returns backends sorted by quality: fsr3 → fsr2, then CPU fallback (linear-blend).
    /// fsr2-rife is not auto-detected.
    #[must_use]
    pub fn detect_available() -> Vec<BackendKind> {
        let mut backends = Vec::new();

        // 1. AMD FSR 3 — Frame Generation + FSR SDK upscale
        if Self::check_fsr3() {
            backends.push(BackendKind::Fsr3);
        }

        // 2. AMD FSR 2 — Vulkan temporal interpolation
        if Self::check_fsr2() {
            backends.push(BackendKind::Fsr2);
        }

        // Last: CPU fallback (always available).
        backends.push(BackendKind::LinearBlend);

        backends
    }

    /// Select the best available backend and create an interpolator.
    pub fn select_best() -> Result<Box<dyn FrameInterpolator>, InterpolateError> {
        let available = Self::detect_available();
        Self::create_backend(
            available
                .first()
                .copied()
                .unwrap_or(BackendKind::LinearBlend),
        )
    }

    /// Create an interpolator for a specific backend kind.
    pub fn create_backend(
        kind: BackendKind,
    ) -> Result<Box<dyn FrameInterpolator>, InterpolateError> {
        match kind {
            BackendKind::LinearBlend => Ok(Box::new(LinearBlendInterpolator)),
            #[cfg(feature = "fsr2")]
            BackendKind::Fsr2 => Ok(Box::new(crate::backends::fsr2::Fsr2Interpolator::new()?)),
            #[cfg(not(feature = "fsr2"))]
            BackendKind::Fsr2 => Err(InterpolateError::InitFailed(
                "fsr2 feature not enabled".into(),
            )),
            #[cfg(feature = "fsr3")]
            BackendKind::Fsr3 => {
                match crate::backends::fsr3::Fsr3Interpolator::new(
                    1280, 720, 1920, 1080,
                ) {
                    Ok(backend) => Ok(Box::new(backend)),
                    #[cfg(feature = "fsr2")]
                    Err(e) => {
                        eprintln!("warn: fsr3 init failed ({}), falling back to fsr2", e);
                        Ok(Box::new(crate::backends::fsr2::Fsr2Interpolator::new()?))
                    }
                    #[cfg(not(feature = "fsr2"))]
                    Err(e) => Err(e),
                }
            }
            #[cfg(not(feature = "fsr3"))]
            BackendKind::Fsr3 => Err(InterpolateError::InitFailed(
                "fsr3 feature not enabled".into(),
            )),
            #[cfg(feature = "fsr2-rife")]
            BackendKind::Fsr2Rife => Ok(Box::new(
                crate::backends::fsr2_rife::Fsr2RifeInterpolator::new()?,
            )),
            #[cfg(not(feature = "fsr2-rife"))]
            BackendKind::Fsr2Rife => Err(InterpolateError::InitFailed(
                "fsr2-rife feature not enabled".into(),
            )),
        }
    }

    fn check_fsr3() -> bool {
        #[cfg(feature = "fsr3")]
        {
            crate::backends::vulkan_context::VulkanContext::is_vulkan_available()
        }
        #[cfg(not(feature = "fsr3"))]
        false
    }

    fn check_fsr2() -> bool {
        #[cfg(feature = "fsr2")]
        {
            crate::backends::vulkan_context::VulkanContext::is_vulkan_available()
        }
        #[cfg(not(feature = "fsr2"))]
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kind_name() {
        assert_eq!(BackendKind::LinearBlend.name(), "linear-blend");
        assert_eq!(BackendKind::Fsr2.name(), "fsr2");
        assert_eq!(BackendKind::Fsr3.name(), "fsr3");
        assert_eq!(BackendKind::Fsr2Rife.name(), "fsr2-rife");
    }

    #[test]
    fn backend_kind_eq() {
        assert_eq!(BackendKind::LinearBlend, BackendKind::LinearBlend);
        assert_ne!(BackendKind::LinearBlend, BackendKind::Fsr2);
    }

    #[test]
    fn detect_available_always_includes_linear_blend() {
        let available = BackendDetector::detect_available();
        assert!(!available.is_empty());
        assert_eq!(*available.last().unwrap(), BackendKind::LinearBlend);
    }

    #[test]
    fn select_best_succeeds() {
        let interp = BackendDetector::select_best().unwrap();
        // Without GPU features: always linear-blend.
        // With GPU features on a system with GPU: may return a GPU backend.
        assert!(!interp.name().is_empty());
    }

    #[test]
    fn create_linear_blend() {
        let interp = BackendDetector::create_backend(BackendKind::LinearBlend).unwrap();
        assert_eq!(interp.name(), "linear-blend");
    }

    #[test]
    fn create_disabled_feature_backends() {
        // Without features enabled, GPU backends return feature-not-enabled errors.
        #[cfg(not(feature = "fsr2"))]
        assert!(BackendDetector::create_backend(BackendKind::Fsr2).is_err());
        #[cfg(not(feature = "fsr3"))]
        assert!(BackendDetector::create_backend(BackendKind::Fsr3).is_err());
        #[cfg(not(feature = "fsr2-rife"))]
        assert!(BackendDetector::create_backend(BackendKind::Fsr2Rife).is_err());
    }

    #[test]
    fn backend_kind_clone_copy() {
        let a = BackendKind::Fsr2;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn backend_kind_debug() {
        let dbg = format!("{:?}", BackendKind::Fsr3);
        assert!(dbg.contains("Fsr3"));
    }

    #[test]
    fn from_str_all_valid_names() {
        assert_eq!(
            "linear-blend".parse::<BackendKind>().unwrap(),
            BackendKind::LinearBlend
        );
        assert_eq!("fsr2".parse::<BackendKind>().unwrap(), BackendKind::Fsr2);
        assert_eq!("fsr3".parse::<BackendKind>().unwrap(), BackendKind::Fsr3);
        assert_eq!(
            "fsr2-rife".parse::<BackendKind>().unwrap(),
            BackendKind::Fsr2Rife
        );
    }

    #[test]
    fn from_str_unknown_returns_error() {
        let err = "unknown-backend".parse::<BackendKind>().unwrap_err();
        assert!(err.contains("unknown backend"));
    }
}
