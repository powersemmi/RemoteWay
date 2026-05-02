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
    /// Optical flow via wgpu compute shaders (requires `wgpu-backend` feature).
    WgpuOpticalFlow,
    /// AMD FSR 2.x temporal interpolation (requires `fsr2` feature + Vulkan).
    Fsr2,
    /// NVIDIA optical flow via `VK_NV_optical_flow` (requires `nvidia-of` feature + Turing+ GPU).
    NvidiaOpticalFlow,
    /// AMD FSR 3 with hardware optical flow (requires `fsr3` feature + RDNA3+).
    Fsr3Hardware,
    /// RIFE neural frame interpolation (requires `rife` feature).
    Rife,
}

impl BackendKind {
    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::LinearBlend => "linear-blend",
            Self::WgpuOpticalFlow => "wgpu-optical-flow",
            Self::Fsr2 => "fsr2",
            Self::NvidiaOpticalFlow => "nvidia-optical-flow",
            Self::Fsr3Hardware => "fsr3-hardware",
            Self::Rife => "rife",
        }
    }
}

impl std::str::FromStr for BackendKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "linear-blend" => Ok(Self::LinearBlend),
            "wgpu-optical-flow" => Ok(Self::WgpuOpticalFlow),
            "fsr2" => Ok(Self::Fsr2),
            "nvidia-optical-flow" => Ok(Self::NvidiaOpticalFlow),
            "fsr3-hardware" | "fsr3" => Ok(Self::Fsr3Hardware),
            "rife" => Ok(Self::Rife),
            other => Err(format!(
                "unknown backend '{other}'. Available: linear-blend, wgpu-optical-flow, \
                 fsr2, nvidia-optical-flow, fsr3-hardware, rife"
            )),
        }
    }
}

/// Detects available interpolation backends and selects the best one.
pub struct BackendDetector;

impl BackendDetector {
    /// Detect all available backends on this system.
    ///
    /// Returns backends sorted by quality (best first).
    #[must_use]
    pub fn detect_available() -> Vec<BackendKind> {
        let mut backends = Vec::new();

        // Check GPU backends (feature-gated).
        if Self::check_nvidia_of() {
            backends.push(BackendKind::NvidiaOpticalFlow);
        }
        if Self::check_fsr3() {
            backends.push(BackendKind::Fsr3Hardware);
        }
        if Self::check_fsr2() {
            backends.push(BackendKind::Fsr2);
        }
        if Self::check_wgpu() {
            backends.push(BackendKind::WgpuOpticalFlow);
        }
        if Self::check_rife() {
            backends.push(BackendKind::Rife);
        }

        // CPU fallback is always available.
        backends.push(BackendKind::LinearBlend);

        backends
    }

    /// Select the best available backend and create an interpolator.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError::InitFailed`] if no backend can be
    /// initialized (e.g., Vulkan unavailable, GPU not compatible).
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
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError::InitFailed`] if the requested backend
    /// cannot be initialized (feature not enabled, driver missing, etc.).
    pub fn create_backend(
        kind: BackendKind,
    ) -> Result<Box<dyn FrameInterpolator>, InterpolateError> {
        match kind {
            BackendKind::LinearBlend => Ok(Box::new(LinearBlendInterpolator)),
            #[cfg(feature = "wgpu-backend")]
            BackendKind::WgpuOpticalFlow => {
                let interp =
                    crate::backends::wgpu_optical_flow::WgpuOpticalFlowInterpolator::new()?;
                Ok(Box::new(interp))
            }
            #[cfg(not(feature = "wgpu-backend"))]
            BackendKind::WgpuOpticalFlow => Err(InterpolateError::InitFailed(
                "wgpu-backend feature not enabled".into(),
            )),
            #[cfg(feature = "fsr2")]
            BackendKind::Fsr2 => {
                let interp = crate::backends::fsr2::Fsr2Interpolator::new()?;
                Ok(Box::new(interp))
            }
            #[cfg(not(feature = "fsr2"))]
            BackendKind::Fsr2 => Err(InterpolateError::InitFailed(
                "fsr2 feature not enabled".into(),
            )),
            #[cfg(feature = "nvidia-of")]
            BackendKind::NvidiaOpticalFlow => {
                let interp =
                    crate::backends::nvidia_optical_flow::NvidiaOpticalFlowInterpolator::new()?;
                Ok(Box::new(interp))
            }
            #[cfg(not(feature = "nvidia-of"))]
            BackendKind::NvidiaOpticalFlow => Err(InterpolateError::InitFailed(
                "nvidia-of feature not enabled".into(),
            )),
            #[cfg(feature = "fsr3")]
            BackendKind::Fsr3Hardware => {
                let interp = crate::backends::fsr3::Fsr3Interpolator::new()?;
                Ok(Box::new(interp))
            }
            #[cfg(not(feature = "fsr3"))]
            BackendKind::Fsr3Hardware => Err(InterpolateError::InitFailed(
                "fsr3 feature not enabled".into(),
            )),
            #[cfg(feature = "rife")]
            BackendKind::Rife => {
                let interp = crate::backends::rife::RifeInterpolator::from_default_path()?;
                Ok(Box::new(interp))
            }
            #[cfg(not(feature = "rife"))]
            BackendKind::Rife => Err(InterpolateError::InitFailed(
                "rife feature not enabled".into(),
            )),
        }
    }

    fn check_nvidia_of() -> bool {
        #[cfg(feature = "nvidia-of")]
        {
            crate::backends::vulkan_context::VulkanContext::probe_nvidia_optical_flow()
        }
        #[cfg(not(feature = "nvidia-of"))]
        false
    }

    fn check_fsr3() -> bool {
        #[cfg(feature = "fsr3")]
        {
            crate::backends::vulkan_context::VulkanContext::probe_rdna3_plus()
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

    fn check_wgpu() -> bool {
        #[cfg(feature = "wgpu-backend")]
        {
            crate::backends::wgpu_optical_flow::is_available()
        }
        #[cfg(not(feature = "wgpu-backend"))]
        false
    }

    fn check_rife() -> bool {
        #[cfg(feature = "rife")]
        {
            crate::backends::rife::RifeInterpolator::from_default_path().is_ok()
        }
        #[cfg(not(feature = "rife"))]
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kind_name() {
        assert_eq!(BackendKind::LinearBlend.name(), "linear-blend");
        assert_eq!(BackendKind::WgpuOpticalFlow.name(), "wgpu-optical-flow");
        assert_eq!(BackendKind::Fsr2.name(), "fsr2");
        assert_eq!(BackendKind::NvidiaOpticalFlow.name(), "nvidia-optical-flow");
        assert_eq!(BackendKind::Fsr3Hardware.name(), "fsr3-hardware");
        assert_eq!(BackendKind::Rife.name(), "rife");
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
        // With features enabled, these may succeed on systems with GPUs.
        #[cfg(not(feature = "wgpu-backend"))]
        assert!(BackendDetector::create_backend(BackendKind::WgpuOpticalFlow).is_err());
        #[cfg(not(feature = "fsr2"))]
        assert!(BackendDetector::create_backend(BackendKind::Fsr2).is_err());
        #[cfg(not(feature = "nvidia-of"))]
        assert!(BackendDetector::create_backend(BackendKind::NvidiaOpticalFlow).is_err());
        #[cfg(not(feature = "fsr3"))]
        assert!(BackendDetector::create_backend(BackendKind::Fsr3Hardware).is_err());
        #[cfg(not(feature = "rife"))]
        assert!(BackendDetector::create_backend(BackendKind::Rife).is_err());
    }

    #[test]
    fn backend_kind_clone_copy() {
        let a = BackendKind::Fsr2;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn backend_kind_debug() {
        let dbg = format!("{:?}", BackendKind::NvidiaOpticalFlow);
        assert!(dbg.contains("NvidiaOpticalFlow"));
    }

    #[test]
    fn from_str_all_valid_names() {
        assert_eq!(
            "linear-blend".parse::<BackendKind>().unwrap(),
            BackendKind::LinearBlend
        );
        assert_eq!(
            "wgpu-optical-flow".parse::<BackendKind>().unwrap(),
            BackendKind::WgpuOpticalFlow
        );
        assert_eq!("fsr2".parse::<BackendKind>().unwrap(), BackendKind::Fsr2);
        assert_eq!(
            "nvidia-optical-flow".parse::<BackendKind>().unwrap(),
            BackendKind::NvidiaOpticalFlow
        );
        assert_eq!(
            "fsr3-hardware".parse::<BackendKind>().unwrap(),
            BackendKind::Fsr3Hardware
        );
        assert_eq!(
            "fsr3".parse::<BackendKind>().unwrap(),
            BackendKind::Fsr3Hardware
        );
        assert_eq!("rife".parse::<BackendKind>().unwrap(), BackendKind::Rife);
    }

    #[test]
    fn from_str_unknown_returns_error() {
        let err = "unknown-backend".parse::<BackendKind>().unwrap_err();
        assert!(err.contains("unknown backend"));
    }
}
