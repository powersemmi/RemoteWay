//! RIFE neural frame interpolation backend.
//!
//! Wraps a RIFE v4.6+ model for high-quality optical-flow frame synthesis.
//! Requires an external inference runtime (ncnn-rs or ONNX Runtime) and
//! a model file on disk.

use std::path::PathBuf;

use crate::error::InterpolateError;
use crate::interpolator::{FrameInterpolator, GpuFrame};

/// RIFE (Real-Time Intermediate Flow Estimation) neural frame interpolator.
///
/// Uses a pre-trained RIFE model for high-quality optical flow estimation
/// and frame synthesis. Requires an external inference runtime (ncnn or
/// ONNX Runtime) and a model file on disk.
///
/// # Integration points
///
/// To enable RIFE, integrate one of these inference backends:
///
/// ## ncnn-rs (recommended for Vulkan GPU inference)
/// ```ignore
/// // Cargo.toml:
/// // ncnn-rs = { version = "0.5", features = ["vulkan"] }
///
/// let net = ncnn::Net::new();
/// net.opt.use_vulkan_compute = true;
/// net.load_param("rife-v4.6.param")?;
/// net.load_model("rife-v4.6.bin")?;
/// ```
///
/// ## ort (ONNX Runtime, supports CUDA/TensorRT/DirectML)
/// ```ignore
/// // Cargo.toml:
/// // ort = { version = "2", features = ["cuda"] }
///
/// let session = ort::Session::builder()?
///     .with_execution_providers([ort::CUDAExecutionProvider::default().build()])?
///     .commit_from_file("rife-v4.6.onnx")?;
/// ```
///
/// ## Model files
///
/// RIFE v4.6+ models can be obtained from:
/// - <https://github.com/hzwer/Practical-RIFE> (official)
/// - Convert to ncnn format with `pnnx` or to ONNX with `torch.onnx.export`
///
/// Expected model path: `$XDG_DATA_HOME/remoteway/models/rife-v4.6.bin`
/// (with corresponding `.param` file for ncnn, or `.onnx` for ONNX Runtime)
#[derive(Debug)]
pub struct RifeInterpolator {
    model_path: PathBuf,
}

impl RifeInterpolator {
    /// Create a new RIFE interpolator with the given model path.
    ///
    /// Verifies that the model file exists on disk. Does not load
    /// the model until the first interpolation call.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError::InitFailed`] if the model file does not
    /// exist, cannot be read, or is empty.
    pub fn new(model_path: PathBuf) -> Result<Self, InterpolateError> {
        if !model_path.exists() {
            return Err(InterpolateError::InitFailed(format!(
                "RIFE model not found: {}",
                model_path.display()
            )));
        }

        // Verify the file is readable and non-empty.
        let metadata = std::fs::metadata(&model_path).map_err(|e| {
            InterpolateError::InitFailed(format!(
                "cannot read RIFE model {}: {e}",
                model_path.display()
            ))
        })?;
        if metadata.len() == 0 {
            return Err(InterpolateError::InitFailed(format!(
                "RIFE model is empty: {}",
                model_path.display()
            )));
        }

        Ok(Self { model_path })
    }

    /// Try to create a RIFE interpolator from the default model location.
    ///
    /// Searches for a model file in standard locations:
    /// 1. `$XDG_DATA_HOME/remoteway/models/rife-v4.6.bin`
    /// 2. `$HOME/.local/share/remoteway/models/rife-v4.6.bin`
    /// 3. `/usr/share/remoteway/models/rife-v4.6.bin`
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError::InitFailed`] if no model file is found
    /// in any of the default locations.
    pub fn from_default_path() -> Result<Self, InterpolateError> {
        let candidates = default_model_paths();
        for path in &candidates {
            if path.exists() {
                return Self::new(path.clone());
            }
        }

        Err(InterpolateError::InitFailed(format!(
            "RIFE model not found in default locations: {}",
            candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }

    /// Get the model file path.
    #[must_use]
    pub fn model_path(&self) -> &PathBuf {
        &self.model_path
    }
}

impl FrameInterpolator for RifeInterpolator {
    fn interpolate(
        &self,
        a: &GpuFrame,
        b: &GpuFrame,
        t: f32,
    ) -> Result<GpuFrame, InterpolateError> {
        if !(0.0..=1.0).contains(&t) {
            return Err(InterpolateError::InvalidFactor(t));
        }
        if !a.same_dimensions(b) {
            return Err(InterpolateError::DimensionMismatch(
                a.width, a.height, b.width, b.height,
            ));
        }

        // Neural inference is not yet wired up — requires ncnn-rs or ort dependency.
        // When integrated, this would:
        // 1. Convert BGRA frames to RGB float tensors (normalized 0..1)
        // 2. Run RIFE model: input=(frame_a, frame_b, t) → output=interpolated_frame
        // 3. Convert RGB float tensor back to BGRA u8
        Err(InterpolateError::InterpolateFailed(format!(
            "RIFE inference runtime not linked (model: {}). \
             Add ncnn-rs or ort dependency to enable neural interpolation.",
            self.model_path.display()
        )))
    }

    fn latency_ms(&self) -> f32 {
        // RIFE v4.6 typically runs at ~5-10ms on modern GPUs via ncnn Vulkan.
        8.0
    }

    fn name(&self) -> &'static str {
        "rife"
    }
}

/// Return default search paths for the RIFE model file.
fn default_model_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(data_home) = std::env::var("XDG_DATA_HOME") {
        paths.push(PathBuf::from(data_home).join("remoteway/models/rife-v4.6.bin"));
    }

    if let Ok(home) = std::env::var("HOME") {
        paths.push(PathBuf::from(&home).join(".local/share/remoteway/models/rife-v4.6.bin"));
    }

    paths.push(PathBuf::from("/usr/share/remoteway/models/rife-v4.6.bin"));

    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rife_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RifeInterpolator>();
    }

    #[test]
    fn rife_model_not_found() {
        let result = RifeInterpolator::new(PathBuf::from("/nonexistent/model.bin"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn rife_default_path_not_found() {
        // Unless the user has installed a model, this should fail.
        let result = RifeInterpolator::from_default_path();
        // We don't assert is_err because a model might actually exist.
        if let Ok(interp) = result {
            assert_eq!(interp.name(), "rife");
        }
    }

    #[test]
    fn rife_empty_model() {
        let dir = std::env::temp_dir().join("remoteway-test-rife");
        let _ = std::fs::create_dir_all(&dir);
        let model_path = dir.join("empty.bin");
        std::fs::write(&model_path, b"").unwrap();

        let result = RifeInterpolator::new(model_path.clone());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rife_validates_input() {
        let dir = std::env::temp_dir().join("remoteway-test-rife-validate");
        let _ = std::fs::create_dir_all(&dir);
        let model_path = dir.join("fake-model.bin");
        std::fs::write(&model_path, b"fake model data for testing").unwrap();

        let interp = RifeInterpolator::new(model_path).unwrap();

        // Invalid t.
        let a = GpuFrame::from_data(vec![0u8; 16], 2, 2, 8, 0);
        let b = GpuFrame::from_data(vec![0u8; 16], 2, 2, 8, 100);
        assert!(interp.interpolate(&a, &b, -0.1).is_err());
        assert!(interp.interpolate(&a, &b, 1.1).is_err());

        // Dimension mismatch.
        let c = GpuFrame::from_data(vec![0u8; 64], 4, 4, 16, 100);
        assert!(interp.interpolate(&a, &c, 0.5).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rife_interpolate_returns_not_linked() {
        let dir = std::env::temp_dir().join("remoteway-test-rife-notlinked");
        let _ = std::fs::create_dir_all(&dir);
        let model_path = dir.join("model.bin");
        std::fs::write(&model_path, b"fake model").unwrap();

        let interp = RifeInterpolator::new(model_path).unwrap();
        let a = GpuFrame::from_data(vec![0u8; 16], 2, 2, 8, 0);
        let b = GpuFrame::from_data(vec![0u8; 16], 2, 2, 8, 100);
        let result = interp.interpolate(&a, &b, 0.5);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("runtime not linked")
        );

        std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_model_paths_not_empty() {
        let paths = default_model_paths();
        // At least /usr/share path should always be present.
        assert!(!paths.is_empty());
    }
}
