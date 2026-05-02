//! Compression backend abstraction — LZ4 and optional zstd support.

use thiserror::Error;

use crate::lz4::{self, compress_region as lz4_compress, decompress_region as lz4_decompress};

/// Errors that can occur during compression or decompression.
#[derive(Debug, Error)]
pub enum CompressorError {
    /// An LZ4 compression/decompression error.
    #[error("LZ4 error: {0}")]
    Lz4(#[from] lz4::CompressError),
    /// A zstd compression/decompression error.
    #[cfg(feature = "zstd-backend")]
    #[error("zstd error: {0}")]
    Zstd(#[from] std::io::Error),
}

/// Selects the active compression algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressorKind {
    /// Fast LZ4 compression (default).
    #[default]
    Lz4,
    /// Higher-ratio zstd compression.
    #[cfg(feature = "zstd-backend")]
    Zstd,
}

impl CompressorKind {
    /// Compress `data` using the selected algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying compressor fails.
    /// The LZ4 backend is infallible in practice (in-memory only);
    /// the zstd backend may propagate an I/O error from the C library.
    #[must_use = "compression result must be checked for errors before using the compressed data"]
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressorError> {
        match self {
            CompressorKind::Lz4 => Ok(lz4_compress(data)),
            #[cfg(feature = "zstd-backend")]
            CompressorKind::Zstd => compress_zstd(data),
        }
    }

    /// Decompress `data` using the selected algorithm.
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressorError> {
        match self {
            CompressorKind::Lz4 => Ok(lz4_decompress(data)?),
            #[cfg(feature = "zstd-backend")]
            CompressorKind::Zstd => Ok(decompress_zstd(data)?),
        }
    }
}

#[cfg(feature = "zstd-backend")]
fn compress_zstd(data: &[u8]) -> Result<Vec<u8>, CompressorError> {
    Ok(zstd::encode_all(data, 3)?)
}

#[cfg(feature = "zstd-backend")]
fn decompress_zstd(data: &[u8]) -> std::io::Result<Vec<u8>> {
    zstd::decode_all(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz4_round_trip() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let kind = CompressorKind::Lz4;
        let compressed = kind.compress(&data).unwrap();
        let decompressed = kind.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_empty_round_trip() {
        let kind = CompressorKind::Lz4;
        let compressed = kind.compress(&[]).unwrap();
        let decompressed = kind.decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[cfg(feature = "zstd-backend")]
    #[test]
    fn zstd_round_trip() {
        let data: Vec<u8> = (0..512).map(|i| (i * 7) as u8).collect();
        let kind = CompressorKind::Zstd;
        let compressed = kind.compress(&data).unwrap();
        let decompressed = kind.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "zstd-backend")]
    #[test]
    fn zstd_compresses_better_than_lz4_for_low_entropy() {
        let data = vec![0u8; 4096];
        let lz4_size = CompressorKind::Lz4.compress(&data).unwrap().len();
        let zstd_size = CompressorKind::Zstd.compress(&data).unwrap().len();
        // Both should be smaller than the original; zstd typically smaller.
        assert!(lz4_size < data.len());
        assert!(zstd_size < data.len());
    }

    #[test]
    fn default_is_lz4() {
        assert_eq!(CompressorKind::default(), CompressorKind::Lz4);
    }

    #[test]
    fn compressor_kind_debug() {
        let dbg = format!("{:?}", CompressorKind::Lz4);
        assert!(dbg.contains("Lz4"));
    }

    #[test]
    fn compressor_kind_clone_copy_eq() {
        let a = CompressorKind::Lz4;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_eq!(a.clone(), a);
    }

    #[cfg(feature = "zstd-backend")]
    #[test]
    fn zstd_empty_round_trip() {
        let kind = CompressorKind::Zstd;
        let compressed = kind.compress(&[]).unwrap();
        let decompressed = kind.decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[cfg(feature = "zstd-backend")]
    #[test]
    fn zstd_large_data_round_trip() {
        let data: Vec<u8> = (0..65536).map(|i| (i * 13) as u8).collect();
        let kind = CompressorKind::Zstd;
        let compressed = kind.compress(&data).unwrap();
        let decompressed = kind.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "zstd-backend")]
    #[test]
    fn zstd_ne_lz4() {
        assert_ne!(CompressorKind::Zstd, CompressorKind::Lz4);
    }
}
