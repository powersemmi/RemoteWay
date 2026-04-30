use thiserror::Error;

use crate::lz4::{self, compress_region as lz4_compress, decompress_region as lz4_decompress};

#[derive(Debug, Error)]
pub enum CompressorError {
    #[error("LZ4 error: {0}")]
    Lz4(#[from] lz4::CompressError),
    #[cfg(feature = "zstd-backend")]
    #[error("zstd error: {0}")]
    Zstd(#[from] std::io::Error),
}

/// Selects the active compression algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressorKind {
    #[default]
    Lz4,
    #[cfg(feature = "zstd-backend")]
    Zstd,
}

impl CompressorKind {
    /// Compress `data` using the selected algorithm.
    pub fn compress(&self, data: &[u8]) -> Vec<u8> {
        match self {
            CompressorKind::Lz4 => lz4_compress(data),
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
fn compress_zstd(data: &[u8]) -> Vec<u8> {
    zstd::encode_all(data, 3).expect("zstd encode cannot fail on in-memory buffer")
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
        let compressed = kind.compress(&data);
        let decompressed = kind.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_empty_round_trip() {
        let kind = CompressorKind::Lz4;
        let compressed = kind.compress(&[]);
        let decompressed = kind.decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[cfg(feature = "zstd-backend")]
    #[test]
    fn zstd_round_trip() {
        let data: Vec<u8> = (0..512).map(|i| (i * 7) as u8).collect();
        let kind = CompressorKind::Zstd;
        let compressed = kind.compress(&data);
        let decompressed = kind.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "zstd-backend")]
    #[test]
    fn zstd_compresses_better_than_lz4_for_low_entropy() {
        let data = vec![0u8; 4096];
        let lz4_size = CompressorKind::Lz4.compress(&data).len();
        let zstd_size = CompressorKind::Zstd.compress(&data).len();
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
        let compressed = kind.compress(&[]);
        let decompressed = kind.decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[cfg(feature = "zstd-backend")]
    #[test]
    fn zstd_large_data_round_trip() {
        let data: Vec<u8> = (0..65536).map(|i| (i * 13) as u8).collect();
        let kind = CompressorKind::Zstd;
        let compressed = kind.compress(&data);
        let decompressed = kind.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "zstd-backend")]
    #[test]
    fn zstd_ne_lz4() {
        assert_ne!(CompressorKind::Zstd, CompressorKind::Lz4);
    }
}
