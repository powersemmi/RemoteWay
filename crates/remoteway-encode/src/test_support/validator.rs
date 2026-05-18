//! FFmpeg-based bitstream validator.
//!
//! Wraps `ffmpeg-next` to take a raw encoded bitstream (Annex-B for
//! H.264/H.265, OBU stream for AV1), decode it back to YUV, and expose
//! a PSNR-Y comparison against a reference frame.
//!
//! Approach: write the bitstream to a `tempfile` with a codec-appropriate
//! extension and open it via `format::input(path)`. This is more reliable
//! than feeding raw bytes via custom IO — FFmpeg's demuxers/probers
//! handle codec-specific framing details (NAL splitting, OBU boundaries)
//! that would be tedious to re-implement.
//!
//! Tests pay the cost of one fs write per validation; for the test suite
//! sizes we expect (3-10 frames per test) this is negligible.

use std::io::Write;

use ffmpeg_next as ff;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::software::scaling::{Context as Scaler, Flags as ScaleFlags};
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use remoteway_vulkan::VideoCodec;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidatorError {
    #[error("ffmpeg init failed: {0}")]
    Init(String),
    #[error("temp file: {0}")]
    Io(#[from] std::io::Error),
    #[error("ffmpeg: {0}")]
    Ffmpeg(#[from] ff::Error),
    #[error("no video stream found in bitstream")]
    NoVideoStream,
    #[error("decoder open: {0}")]
    DecoderOpen(String),
}

#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    /// Luma plane, tightly packed (no row padding).
    pub y: Vec<u8>,
    /// Cb plane at half resolution.
    pub u: Vec<u8>,
    /// Cr plane at half resolution.
    pub v: Vec<u8>,
}

/// Decodes a raw codec bitstream and returns every frame the decoder
/// produces.
///
/// Internally:
/// 1. Writes `data` to a tempfile with the codec's conventional extension.
/// 2. Opens via `ffmpeg::format::input`.
/// 3. Streams packets to the decoder, collecting every emitted `VideoFrame`.
/// 4. Converts each frame to YUV420P (one byte per sample, three planes)
///    via `sws_scale` regardless of the decoder's native output format.
/// 5. Copies plane data into owned `Vec<u8>`s.
pub fn decode_bitstream(
    codec: VideoCodec,
    data: &[u8],
) -> Result<Vec<DecodedFrame>, ValidatorError> {
    ff::init().map_err(|e| ValidatorError::Init(e.to_string()))?;

    let mut tmp = tempfile::Builder::new()
        .prefix("remoteway-bitstream-")
        .suffix(match codec {
            VideoCodec::H264 => ".h264",
            VideoCodec::H265 => ".h265",
            VideoCodec::Av1 => ".obu",
        })
        .tempfile()?;
    tmp.write_all(data)?;
    tmp.flush()?;

    let mut input = ff::format::input(tmp.path())?;
    let stream = input
        .streams()
        .best(ff::media::Type::Video)
        .ok_or(ValidatorError::NoVideoStream)?;
    let stream_idx = stream.index();

    let codec_params = stream.parameters();
    let decoder_ctx = ff::codec::Context::from_parameters(codec_params)?;
    let mut decoder = decoder_ctx
        .decoder()
        .video()
        .map_err(|e| ValidatorError::DecoderOpen(e.to_string()))?;

    let mut frames = Vec::new();
    let mut scaler: Option<Scaler> = None;

    let mut decode_pending = |dec: &mut ff::decoder::Video,
                              scaler: &mut Option<Scaler>,
                              out: &mut Vec<DecodedFrame>|
     -> Result<(), ValidatorError> {
        let mut frame = VideoFrame::empty();
        while dec.receive_frame(&mut frame).is_ok() {
            let s = scaler.get_or_insert_with(|| {
                Scaler::get(
                    frame.format(),
                    frame.width(),
                    frame.height(),
                    Pixel::YUV420P,
                    frame.width(),
                    frame.height(),
                    ScaleFlags::FAST_BILINEAR,
                )
                .expect("scaler init")
            });
            let mut out_frame = VideoFrame::empty();
            s.run(&frame, &mut out_frame)?;
            out.push(frame_to_yuv420p(&out_frame));
        }
        Ok(())
    };

    for (stream, packet) in input.packets() {
        if stream.index() != stream_idx {
            continue;
        }
        decoder.send_packet(&packet)?;
        decode_pending(&mut decoder, &mut scaler, &mut frames)?;
    }
    decoder.send_eof()?;
    decode_pending(&mut decoder, &mut scaler, &mut frames)?;

    Ok(frames)
}

fn frame_to_yuv420p(frame: &VideoFrame) -> DecodedFrame {
    let w = frame.width();
    let h = frame.height();
    let y = copy_plane(frame, 0, w as usize, h as usize);
    let u = copy_plane(frame, 1, (w / 2) as usize, (h / 2) as usize);
    let v = copy_plane(frame, 2, (w / 2) as usize, (h / 2) as usize);
    DecodedFrame {
        width: w,
        height: h,
        y,
        u,
        v,
    }
}

fn copy_plane(frame: &VideoFrame, idx: usize, w: usize, h: usize) -> Vec<u8> {
    let stride = frame.stride(idx);
    let data = frame.data(idx);
    let mut out = Vec::with_capacity(w * h);
    for row in 0..h {
        let row_start = row * stride;
        out.extend_from_slice(&data[row_start..row_start + w]);
    }
    out
}

/// Peak signal-to-noise ratio on the luma plane.
///
/// `MAX = 255`. Returns `f64::INFINITY` for identical frames (MSE = 0).
/// Panics if the frames have different dimensions.
pub fn psnr_y(a: &DecodedFrame, b: &DecodedFrame) -> f64 {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "psnr_y: dimension mismatch"
    );
    assert_eq!(a.y.len(), b.y.len(), "psnr_y: y-plane length mismatch");

    let mse: f64 = a
        .y
        .iter()
        .zip(b.y.iter())
        .map(|(x, y)| {
            let diff = i32::from(*x) - i32::from(*y);
            f64::from(diff * diff)
        })
        .sum::<f64>()
        / a.y.len() as f64;

    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psnr_identical_is_infinite() {
        let f = DecodedFrame {
            width: 4,
            height: 4,
            y: vec![128; 16],
            u: vec![128; 4],
            v: vec![128; 4],
        };
        assert!(psnr_y(&f, &f).is_infinite());
    }

    #[test]
    fn psnr_small_difference_is_high() {
        let a = DecodedFrame {
            width: 4,
            height: 4,
            y: vec![128; 16],
            u: vec![128; 4],
            v: vec![128; 4],
        };
        let mut b = a.clone();
        b.y[0] = 130; // 2-step difference in one pixel
        let psnr = psnr_y(&a, &b);
        assert!(psnr > 40.0, "psnr should be high for tiny diff, got {psnr}");
    }

    #[test]
    fn psnr_large_difference_is_low() {
        let a = DecodedFrame {
            width: 4,
            height: 4,
            y: vec![0; 16],
            u: vec![128; 4],
            v: vec![128; 4],
        };
        let mut b = a.clone();
        b.y.fill(255);
        let psnr = psnr_y(&a, &b);
        assert!(psnr < 5.0, "psnr should be low for max diff, got {psnr}");
    }
}
