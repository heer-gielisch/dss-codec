pub mod bitstream;
pub mod codec;
pub mod demux;
pub mod error;
pub mod output;
pub mod tables;

use crate::codec::ds2_qp::Ds2QpDecoder;
use crate::codec::ds2_sp::Ds2SpDecoder;
use crate::codec::dss_sp::DssSpDecoder;
use crate::demux::ds2::{demux_ds2, DemuxedDs2};
use crate::demux::dss::demux_dss;
use crate::demux::{detect_format, AudioFormat};
use crate::error::{DecodeError, Result};
use crate::output::resample::resample;
use crate::output::wav::write_wav;
use crate::output::OutputConfig;
use std::path::Path;

/// Decoded audio buffer
pub struct AudioBuffer {
    /// Samples as f64 (mono)
    pub samples: Vec<f64>,
    /// Native sample rate before any resampling
    pub native_rate: u32,
    /// Detected format
    pub format: AudioFormat,
}

/// Decode a DSS/DS2 file to an AudioBuffer.
pub fn decode_file(path: &Path) -> Result<AudioBuffer> {
    let data = std::fs::read(path)?;
    decode_to_buffer(&data)
}

/// Decode raw file bytes to an AudioBuffer.
pub fn decode_to_buffer(data: &[u8]) -> Result<AudioBuffer> {
    let format = detect_format(data)
        .ok_or_else(|| DecodeError::UnsupportedFormat(data.first().copied().unwrap_or(0)))?;
    match format {
        AudioFormat::DssSp => decode_dss_sp(data),
        AudioFormat::Ds2Sp => decode_ds2_sp(data),
        AudioFormat::Ds2Qp => decode_ds2_qp(data),
    }
}

fn decode_dss_sp(data: &[u8]) -> Result<AudioBuffer> {
    let (packets, total_frames) = demux_dss(data)?;
    let mut decoder = DssSpDecoder::new();
    let mut all_samples = Vec::with_capacity(total_frames * 264);
    for pkt in &packets {
        let frame_samples = decoder.decode_frame(pkt);
        all_samples.extend(frame_samples.iter().map(|&s| s as f64));
    }
    Ok(AudioBuffer {
        samples: all_samples,
        native_rate: 11025,
        format: AudioFormat::DssSp,
    })
}

fn decode_ds2_sp(data: &[u8]) -> Result<AudioBuffer> {
    let demuxed = demux_ds2(data)?;
    match demuxed {
        DemuxedDs2::Sp {
            packets,
            total_frames,
        } => {
            let mut decoder = Ds2SpDecoder::new();
            let mut all_samples = Vec::with_capacity(total_frames * 288);
            for pkt in &packets {
                let frame_samples = decoder.decode_frame(pkt);
                all_samples.extend_from_slice(&frame_samples);
            }
            Ok(AudioBuffer {
                samples: all_samples,
                native_rate: 12000,
                format: AudioFormat::Ds2Sp,
            })
        }
        _ => Err(DecodeError::UnsupportedFormat(6)),
    }
}

fn decode_ds2_qp(data: &[u8]) -> Result<AudioBuffer> {
    let demuxed = demux_ds2(data)?;
    match demuxed {
        DemuxedDs2::Qp {
            segments,
            total_frames: _,
        } => {
            let mut decoder = Ds2QpDecoder::new();
            let all_samples = decoder.decode_segments(&segments);
            Ok(AudioBuffer {
                samples: all_samples,
                native_rate: 16000,
                format: AudioFormat::Ds2Qp,
            })
        }
        _ => Err(DecodeError::UnsupportedFormat(0)),
    }
}

/// Decode a file and write to WAV with given output configuration.
pub fn decode_and_write(
    input: &Path,
    output: &Path,
    config: &OutputConfig,
) -> Result<AudioBuffer> {
    let mut buf = decode_file(input)?;
    let target_rate = config.sample_rate.unwrap_or(buf.native_rate);
    if target_rate != buf.native_rate {
        buf.samples = resample(&buf.samples, buf.native_rate, target_rate)?;
    }
    write_wav(output, &buf.samples, target_rate, config.bit_depth, config.channels)?;
    Ok(buf)
}