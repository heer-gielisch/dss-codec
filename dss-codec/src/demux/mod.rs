pub mod dss;
pub mod ds2;

/// Detected audio format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// Pure DSS file (.dss), SP codec at 11025 Hz output
    DssSp,
    /// DS2 file (.ds2), SP mode (mode byte 0-1), 12000 Hz
    Ds2Sp,
    /// DS2 file (.ds2), QP mode (mode byte 6-7), 16000 Hz
    Ds2Qp,
}

impl AudioFormat {
    pub fn native_sample_rate(&self) -> u32 {
        match self {
            AudioFormat::DssSp => 11025,
            AudioFormat::Ds2Sp => 12000,
            AudioFormat::Ds2Qp => 16000,
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::DssSp => "dss",
            AudioFormat::Ds2Sp | AudioFormat::Ds2Qp => "ds2",
        }
    }
}

/// Result of demuxing a file
pub struct DemuxResult {
    pub format: AudioFormat,
    pub frame_data: FrameData,
    pub total_frames: usize,
}

/// Frame data varies by format
pub enum FrameData {
    /// List of fixed-size packets (DSS SP, DS2 SP)
    Packets(Vec<Vec<u8>>),
    /// Continuous bitstream (DS2 QP)
    Stream(Vec<u8>),
}

/// Detect format from file header bytes
pub fn detect_format(data: &[u8]) -> Option<AudioFormat> {
    if data.len() < 4 {
        return None;
    }
    if data[1..4] == *b"dss" && (data[0] == 2 || data[0] == 3) {
        return Some(AudioFormat::DssSp);
    }
    if data[1..4] == *b"ds2" && (data[0] == 0x01 || data[0] == 0x03 || data[0] == 0x07) {
        let header_size = if data[0] == 0x07 { 0x1000usize } else { 0x600usize };
        if data.len() > header_size + 4 {
            let format_type = data[header_size + 4];
            if format_type >= 6 {
                return Some(AudioFormat::Ds2Qp);
            } else {
                return Some(AudioFormat::Ds2Sp);
            }
        }
    }
    None
}
