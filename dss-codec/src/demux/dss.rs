/// DSS block-aware demuxer with cut-point detection.
///
/// Concatenates all raw block payloads and computes `frames_raw_start = bi * 506 + cont_size`
/// for each block. A mismatch with `raw_read_pos` signals an edit boundary; the decoder
/// must reset its state at every such boundary.
use crate::error::{DecodeError, Result};

const DSS_BLOCK_SIZE: usize = 512;
const DSS_BLOCK_HEADER_SIZE: usize = 6;
const DSS_BLOCK_PAYLOAD_SIZE: usize = DSS_BLOCK_SIZE - DSS_BLOCK_HEADER_SIZE; // 506
pub const DSS_SP_FRAME_SIZE: usize = 42;

pub struct DssSegment {
    pub raw: Vec<u8>,
    pub frame_count: usize,
    pub init_swap: usize,
    pub reset_before: bool,
}

pub fn demux_dss(data: &[u8]) -> Result<(Vec<DssSegment>, usize)> {
    if data.len() < 4 || data[1..4] != *b"dss" {
        return Err(DecodeError::NotDss(std::path::PathBuf::from("<bytes>")));
    }

    let version = data[0] as usize;
    let header_size = version * DSS_BLOCK_SIZE;
    let num_blocks = (data.len() - header_size) / DSS_BLOCK_SIZE;

    let total_frames: usize = (0..num_blocks)
        .map(|bi| data[header_size + bi * DSS_BLOCK_SIZE + 2] as usize)
        .sum();

    // Concatenate all raw payloads
    let mut raw = Vec::with_capacity(num_blocks * DSS_BLOCK_PAYLOAD_SIZE);
    for bi in 0..num_blocks {
        let bstart = header_size + bi * DSS_BLOCK_SIZE;
        raw.extend_from_slice(&data[bstart + DSS_BLOCK_HEADER_SIZE..bstart + DSS_BLOCK_SIZE]);
    }

    // Detect cut points and build segments
    let mut segments: Vec<DssSegment> = Vec::new();
    let mut seg_raw_start: usize = 0;
    let mut seg_frames: usize = 0;
    let mut seg_swap: usize = 0;
    let mut raw_read_pos: usize = 0;
    let mut current_swap: usize = 0;
    let mut first_seg = true;

    for bi in 0..num_blocks {
        let bstart = header_size + bi * DSS_BLOCK_SIZE;
        let byte0 = data[bstart];
        let byte1 = data[bstart + 1] as usize;
        let fc = data[bstart + 2] as usize;
        let blk_swap = ((byte0 >> 7) & 1) as usize;
        let cont_size = (2 * byte1 + 2 * blk_swap).saturating_sub(DSS_BLOCK_HEADER_SIZE);
        let frames_raw_start = bi * DSS_BLOCK_PAYLOAD_SIZE + cont_size;

        if bi == 0 {
            raw_read_pos = frames_raw_start;
            seg_raw_start = frames_raw_start;
            seg_swap = blk_swap;
            current_swap = blk_swap;
        } else if fc > 0 && frames_raw_start != raw_read_pos {
            if seg_frames > 0 {
                segments.push(DssSegment {
                    raw: raw[seg_raw_start..raw_read_pos].to_vec(),
                    frame_count: seg_frames,
                    init_swap: seg_swap,
                    reset_before: !first_seg,
                });
                first_seg = false;
            }
            seg_raw_start = frames_raw_start;
            seg_frames = 0;
            seg_swap = blk_swap;
            current_swap = blk_swap;
            raw_read_pos = frames_raw_start;
        }

        if fc > 0 {
            seg_frames += fc;
            for _ in 0..fc {
                raw_read_pos += if current_swap != 0 { 40 } else { DSS_SP_FRAME_SIZE };
                current_swap ^= 1;
            }
        }
    }

    if seg_frames > 0 {
        segments.push(DssSegment {
            raw: raw[seg_raw_start..raw_read_pos.min(raw.len())].to_vec(),
            frame_count: seg_frames,
            init_swap: seg_swap,
            reset_before: !first_seg,
        });
    }

    Ok((segments, total_frames))
}

pub fn demux_segment(seg: &DssSegment) -> Vec<Vec<u8>> {
    let mut swap = seg.init_swap;
    let mut swap_byte: u8 = 0;
    let mut pos: usize = 0;
    let mut packets = Vec::with_capacity(seg.frame_count);
    let stream = &seg.raw;

    for _ in 0..seg.frame_count {
        let mut pkt = [0u8; DSS_SP_FRAME_SIZE + 1];
        if swap != 0 {
            let read_size = 40;
            let end = (pos + read_size).min(stream.len());
            let count = end - pos;
            pkt[3..3 + count].copy_from_slice(&stream[pos..end]);
            pos += read_size;
            for i in (0..DSS_SP_FRAME_SIZE - 2).step_by(2) {
                pkt[i] = pkt[i + 4];
            }
            pkt[DSS_SP_FRAME_SIZE] = 0;
            pkt[1] = swap_byte;
        } else {
            let end = (pos + DSS_SP_FRAME_SIZE).min(stream.len());
            let count = end - pos;
            pkt[..count].copy_from_slice(&stream[pos..end]);
            pos += DSS_SP_FRAME_SIZE;
            swap_byte = pkt[DSS_SP_FRAME_SIZE - 2];
        }
        pkt[DSS_SP_FRAME_SIZE - 2] = 0;
        swap ^= 1;
        packets.push(pkt[..DSS_SP_FRAME_SIZE].to_vec());
    }

    packets
}
