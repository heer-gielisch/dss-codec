//! DS2 file demuxer.
//!
//! SP mode (0-1): byte-swap demuxing, returns list of 42-byte packets.
//! QP mode (6-7): segmented bitstream with cut-point detection.
//!
//! ## QP block layout
//!
//! Every 512-byte block has a 6-byte header followed by 506 bytes of payload:
//!
//!   Byte 0: bit 7 = swap state (SP only)
//!   Byte 1: continuation offset in words (cont_bytes = Byte1 × 2)
//!           Offset is relative to block start (incl. header), so the first
//!           frame-byte in this block's payload is at:
//!               payload_off = cont_bytes - BLOCK_HEADER_SIZE
//!   Byte 2: frame count in this block
//!   Byte 3: 0xFF marker
//!   Byte 4: format type  (0x00/0x01 = SP, 0x06/0x07 = QP)
//!   Byte 5: 0xFF marker
//!
//! All 506-byte payloads are concatenated into one `raw` buffer.  For each
//! block, the first `payload_off` bytes in the payload are the tail of a frame
//! that started in the previous block (continuation bytes); the block's own
//! frames begin at `block_raw_start + payload_off`.
//!
//! ## Cut-point detection
//!
//! When a DS2 file has been edited, the editing software reduces `frame_count`
//! in the last block before the cut and adjusts `Byte 1` (cont offset) in the
//! first block after the cut.  This causes a discontinuity: after consuming
//! `frame_count × 56` bytes for a block, `raw_read_pos` no longer equals
//! `frames_raw_start` of the next block.  That gap is detected here and the
//! payload is split into separate `QpSegment`s.
//!
//! The decoder **must** reset its state (`lattice_state`, `pitch_memory`,
//! `deemph_state`) before processing each segment where `reset_before` is true.

use crate::error::{DecodeError, Result};

// ── layout constants ──────────────────────────────────────────────────────────
const DS2_HEADER_SIZE: usize = 0x600;
const DS2_BLOCK_SIZE: usize = 512;
const DS2_BLOCK_HEADER_SIZE: usize = 6;
const DS2_BLOCK_PAYLOAD_SIZE: usize = DS2_BLOCK_SIZE - DS2_BLOCK_HEADER_SIZE; // 506

const DSS_SP_PACKET_SIZE: usize = 42;

/// QP: 448 bits per frame = 56 bytes per frame.
pub const QP_FRAME_BYTES: usize = 56;

// ── public types ──────────────────────────────────────────────────────────────

/// One uninterrupted run of QP frames.
///
/// `stream` contains exactly `frame_count × 56` bytes, ready for the
/// bitstream reader.  When `reset_before` is true, the decoder state must be
/// reset before processing this segment (i.e. it follows a cut point).
pub struct QpSegment {
    pub stream: Vec<u8>,
    pub frame_count: usize,
    /// True for every segment after the first — the decoder must be reset.
    pub reset_before: bool,
}

pub enum DemuxedDs2 {
    Sp {
        packets: Vec<Vec<u8>>,
        total_frames: usize,
    },
    Qp {
        segments: Vec<QpSegment>,
        total_frames: usize,
    },
}

// ── entry point ───────────────────────────────────────────────────────────────

pub fn demux_ds2(data: &[u8]) -> Result<DemuxedDs2> {
    if data.len() < 4
        || (data[..4] != *b"\x03ds2" && data[..4] != *b"\x01ds2")
    {
        return Err(DecodeError::NotDs2(std::path::PathBuf::from("<bytes>")));
    }

    let num_blocks = (data.len() - DS2_HEADER_SIZE) / DS2_BLOCK_SIZE;
    if num_blocks == 0 {
        return Err(DecodeError::NoAudioData);
    }

    let format_type = data[DS2_HEADER_SIZE + 4];

    if format_type >= 6 {
        demux_qp(data, num_blocks)
    } else {
        demux_sp(data, num_blocks)
    }
}

// ── QP demuxer ────────────────────────────────────────────────────────────────

fn demux_qp(data: &[u8], num_blocks: usize) -> Result<DemuxedDs2> {
    // Step 1: concatenate all 506-byte payloads into one linear buffer.
    let mut raw: Vec<u8> = Vec::with_capacity(num_blocks * DS2_BLOCK_PAYLOAD_SIZE);
    let mut total_frames: usize = 0;
    for bi in 0..num_blocks {
        let bstart = DS2_HEADER_SIZE + bi * DS2_BLOCK_SIZE;
        total_frames += data[bstart + 2] as usize;
        raw.extend_from_slice(
            &data[bstart + DS2_BLOCK_HEADER_SIZE..bstart + DS2_BLOCK_SIZE],
        );
    }

    // Step 2: walk blocks, detect cut points, and slice `raw` into segments.
    //
    // For each block:
    //   cont_bytes       = Byte1 × 2
    //   payload_off      = cont_bytes - 6   (offset within this block's payload
    //                                        where the block's own frames begin)
    //   frames_raw_start = bi × 506 + payload_off
    //
    // After consuming frame_count × 56 bytes, raw_read_pos must equal
    // frames_raw_start of the next block.  A mismatch signals a cut point.

    let mut segments: Vec<QpSegment> = Vec::new();
    let mut seg_raw_start: usize = 0;
    let mut seg_frames: usize = 0;
    let mut raw_read_pos: usize = 0;
    let mut first_seg = true;

    for bi in 0..num_blocks {
        let bstart = DS2_HEADER_SIZE + bi * DS2_BLOCK_SIZE;
        let cont_bytes = data[bstart + 1] as usize * 2;
        let frame_count = data[bstart + 2] as usize;

        let payload_off = cont_bytes.saturating_sub(DS2_BLOCK_HEADER_SIZE);
        let frames_raw_start = bi * DS2_BLOCK_PAYLOAD_SIZE + payload_off;

        if bi == 0 {
            raw_read_pos = frames_raw_start;
            seg_raw_start = frames_raw_start;
        } else if frames_raw_start != raw_read_pos {
            // ── Cut point detected ──────────────────────────────────────────
            // Flush the current segment.
            let end = raw_read_pos.min(raw.len());
            if seg_frames > 0 && end > seg_raw_start {
                segments.push(QpSegment {
                    stream: raw[seg_raw_start..end].to_vec(),
                    frame_count: seg_frames,
                    reset_before: !first_seg,
                });
                first_seg = false;
            }
            // Start a new segment at the new position.
            seg_raw_start = frames_raw_start;
            seg_frames = 0;
            raw_read_pos = frames_raw_start;
        }

        if frame_count > 0 {
            seg_frames += frame_count;
            raw_read_pos += frame_count * QP_FRAME_BYTES;
        }
    }

    // Flush the final (or only) segment.
    let end = raw_read_pos.min(raw.len());
    if seg_frames > 0 && end > seg_raw_start {
        segments.push(QpSegment {
            stream: raw[seg_raw_start..end].to_vec(),
            frame_count: seg_frames,
            reset_before: !first_seg,
        });
    }

    Ok(DemuxedDs2::Qp {
        segments,
        total_frames,
    })
}

// ── SP demuxer ────────────────────────────────────────────────────────────────

fn demux_sp(data: &[u8], num_blocks: usize) -> Result<DemuxedDs2> {
    let mut raw: Vec<u8> = Vec::with_capacity(num_blocks * DS2_BLOCK_PAYLOAD_SIZE);
    let mut total_frames: usize = 0;
    for bi in 0..num_blocks {
        let bstart = DS2_HEADER_SIZE + bi * DS2_BLOCK_SIZE;
        total_frames += data[bstart + 2] as usize;
        raw.extend_from_slice(
            &data[bstart + DS2_BLOCK_HEADER_SIZE..bstart + DS2_BLOCK_SIZE],
        );
    }

    // Byte-swap demuxing (FFmpeg dss.c scheme, adapted for DS2 SP).
    let mut swap = ((data[DS2_HEADER_SIZE] >> 7) & 1) as usize;
    let mut swap_byte: u8 = 0;
    let mut pos: usize = 0;
    let mut frame_packets: Vec<Vec<u8>> = Vec::with_capacity(total_frames);

    for _fi in 0..total_frames {
        let mut pkt = [0u8; DSS_SP_PACKET_SIZE + 1];

        if swap != 0 {
            let read_size = 40;
            let end = (pos + read_size).min(raw.len());
            let count = end - pos;
            pkt[3..3 + count].copy_from_slice(&raw[pos..end]);
            pos += read_size;
            for i in (0..DSS_SP_PACKET_SIZE - 2).step_by(2) {
                pkt[i] = pkt[i + 4];
            }
            pkt[DSS_SP_PACKET_SIZE] = 0;
            pkt[1] = swap_byte;
        } else {
            let end = (pos + DSS_SP_PACKET_SIZE).min(raw.len());
            let count = end - pos;
            pkt[..count].copy_from_slice(&raw[pos..end]);
            pos += DSS_SP_PACKET_SIZE;
            swap_byte = pkt[DSS_SP_PACKET_SIZE - 2];
        }

        pkt[DSS_SP_PACKET_SIZE - 2] = 0;
        swap ^= 1;
        frame_packets.push(pkt[..DSS_SP_PACKET_SIZE].to_vec());
    }

    Ok(DemuxedDs2::Sp {
        packets: frame_packets,
        total_frames,
    })
}
