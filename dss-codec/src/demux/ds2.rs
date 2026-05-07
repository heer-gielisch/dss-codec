//! DS2 file demuxer.
//!
//! SP mode (0-1): byte-swap demuxing, returns list of 42-byte packets.
//! QP mode (6-7): segmented bitstream with annotation-aware extraction.
//!
//! ## QP block layout
//!
//! Every 512-byte block has a 6-byte header followed by 506 bytes of payload:
//!
//!   Byte 0: version / flags
//!   Byte 1: continuation offset in words (cont_bytes = Byte1 × 2),
//!           relative to block start (incl. header); first frame-byte in
//!           this block's payload is at payload_off = cont_bytes − 6.
//!   Byte 2: frame count in this block
//!   Byte 3: 0xFF marker
//!   Byte 4: format type  (0x00/0x01 = SP, 0x06/0x07 = QP)
//!   Byte 5: segment marker (0xFF = normal; anything else = last block of a
//!           physical segment; 0x00 = end of file)
//!
//! ## Physical segments
//!
//! The recorder stores audio as a sequence of independent 28-block cycles.
//! A discontinuity between the expected `frames_raw_start` and `raw_read_pos`
//! signals a boundary between two physically independent recording segments.
//! The decoder must reset its state at every such boundary.
//!
//! ## Annotations
//!
//! The file header at offset 0x400 contains pairs of uint32-LE values that
//! give the start and end of each annotation region in **milliseconds**:
//!
//!   annotation_start_ms = u32_le(data[0x400 + i*8])
//!   annotation_end_ms   = u32_le(data[0x404 + i*8])
//!
//! The list is terminated by 0xFFFF_FFFF or by reaching offset 0x500.
//! Each QP frame covers 256 samples at 16 000 Hz = 16 ms exactly.
//!
//! ## Extraction modes
//!
//! `ExtractionMode::All`              — return every frame in order (default)
//! `ExtractionMode::MainOnly`         — skip annotation frames
//! `ExtractionMode::AnnotationsOnly`  — skip main-text frames

use crate::error::{DecodeError, Result};

// ── layout constants ──────────────────────────────────────────────────────────
const DS2_HEADER_SIZE: usize = 0x600;
const DS2_BLOCK_SIZE: usize = 512;
const DS2_BLOCK_HEADER_SIZE: usize = 6;
const DS2_BLOCK_PAYLOAD_SIZE: usize = DS2_BLOCK_SIZE - DS2_BLOCK_HEADER_SIZE; // 506

const DSS_SP_PACKET_SIZE: usize = 42;

/// QP: 448 bits per frame = 56 bytes per frame.
pub const QP_FRAME_BYTES: usize = 56;

/// Each QP frame covers 256 samples at 16 000 Hz = 16 ms.
const MS_PER_FRAME: usize = 16;

/// Maximum number of annotation entries read from the header.
const MAX_ANNOTATIONS: usize = 32;

// ── public types ──────────────────────────────────────────────────────────────

/// Controls which frames are included in the demuxer output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionMode {
    /// Return all frames in their original order.
    All,
    /// Return only main-text frames (annotation frames are skipped).
    MainOnly,
    /// Return only annotation frames (main-text frames are skipped).
    AnnotationsOnly,
}

/// One uninterrupted run of QP frames ready for the bitstream decoder.
///
/// `stream` contains exactly `frame_count × 56` bytes.
/// `reset_before` is true for every segment that follows a physical segment
/// boundary or a non-contiguous splice — the decoder **must** reset its state
/// (`lattice_state`, `pitch_memory`, `deemph_state`) before processing it.
pub struct QpSegment {
    pub stream: Vec<u8>,
    pub frame_count: usize,
    /// True → decoder must call `reset()` before processing this segment.
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

// ── entry points ──────────────────────────────────────────────────────────────

/// Demux a DS2 file, returning all frames in order.
///
/// Equivalent to `demux_ds2_ex(data, ExtractionMode::All)`.
pub fn demux_ds2(data: &[u8]) -> Result<DemuxedDs2> {
    demux_ds2_ex(data, ExtractionMode::All)
}

/// Demux a DS2 file with explicit extraction mode.
///
/// For SP files the `mode` parameter is ignored — SP has no annotation
/// metadata and all frames are always returned.
pub fn demux_ds2_ex(data: &[u8], mode: ExtractionMode) -> Result<DemuxedDs2> {
    // Bug fix: check bytes 1-3 for "ds2" and byte 0 for version (0x01, 0x03, or
    // 0x07) separately, instead of comparing all 4 bytes as a literal, which can
    // fail silently when the escape sequence is mis-encoded in the source file.
    if data.len() < 4
        || &data[1..4] != b"ds2"
        || (data[0] != 0x01 && data[0] != 0x03 && data[0] != 0x07)
    {
        return Err(DecodeError::NotDs2(std::path::PathBuf::from("<bytes>")));
    }

    // The 0x07 variant uses a 0x1000-byte file header; all others use 0x600.
    let header_size = if data[0] == 0x07 { 0x1000usize } else { DS2_HEADER_SIZE };

    if data.len() < header_size + DS2_BLOCK_SIZE {
        return Err(DecodeError::NotDs2(std::path::PathBuf::from("<bytes>")));
    }

    let num_blocks = (data.len() - header_size) / DS2_BLOCK_SIZE;
    if num_blocks == 0 {
        return Err(DecodeError::NoAudioData);
    }

    // Detect format from byte 4 of the first block that actually carries
    // frames. Some files (notably the \x07ds2 variant) have leading
    // zero-padding blocks whose header bytes are all zero; reading byte 4
    // there would falsely route a QP file to the SP branch.
    let mut format_type: u8 = 0;
    for bi in 0..num_blocks {
        let bstart = header_size + bi * DS2_BLOCK_SIZE;
        if data[bstart + 2] > 0 {
            format_type = data[bstart + 4];
            break;
        }
    }

    if format_type >= 6 {
        demux_qp(data, num_blocks, header_size, mode)
    } else {
        demux_sp(data, num_blocks, header_size)
    }
}

// ── annotation metadata ───────────────────────────────────────────────────────

/// Annotation region as a half-open frame-index range [first_frame, last_frame).
#[derive(Debug, Clone, Copy)]
struct AnnRange {
    first_frame: usize, // inclusive
    last_frame: usize,  // exclusive
}

/// Parse annotation time ranges from file header offset 0x400.
///
/// Pairs of uint32-LE values encode (start_ms, end_ms).
/// Terminated by 0xFFFF_FFFF or by reaching offset 0x500.
fn read_annotations(data: &[u8]) -> Vec<AnnRange> {
    let mut result = Vec::new();
    let mut off = 0x400usize;

    while off + 8 <= 0x500.min(data.len()) && result.len() < MAX_ANNOTATIONS {
        let start_ms =
            u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        if start_ms == 0xFFFF_FFFF {
            break;
        }
        let end_ms =
            u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;

        result.push(AnnRange {
            first_frame: start_ms / MS_PER_FRAME,
            last_frame: end_ms / MS_PER_FRAME,
        });

        off += 8;
    }

    result
}

/// Return true if `abs_frame` falls inside any annotation region.
#[inline]
fn is_annotation(abs_frame: usize, annotations: &[AnnRange]) -> bool {
    annotations
        .iter()
        .any(|a| abs_frame >= a.first_frame && abs_frame < a.last_frame)
}

// ── QP demuxer ────────────────────────────────────────────────────────────────

fn demux_qp(data: &[u8], num_blocks: usize, header_size: usize, mode: ExtractionMode) -> Result<DemuxedDs2> {
    // ── Step 1: build the linear raw payload buffer ───────────────────────────
    let mut raw: Vec<u8> = Vec::with_capacity(num_blocks * DS2_BLOCK_PAYLOAD_SIZE);
    let mut total_frames_in_file: usize = 0;
    for bi in 0..num_blocks {
        let bstart = header_size + bi * DS2_BLOCK_SIZE;
        total_frames_in_file += data[bstart + 2] as usize;
        raw.extend_from_slice(
            &data[bstart + DS2_BLOCK_HEADER_SIZE..bstart + DS2_BLOCK_SIZE],
        );
    }

    // Bug fix: if the file ends without an empty-block trailer, the last frame
    // may extend a few bytes beyond the raw buffer.  Pad with zeros so the
    // bitstream reader never reads out of bounds.  The padding produces silent
    // audio for the last partial frame rather than garbage or a panic.
    let first_cont = data[header_size + 1] as usize * 2;
    let first_payload_off = first_cont.saturating_sub(DS2_BLOCK_HEADER_SIZE);
    let min_raw_len = first_payload_off + total_frames_in_file * QP_FRAME_BYTES;
    if raw.len() < min_raw_len {
        raw.resize(min_raw_len, 0u8);
    }

    // ── Step 2: read annotation metadata ─────────────────────────────────────
    let annotations = read_annotations(data);

    // ── Step 3: identify physical segment boundaries ──────────────────────────
    //
    // For each block:
    //   payload_off      = Byte1*2 − 6          (where this block's frames start)
    //   frames_raw_start = bi*506 + payload_off
    //
    // After consuming frame_count*56 bytes, raw_read_pos must equal
    // frames_raw_start of the next block.  A mismatch is a physical boundary.

    struct PhysSeg {
        raw_start: usize,
        frame_start: usize,
        frame_count: usize,
    }

    let mut phys_segs: Vec<PhysSeg> = Vec::new();
    let mut raw_read_pos: usize = 0;
    let mut seg_raw_start: usize = 0;
    let mut seg_frames: usize = 0;
    let mut seg_frame_start: usize = 0;
    let mut abs_frames_done: usize = 0;

    for bi in 0..num_blocks {
        let bstart = header_size + bi * DS2_BLOCK_SIZE;
        let cont_bytes = data[bstart + 1] as usize * 2;
        let frame_count = data[bstart + 2] as usize;

        let payload_off = cont_bytes.saturating_sub(DS2_BLOCK_HEADER_SIZE);
        let frames_raw_start = bi * DS2_BLOCK_PAYLOAD_SIZE + payload_off;

        if bi == 0 {
            raw_read_pos = frames_raw_start;
            seg_raw_start = frames_raw_start;
            seg_frame_start = 0;
        } else if frames_raw_start != raw_read_pos {
            // Physical segment boundary — flush current segment.
            let raw_end = raw_read_pos.min(raw.len());
            if seg_frames > 0 && raw_end > seg_raw_start {
                phys_segs.push(PhysSeg {
                    raw_start: seg_raw_start,
                    frame_start: seg_frame_start,
                    frame_count: seg_frames,
                });
            }
            abs_frames_done += seg_frames;
            seg_frame_start = abs_frames_done;
            seg_frames = 0;
            seg_raw_start = frames_raw_start;
            raw_read_pos = frames_raw_start;
        }

        if frame_count > 0 {
            seg_frames += frame_count;
            raw_read_pos += frame_count * QP_FRAME_BYTES;
        }
    }
    // Flush the last physical segment.
    let raw_end = raw_read_pos.min(raw.len());
    if seg_frames > 0 && raw_end > seg_raw_start {
        phys_segs.push(PhysSeg {
            raw_start: seg_raw_start,
            frame_start: seg_frame_start,
            frame_count: seg_frames,
        });
    }

    // ── Step 4: slice each physical segment into annotation/main runs ─────────
    //
    // Within each physical segment we scan frame-by-frame.  Whenever the
    // annotation classification changes, we flush the current run as a
    // QpSegment (if its classification is selected by `mode`).
    //
    // Every output segment after the first has reset_before = true because
    // the encoder context at its start is unknown to the decoder.

    let mut out_segments: Vec<QpSegment> = Vec::new();
    let mut total_frames_out: usize = 0;
    let mut first_output_seg = true;

    for phys in &phys_segs {
        if phys.frame_count == 0 {
            continue;
        }

        let mut run_is_ann = is_annotation(phys.frame_start, &annotations);
        let mut run_start: usize = 0;

        for local in 1..=phys.frame_count {
            let cur_is_ann = if local < phys.frame_count {
                is_annotation(phys.frame_start + local, &annotations)
            } else {
                !run_is_ann // force flush at end of segment
            };

            if cur_is_ann != run_is_ann {
                let run_len = local - run_start;
                let include = match mode {
                    ExtractionMode::All => true,
                    ExtractionMode::MainOnly => !run_is_ann,
                    ExtractionMode::AnnotationsOnly => run_is_ann,
                };

                if include && run_len > 0 {
                    let byte_off = phys.raw_start + run_start * QP_FRAME_BYTES;
                    let byte_end = byte_off + run_len * QP_FRAME_BYTES;
                    if byte_end <= raw.len() {
                        out_segments.push(QpSegment {
                            stream: raw[byte_off..byte_end].to_vec(),
                            frame_count: run_len,
                            reset_before: !first_output_seg,
                        });
                        first_output_seg = false;
                        total_frames_out += run_len;
                    }
                }

                run_start = local;
                run_is_ann = cur_is_ann;
            }
        }
    }

    Ok(DemuxedDs2::Qp {
        segments: out_segments,
        total_frames: total_frames_out,
    })
}

// ── SP demuxer (unchanged) ────────────────────────────────────────────────────

fn demux_sp(data: &[u8], num_blocks: usize, header_size: usize) -> Result<DemuxedDs2> {
    let mut raw: Vec<u8> = Vec::with_capacity(num_blocks * DS2_BLOCK_PAYLOAD_SIZE);
    let mut total_frames: usize = 0;

    for bi in 0..num_blocks {
        let bstart = header_size + bi * DS2_BLOCK_SIZE;
        total_frames += data[bstart + 2] as usize;
        raw.extend_from_slice(
            &data[bstart + DS2_BLOCK_HEADER_SIZE..bstart + DS2_BLOCK_SIZE],
        );
    }

    let mut swap = ((data[header_size] >> 7) & 1) as usize;
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