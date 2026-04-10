/// DS2 file demuxer.
///
/// SP mode (0-1): byte-swap demuxing, returns list of 42-byte packets.
/// QP mode (6-7): continuous bitstream, returns raw byte stream + frame count.
use crate::error::{DecodeError, Result};

const DS2_HEADER_SIZE: usize = 0x600;
const DS2_BLOCK_SIZE: usize = 512;
const DS2_BLOCK_HEADER_SIZE: usize = 6;
const DSS_SP_PACKET_SIZE: usize = 42;

/// Demux a DS2 file.
/// Returns (frame_data, total_frames, is_qp).
/// For SP: frame_data is a Vec<Vec<u8>> of packets.
/// For QP: frame_data is a single Vec<u8> continuous bitstream.
pub fn demux_ds2(data: &[u8]) -> Result<DemuxedDs2> {
    if data.len() < 4 || data[..4] != *b"\x03ds2" {
        return Err(DecodeError::NotDs2(std::path::PathBuf::from("<bytes>")));
    }

    let num_blocks = (data.len() - DS2_HEADER_SIZE) / DS2_BLOCK_SIZE;
    let format_type = data[DS2_HEADER_SIZE + 4];

    let mut total_frames: usize = 0;
    for bi in 0..num_blocks {
        total_frames += data[DS2_HEADER_SIZE + bi * DS2_BLOCK_SIZE + 2] as usize;
    }

    if format_type >= 6 {
        // QP mode: continuous bitstream (no byte-swap).
        // Empty blocks (frame_count=0) contain only a partial frame continuation;
        // the remaining payload bytes are garbage and must be discarded.
        // Valid continuation bytes = max(0, byte1*2 - 6), same formula as DSS empty blocks.
        let mut stream = Vec::new();
        for bi in 0..num_blocks {
            let bstart = DS2_HEADER_SIZE + bi * DS2_BLOCK_SIZE;
            let fc = data[bstart + 2];
            let b1 = data[bstart + 1] as usize;
            if fc == 0 {
                let cont_size = (b1 * 2).saturating_sub(DS2_BLOCK_HEADER_SIZE);
                stream.extend_from_slice(
                    &data[bstart + DS2_BLOCK_HEADER_SIZE
                        ..bstart + DS2_BLOCK_HEADER_SIZE + cont_size],
                );
            } else {
                stream.extend_from_slice(
                    &data[bstart + DS2_BLOCK_HEADER_SIZE..bstart + DS2_BLOCK_SIZE],
                );
            }
        }
        Ok(DemuxedDs2::Qp {
            stream,
            total_frames,
        })
    } else {
        // SP mode: byte-swap demuxing
        let mut stream = Vec::new();
        for bi in 0..num_blocks {
            let bstart = DS2_HEADER_SIZE + bi * DS2_BLOCK_SIZE;
            stream.extend_from_slice(
                &data[bstart + DS2_BLOCK_HEADER_SIZE..bstart + DS2_BLOCK_SIZE],
            );
        }

        let mut swap = ((data[DS2_HEADER_SIZE] >> 7) & 1) as usize;
        let mut swap_byte: u8 = 0;
        let mut pos: usize = 0;
        let mut frame_packets = Vec::with_capacity(total_frames);

        for _fi in 0..total_frames {
            let mut pkt = [0u8; DSS_SP_PACKET_SIZE + 1];
            if swap != 0 {
                let read_size = 40;
                let end = (pos + read_size).min(stream.len());
                let count = end - pos;
                pkt[3..3 + count].copy_from_slice(&stream[pos..end]);
                pos += read_size;
                for i in (0..DSS_SP_PACKET_SIZE - 2).step_by(2) {
                    pkt[i] = pkt[i + 4];
                }
                pkt[DSS_SP_PACKET_SIZE] = 0;
                pkt[1] = swap_byte;
            } else {
                let end = (pos + DSS_SP_PACKET_SIZE).min(stream.len());
                let count = end - pos;
                pkt[..count].copy_from_slice(&stream[pos..end]);
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
}

pub enum DemuxedDs2 {
    Sp {
        packets: Vec<Vec<u8>>,
        total_frames: usize,
    },
    Qp {
        stream: Vec<u8>,
        total_frames: usize,
    },
}
