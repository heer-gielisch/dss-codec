//! DS2 QP decoder — f64 lattice synthesis, 16000 Hz output.
//!
//! 16 reflection coefficients, 64-sample subframes, 4 subframes/frame,
//! 11-pulse combinatorial codebook C(64,11), continuous bitstream,
//! per-subframe pitch encoding (8 bits each), de-emphasis filter.
//!
//! ## Cut-file handling
//!
//! Use `decode_segments()` when decoding a file that may have been edited.
//! The demuxer splits the bitstream into `QpSegment`s and sets `reset_before`
//! on every segment that follows a cut point.  `decode_segments()` calls
//! `reset()` automatically before those segments so that stale `pitch_memory`
//! and `lattice_state` from before the cut do not corrupt the audio after it.

use crate::bitstream::BitstreamReader;
use crate::codec::common::{decode_combinatorial_index, lattice_synthesis};
use crate::demux::ds2::QpSegment;
use crate::tables::ds2_qp::qp_codebook_lookup;
use crate::tables::ds2_quant::{QP_EXCITATION_GAIN, QP_PITCH_GAIN, QP_PULSE_AMP};

const NUM_COEFFS: usize = 16;
const NUM_SUBFRAMES: usize = 4;
const SUBFRAME_SIZE: usize = 64;
const SAMPLES_PER_FRAME: usize = NUM_SUBFRAMES * SUBFRAME_SIZE; // 256
const MIN_PITCH: u32 = 45;
const MAX_PITCH: u32 = 300;
const EXCITATION_PULSES: usize = 11;
const REFL_BIT_ALLOC: [u32; NUM_COEFFS] = [7, 7, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3];
const PITCH_GAIN_BITS: u32 = 6;
const GAIN_BITS: u32 = 6;
const PULSE_BITS: u32 = 3;
const PITCH_BITS: u32 = 8;
// CB_BITS: verified from frame layout 76 + 4*(8+6+CB+6+33) = 448 → CB = 40
const CB_BITS: u32 = 40;

const DEEMPH_ALPHA: f64 = 0.1;
const PITCH_MEM_LEN: usize = MAX_PITCH as usize + SUBFRAME_SIZE; // 364

pub struct Ds2QpDecoder {
    lattice_state: [f64; NUM_COEFFS],
    pitch_memory: Vec<f64>,
    deemph_state: f64,
}

impl Default for Ds2QpDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Ds2QpDecoder {
    pub fn new() -> Self {
        Self {
            lattice_state: [0.0; NUM_COEFFS],
            pitch_memory: vec![0.0; PITCH_MEM_LEN],
            deemph_state: 0.0,
        }
    }

    /// Reset all decoder state to zero.
    ///
    /// Called automatically by `decode_segments()` before any segment that
    /// has `reset_before == true` (i.e. every segment after a cut point).
    pub fn reset(&mut self) {
        self.lattice_state = [0.0; NUM_COEFFS];
        self.pitch_memory.fill(0.0);
        self.deemph_state = 0.0;
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Decode all segments produced by the demuxer.
    ///
    /// Resets decoder state before every segment where `reset_before` is true.
    /// De-emphasis is applied per segment so the IIR tail cannot leak across
    /// cut points.
    ///
    /// This is the correct entry point for both uncut and cut files.
    pub fn decode_segments(&mut self, segments: &[QpSegment]) -> Vec<f64> {
        let total_cap: usize = segments
            .iter()
            .map(|s| s.frame_count * SAMPLES_PER_FRAME)
            .sum();
        let mut all_samples = Vec::with_capacity(total_cap);

        for seg in segments {
            if seg.reset_before {
                self.reset();
            }
            let samples = self.decode_all_frames(&seg.stream, seg.frame_count);
            all_samples.extend(samples);
        }

        all_samples
    }

    /// Decode `total_frames` QP frames from a raw continuous bitstream.
    ///
    /// De-emphasis is applied to the entire output before returning.
    /// `deemph_state` carries the IIR memory across calls within the same
    /// segment; call `reset()` between segments.
    pub fn decode_all_frames(&mut self, stream: &[u8], total_frames: usize) -> Vec<f64> {
        let mut reader = BitstreamReader::new(stream);
        let mut samples = Vec::with_capacity(total_frames * SAMPLES_PER_FRAME);

        for _ in 0..total_frames {
            let frame = self.decode_frame(&mut reader);
            samples.extend_from_slice(&frame);
        }

        // Apply de-emphasis: y[n] = x[n] + alpha * y[n-1]
        if !samples.is_empty() {
            samples[0] += DEEMPH_ALPHA * self.deemph_state;
            for i in 1..samples.len() {
                samples[i] += DEEMPH_ALPHA * samples[i - 1];
            }
            self.deemph_state = *samples.last().unwrap();
        }

        samples
    }

    // ── frame decode ──────────────────────────────────────────────────────────

    fn decode_frame(&mut self, reader: &mut BitstreamReader) -> Vec<f64> {
        // 1. Read 16 reflection coefficient indices.
        let mut refl_indices = [0usize; NUM_COEFFS];
        for (i, &bits) in REFL_BIT_ALLOC.iter().enumerate() {
            refl_indices[i] = reader.read_bits(bits) as usize;
        }

        // 2. Read per-subframe parameters.
        //    QP uses absolute per-subframe pitch: pitch = index + MIN_PITCH.
        let mut subframe_params = Vec::with_capacity(NUM_SUBFRAMES);
        let mut pitches = [0usize; NUM_SUBFRAMES];

        for sf in 0..NUM_SUBFRAMES {
            let pitch_idx = reader.read_bits(PITCH_BITS);
            let pg_idx = reader.read_bits(PITCH_GAIN_BITS) as usize;
            let cb_idx = reader.read_bits_u64(CB_BITS);
            let gain_idx = reader.read_bits(GAIN_BITS) as usize;
            let mut pulses = [0usize; EXCITATION_PULSES];
            for p in &mut pulses {
                *p = reader.read_bits(PULSE_BITS) as usize;
            }
            pitches[sf] = (pitch_idx + MIN_PITCH) as usize;
            subframe_params.push((pg_idx, cb_idx, gain_idx, pulses));
        }

        // 3. Dequantize reflection coefficients.
        let mut coeffs = [0.0f64; NUM_COEFFS];
        for i in 0..NUM_COEFFS {
            coeffs[i] = qp_codebook_lookup(i, refl_indices[i]);
        }

        // 4. Synthesise subframes.
        let mut frame_output = Vec::with_capacity(SAMPLES_PER_FRAME);

        for sf in 0..NUM_SUBFRAMES {
            let (pg_idx, cb_idx, gain_idx, pulses) = &subframe_params[sf];
            let pitch = pitches[sf];
            let gp = QP_PITCH_GAIN[*pg_idx];
            let gc = QP_EXCITATION_GAIN[*gain_idx];

            // 4a. Adaptive excitation: look back `pitch` samples in pitch memory.
            let mut adaptive_exc = [0.0f64; SUBFRAME_SIZE];
            let mem_len = self.pitch_memory.len();
            for i in 0..SUBFRAME_SIZE {
                let mem_idx = if pitch < SUBFRAME_SIZE {
                    // Short pitch: repeat cyclically within the subframe.
                    mem_len - pitch + (i % pitch)
                } else {
                    mem_len - pitch + i
                };
                if mem_idx < mem_len {
                    adaptive_exc[i] = self.pitch_memory[mem_idx];
                }
            }

            // 4b. Fixed codebook excitation: 11 pulses, C(64,11) codebook.
            let positions =
                decode_combinatorial_index(*cb_idx, SUBFRAME_SIZE, EXCITATION_PULSES);
            let mut fixed_exc = [0.0f64; SUBFRAME_SIZE];
            for (pi, &pos) in positions.iter().enumerate() {
                if pos < SUBFRAME_SIZE {
                    fixed_exc[pos] += QP_PULSE_AMP[pulses[pi]] * gc;
                }
            }

            // 4c. Combine.
            let mut excitation = [0.0f64; SUBFRAME_SIZE];
            for i in 0..SUBFRAME_SIZE {
                excitation[i] = gp * adaptive_exc[i] + fixed_exc[i];
            }

            // 4d. Lattice synthesis filter.
            let output =
                lattice_synthesis(&excitation, &coeffs, &mut self.lattice_state);

            // 4e. Update pitch memory: shift left, append new excitation.
            //     The Olympus QP codec uses open-loop pitch prediction on the
            //     excitation domain — pitch_memory stores excitation, not output.
            self.pitch_memory.copy_within(SUBFRAME_SIZE..mem_len, 0);
            let tail = mem_len - SUBFRAME_SIZE;
            self.pitch_memory[tail..].copy_from_slice(&excitation);

            frame_output.extend_from_slice(&output);
        }

        frame_output
    }
}
