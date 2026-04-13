use clap::Parser;
use dss_codec::demux::detect_format;
use dss_codec::demux::ds2::ExtractionMode;
use dss_codec::output::OutputConfig;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "dss-decode", about = "Decode DSS/DS2 audio files")]
struct Cli {
    /// Input file(s)
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// Output file (single input) or ignored (batch mode)
    #[arg(short = 'O', long)]
    output_file: Option<PathBuf>,

    /// Output format
    #[arg(short = 'f', long, default_value = "wav")]
    format: String,

    /// Output sample rate (default: native)
    #[arg(short = 'r', long)]
    rate: Option<u32>,

    /// Bit depth
    #[arg(short = 'b', long, default_value = "16")]
    bits: u16,

    /// Channels (1=mono, 2=stereo)
    #[arg(short = 'c', long, default_value = "1")]
    channels: u16,

    /// Batch output directory
    #[arg(short = 'o', long)]
    output_dir: Option<PathBuf>,

    /// What to extract from DS2 QP files that contain annotations:
    ///   all          — everything in original order (default)
    ///   main         — main dictation only (skip annotations)
    ///   annotations  — annotations only (skip main dictation)
    #[arg(
        short = 'e',
        long = "extract",
        value_name = "CONTENT",
        default_value = "all",
        verbatim_doc_comment
    )]
    extract: ExtractArg,

    /// Suppress output
    #[arg(short = 'q', long)]
    quiet: bool,

    /// Print file metadata only
    #[arg(long)]
    info: bool,
}

/// Clap value type for --extract.
#[derive(Clone, Copy, Debug)]
enum ExtractArg {
    All,
    Main,
    Annotations,
}

impl std::str::FromStr for ExtractArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "all"         => Ok(ExtractArg::All),
            "main"        => Ok(ExtractArg::Main),
            "annotations" => Ok(ExtractArg::Annotations),
            other => Err(format!(
                "unknown value '{}'; expected: all, main, annotations",
                other
            )),
        }
    }
}

impl From<ExtractArg> for ExtractionMode {
    fn from(a: ExtractArg) -> Self {
        match a {
            ExtractArg::All         => ExtractionMode::All,
            ExtractArg::Main        => ExtractionMode::MainOnly,
            ExtractArg::Annotations => ExtractionMode::AnnotationsOnly,
        }
    }
}

fn main() {
    let cli = Cli::parse();

    if cli.info {
        for path in &cli.input {
            print_info(path, cli.quiet);
        }
        return;
    }

    let config = OutputConfig {
        sample_rate: cli.rate,
        bit_depth: cli.bits,
        channels: cli.channels,
    };

    let mode: ExtractionMode = cli.extract.into();

    for input_path in &cli.input {
        let output_path = if let Some(ref out) = cli.output_file {
            if cli.input.len() == 1 {
                out.clone()
            } else {
                make_output_path(input_path, cli.output_dir.as_deref(), &cli.format)
            }
        } else {
            make_output_path(input_path, cli.output_dir.as_deref(), &cli.format)
        };

        if !cli.quiet {
            eprintln!("Decoding: {}", input_path.display());
        }

        match dss_codec::decode_and_write_ex(input_path, &output_path, &config, mode) {
            Ok(buf) => {
                if !cli.quiet {
                    let duration = buf.samples.len() as f64 / buf.native_rate as f64;
                    eprintln!(
                        "  {} → {} ({:.1}s, {} Hz, {:?})",
                        input_path.display(),
                        output_path.display(),
                        duration,
                        buf.native_rate,
                        buf.format,
                    );
                }
            }
            Err(e) => {
                eprintln!("Error decoding {}: {}", input_path.display(), e);
                std::process::exit(1);
            }
        }
    }
}

fn make_output_path(
    input: &PathBuf,
    output_dir: Option<&std::path::Path>,
    ext: &str,
) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default();
    let filename = format!("{}.{}", stem.to_string_lossy(), ext);
    if let Some(dir) = output_dir {
        dir.join(filename)
    } else {
        input.with_file_name(filename)
    }
}

fn print_info(path: &PathBuf, _quiet: bool) {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading {}: {}", path.display(), e);
            return;
        }
    };
    match detect_format(&data) {
        Some(fmt) => {
            println!(
                "{}: {:?}, native rate {} Hz",
                path.display(),
                fmt,
                fmt.native_sample_rate()
            );
        }
        None => {
            println!("{}: unknown format", path.display());
        }
    }
}