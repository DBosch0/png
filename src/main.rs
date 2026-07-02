use std::{
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::LazyLock,
};

use crate::zlib::decompress;
mod zlib;

#[derive(Debug)]
struct IHDRChunk {
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    color_type: ColorType,
    interlace_method: InterlaceMethod,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Palette {
    r: u8,
    g: u8,
    b: u8,
}

#[derive(Debug)]
struct PLTEChunk {
    palettes: Vec<Palette>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BKGDChunk {
    PalleteIndex(u8),
    Gray(u16),
    Color { r: u16, g: u16, b: u16 },
}

impl BKGDChunk {
    fn read_bytes<R: Read>(
        read: &mut R,
        length: usize,
        color_type: ColorType,
    ) -> std::io::Result<Self> {
        match color_type {
            ColorType::IndexedColor => {
                assert!(
                    length == 1,
                    "invalid bKGD length for color_type {:?}",
                    color_type
                );
                Ok(Self::PalleteIndex(read_u8(read)?))
            }
            ColorType::Grayscale | ColorType::GrayScaleAlpha => {
                assert!(
                    length == 2,
                    "invalid bKGD length for color_type {:?}",
                    color_type
                );
                Ok(Self::Gray(read_u16(read)?))
            }
            ColorType::TrueColor | ColorType::TrueColorAlpha => {
                assert!(
                    length == 6,
                    "invalid bKGD length for color_type {:?}",
                    color_type
                );
                Ok(Self::Color {
                    r: read_u16(read)?,
                    g: read_u16(read)?,
                    b: read_u16(read)?,
                })
            }
        }
    }
}

/// Per-color-type transparency data. Grayscale/TrueColor mark a single exact
/// sample value/triplet as fully transparent (everything else fully opaque);
/// IndexedColor instead gives a per-palette-entry alpha, possibly shorter
/// than PLTE (entries past the end default to fully opaque). Not valid for
/// GrayScaleAlpha/TrueColorAlpha, which already carry a real alpha channel.
#[derive(Debug, Clone, PartialEq)]
enum TRNSChunk {
    Gray(u16),
    TrueColor { r: u16, g: u16, b: u16 },
    Indexed(Vec<u8>),
}

impl TRNSChunk {
    fn read_bytes<R: Read>(
        read: &mut R,
        length: usize,
        color_type: ColorType,
    ) -> std::io::Result<Self> {
        match color_type {
            ColorType::Grayscale => {
                assert!(
                    length == 2,
                    "invalid tRNS chunk length for color_type {:?}",
                    color_type
                );
                Ok(Self::Gray(read_u16(read)?))
            }
            ColorType::TrueColor => {
                assert!(
                    length == 6,
                    "invalid tRNS chunk length for color_type {:?}",
                    color_type
                );
                Ok(Self::TrueColor {
                    r: read_u16(read)?,
                    g: read_u16(read)?,
                    b: read_u16(read)?,
                })
            }
            ColorType::IndexedColor => {
                let mut alphas = vec![0u8; length];
                read.read_exact(&mut alphas)?;
                Ok(Self::Indexed(alphas))
            }
            ColorType::GrayScaleAlpha | ColorType::TrueColorAlpha => {
                unreachable!("tRNS chunk is not allowed for color_type {:?}", color_type)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SBITChunk {
    GrayScale(u8),
    TrueColor([u8; 3]),
    IndexedColor([u8; 3]),
    GrayScaleAlpha([u8; 2]),
    TrueColorAlpha([u8; 4]),
}

impl SBITChunk {
    fn read_bytes<R: Read>(
        read: &mut R,
        length: usize,
        color_type: ColorType,
    ) -> std::io::Result<Self> {
        match color_type {
            ColorType::Grayscale => {
                assert!(
                    length == 1,
                    "invalid sBIT chunk length for color_type {:?}",
                    color_type
                );
                Ok(Self::GrayScale(read_u8(read)?))
            }
            ColorType::TrueColor => {
                assert!(
                    length == 3,
                    "invalid sBIT chunk length for color_type {:?}",
                    color_type
                );
                Ok(Self::TrueColor([
                    read_u8(read)?,
                    read_u8(read)?,
                    read_u8(read)?,
                ]))
            }
            ColorType::IndexedColor => {
                assert!(
                    length == 3,
                    "invalid sBIT chunk length for color_type {:?}",
                    color_type
                );
                Ok(Self::IndexedColor([
                    read_u8(read)?,
                    read_u8(read)?,
                    read_u8(read)?,
                ]))
            }
            ColorType::GrayScaleAlpha => {
                assert!(
                    length == 2,
                    "invalid sBIT chunk length for color_type {:?}",
                    color_type
                );
                Ok(Self::GrayScaleAlpha([read_u8(read)?, read_u8(read)?]))
            }
            ColorType::TrueColorAlpha => {
                assert!(
                    length == 4,
                    "invalid sBIT chunk length for color_type {:?}",
                    color_type
                );
                Ok(Self::TrueColorAlpha([
                    read_u8(read)?,
                    read_u8(read)?,
                    read_u8(read)?,
                    read_u8(read)?,
                ]))
            }
        }
    }
}

static CRC_TABLE: LazyLock<[u32; 256]> = LazyLock::new(|| {
    let mut table = [0; 256];
    for n in 0..256 {
        let mut c = n as u32;
        for _ in 0..8 {
            if c & 1 != 0 {
                c = 0xedb88320 ^ (c >> 1);
            } else {
                c = c >> 1;
            }
        }
        table[n] = c;
    }
    table
});

fn compute_crc(buf: &[u8]) -> u32 {
    fn update_crc(crc: u32, buf: &[u8]) -> u32 {
        let mut c = crc;
        for n in 0..buf.len() {
            c = CRC_TABLE[((c ^ buf[n] as u32) & 0xff as u32) as usize] ^ (c >> 8);
        }
        c
    }
    update_crc(0xffffffff, buf) ^ 0xffffffff
}

fn read_u32<R: Read>(read: &mut R) -> std::io::Result<u32> {
    let mut out = [0; 4];
    read.read_exact(&mut out)?;
    Ok(u32::from_be_bytes(out))
}

fn read_u8<R: Read>(read: &mut R) -> std::io::Result<u8> {
    let mut out = [0; 1];
    read.read_exact(&mut out)?;
    Ok(u8::from_be_bytes(out))
}

fn read_u16<R: Read>(read: &mut R) -> std::io::Result<u16> {
    let mut out = [0; 2];
    read.read_exact(&mut out)?;
    Ok(u16::from_be_bytes(out))
}

fn read_chunk_type<R: Read>(read: &mut R) -> std::io::Result<[u8; 4]> {
    let mut chunk_type = [0u8; 4];
    read.read_exact(&mut chunk_type)?;
    assert!(
        chunk_type
            .iter()
            .all(|b| (b'a'..=b'z').contains(b) || (b'A'..=b'Z').contains(b))
    );
    Ok(chunk_type)
}

/// Reads one PNG chunk: a 4-byte length, its 4-byte type, that many bytes of
/// chunk data, and the trailing CRC-32, which is verified against the type
/// and data before the chunk is handed back.
fn read_chunk<R: Read>(read: &mut R) -> std::io::Result<([u8; 4], Vec<u8>)> {
    let length = read_u32(read)?;
    let chunk_type = read_chunk_type(read)?;
    let mut chunk_data = vec![0u8; length as usize];
    read.read_exact(&mut chunk_data)?;
    let crc = read_u32(read)?;

    let mut crc_input = Vec::with_capacity(4 + chunk_data.len());
    crc_input.extend_from_slice(&chunk_type);
    crc_input.extend_from_slice(&chunk_data);
    assert!(
        compute_crc(&crc_input) == crc,
        "CRC mismatch in {:?} chunk",
        str::from_utf8(&chunk_type).unwrap_or("<invalid utf8>")
    );

    Ok((chunk_type, chunk_data))
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BitDepth {
    B1,
    B2,
    B4,
    B8,
    B16,
}

impl From<u8> for BitDepth {
    fn from(value: u8) -> Self {
        match value {
            1 => Self::B1,
            2 => Self::B2,
            4 => Self::B4,
            8 => Self::B8,
            16 => Self::B16,
            _ => unreachable!("invalid bit depth, expected 1, 2, 4, 8, 16, got {value}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ColorType {
    Grayscale,
    TrueColor,
    IndexedColor,
    GrayScaleAlpha,
    TrueColorAlpha,
}

impl From<u8> for ColorType {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Grayscale,
            2 => Self::TrueColor,
            3 => Self::IndexedColor,
            4 => Self::GrayScaleAlpha,
            6 => Self::TrueColorAlpha,
            _ => unreachable!("invalid color type value, expected 0, 2, 3, 4, 6, got {value}"),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
enum InterlaceMethod {
    None,
    Adam7,
}
impl From<u8> for InterlaceMethod {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Adam7,
            _ => unreachable!("invalid interlace method, expected 0 or 1, got {value}"),
        }
    }
}

impl IHDRChunk {
    fn from_bytes<R: Read>(read: &mut R) -> std::io::Result<Self> {
        let width = read_u32(read)?;
        let height = read_u32(read)?;
        let bit_depth = BitDepth::from(read_u8(read)?);
        let color_type = ColorType::from(read_u8(read)?);
        let compression_method = read_u8(read)?;
        let filter_method = read_u8(read)?;
        let interlace_method = InterlaceMethod::from(read_u8(read)?);

        assert!(width <= i32::MAX as u32, "width too large");
        assert!(height <= i32::MAX as u32, "height too large");

        match color_type {
            ColorType::Grayscale => {}
            ColorType::TrueColor => assert!(
                matches!(bit_depth, BitDepth::B8 | BitDepth::B16),
                "invalid bit depth for TrueColor Color type, expected 8 or 16, got {:?}",
                bit_depth
            ),
            ColorType::IndexedColor => assert!(
                matches!(
                    bit_depth,
                    BitDepth::B1 | BitDepth::B2 | BitDepth::B4 | BitDepth::B8
                ),
                "invalid bit depth for IndexedColor Color type, expected 1, 2, 4, 8, got {:?}",
                bit_depth
            ),
            ColorType::GrayScaleAlpha => assert!(
                matches!(bit_depth, BitDepth::B8 | BitDepth::B16),
                "invalid bit depth for GrayScaleAlpha Color type, expected 8 or 16, got {:?}",
                bit_depth
            ),
            ColorType::TrueColorAlpha => assert!(
                matches!(bit_depth, BitDepth::B8 | BitDepth::B16),
                "invalid bit depth for TrueColorAlpha Color type, expected 8 or 16, got {:?}",
                bit_depth
            ),
        }
        assert!(
            compression_method == 0,
            "unsupported compression mode, expected 0, got {compression_method}"
        );

        assert!(
            filter_method == 0,
            "only filter method 0 is supported, got {filter_method}"
        );

        Ok(IHDRChunk {
            width,
            height,
            bit_depth,
            color_type,
            interlace_method,
        })
    }
}

impl PLTEChunk {
    fn from_bytes<R: Read>(read: &mut R, length: usize) -> std::io::Result<Self> {
        assert!(length % 3 == 0, "PLTEChunk length must be divisible by 3");
        let mut palettes = Vec::new();
        for _ in 0..length / 3 {
            palettes.push(Palette {
                r: read_u8(read)?,
                g: read_u8(read)?,
                b: read_u8(read)?,
            });
        }
        Ok(Self { palettes })
    }
}

struct Png {
    header: IHDRChunk,
    palette: Option<PLTEChunk>,
    gamma: Option<f64>,
    sbit: Option<SBITChunk>,
    bkgd: Option<BKGDChunk>,
    trns: Option<TRNSChunk>,
    /// Fully reconstructed (unfiltered, deinterlaced, bit-unpacked) samples in
    /// row-major order, `raw_channels(header.color_type)` values per pixel.
    /// For IndexedColor this holds raw palette indices, not resolved RGB.
    samples: Vec<u16>,
}

impl Debug for Png {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Png")
            .field("header", &self.header)
            .field("palette", &self.palette)
            .field("gamma", &self.gamma)
            .field("sbit", &self.sbit)
            .field("bkgd", &self.bkgd)
            .field("trns", &self.trns)
            .finish()
    }
}

impl Png {
    fn read_png<R: BufRead>(read: &mut R) -> std::io::Result<Self> {
        let mut magic_bytes = [0u8; 8];
        read.read_exact(&mut magic_bytes)?;
        assert!(
            magic_bytes == [137, 80, 78, 71, 13, 10, 26, 10],
            "Invalid png file, magic bytes incorrect"
        );

        //header chunk
        let header = {
            let (chunk_type, chunk_data) = read_chunk(read)?;
            assert!(&chunk_type == b"IHDR", "png must start with header");
            IHDRChunk::from_bytes(&mut chunk_data.as_slice())?
        };

        let mut palette: Option<PLTEChunk> = None;
        let mut data: Vec<u8> = Vec::new();
        let mut gamma: Option<f64> = None;
        let mut sbit: Option<SBITChunk> = None;
        let mut bkgd: Option<BKGDChunk> = None;
        let mut trns: Option<TRNSChunk> = None;

        loop {
            let (chunk_type, chunk_data) = read_chunk(read)?;
            match &chunk_type {
                b"IHDR" => unreachable!("png can only have 1 header chunk"),
                b"gAMA" => {
                    assert!(palette.is_none(), "gAMA chunk must precede PLTE data");
                    assert!(data.is_empty(), "gAMA chunk must precede IDAT data");
                    let gamma_u32 = read_u32(&mut chunk_data.as_slice())?;
                    gamma = Some(gamma_u32 as f64 / 100000.0);
                }
                b"sBIT" => {
                    assert!(palette.is_none(), "sBIT chunk must precede PLTE data");
                    assert!(data.is_empty(), "sBIT chunk must precede IDAT data");
                    sbit = Some(SBITChunk::read_bytes(
                        &mut chunk_data.as_slice(),
                        chunk_data.len(),
                        header.color_type,
                    )?);
                }
                b"bKGD" => {
                    if header.color_type == ColorType::IndexedColor {
                        assert!(palette.is_some(), "PLTE chunk must precede bKGD chunk");
                    }
                    assert!(data.is_empty(), "bKGD chunk must precede IDAT data");
                    bkgd = Some(BKGDChunk::read_bytes(
                        &mut chunk_data.as_slice(),
                        chunk_data.len(),
                        header.color_type,
                    )?)
                }
                b"tRNS" => {
                    if header.color_type == ColorType::IndexedColor {
                        assert!(palette.is_some(), "PLTE chunk must precede tRNS chunk");
                    }
                    assert!(data.is_empty(), "tRNS chunk must precede IDAT data");
                    trns = Some(TRNSChunk::read_bytes(
                        &mut chunk_data.as_slice(),
                        chunk_data.len(),
                        header.color_type,
                    )?)
                }
                b"PLTE" => {
                    if palette.is_none() {
                        palette = Some(PLTEChunk::from_bytes(
                            &mut chunk_data.as_slice(),
                            chunk_data.len(),
                        )?)
                    } else {
                        unreachable!("png can only have 1 palette chunk");
                    }
                }
                b"IDAT" => {
                    data.extend(chunk_data);
                }
                b"IEND" => break,
                x => unimplemented!(
                    "unimplemented chunk type, {}",
                    str::from_utf8(x).expect("valid ascii string")
                ),
            }
        }

        let n = read.read(&mut [0])?;
        assert!(n == 0, "png has data remaing after IEND chunk");

        let (inflated, _checksum) = decompress(&data, zlib::Format::Zlib).unwrap();
        let raw_channels = Self::raw_channels(header.color_type);
        let samples = match header.interlace_method {
            InterlaceMethod::None => Self::decode_pass(
                header.bit_depth,
                header.color_type,
                raw_channels,
                header.width,
                header.height,
                &inflated,
            ),
            InterlaceMethod::Adam7 => Self::decode_adam7(&header, raw_channels, &inflated),
        };

        Ok(Png {
            header,
            palette,
            gamma,
            sbit,
            bkgd,
            trns,
            samples,
        })
    }

    /// Number of raw samples per pixel as they appear in the bitstream, i.e.
    /// before IndexedColor is resolved through the palette.
    fn raw_channels(color_type: ColorType) -> usize {
        match color_type {
            ColorType::Grayscale => 1,
            ColorType::TrueColor => 3,
            ColorType::IndexedColor => 1,
            ColorType::GrayScaleAlpha => 2,
            ColorType::TrueColorAlpha => 4,
        }
    }

    fn scanline_length_for(bit_depth: BitDepth, color_type: ColorType, width: u32) -> usize {
        let mut scan_line_len = match bit_depth {
            BitDepth::B1 => (width + 7) / 8,
            BitDepth::B2 => (width + 3) / 4,
            BitDepth::B4 => (width + 1) / 2,
            BitDepth::B8 => width,
            BitDepth::B16 => width * 2,
        };
        scan_line_len *= Self::raw_channels(color_type) as u32;
        scan_line_len as usize + 1 //1 added for the filter byte.
    }

    fn get_bpp_for(bit_depth: BitDepth, color_type: ColorType) -> usize {
        match color_type {
            ColorType::Grayscale => match bit_depth {
                BitDepth::B1 | BitDepth::B2 | BitDepth::B4 | BitDepth::B8 => 1,
                BitDepth::B16 => 2,
            },
            ColorType::TrueColor => match bit_depth {
                BitDepth::B8 => 3,
                BitDepth::B16 => 6,
                _ => unreachable!(),
            },
            ColorType::IndexedColor => match bit_depth {
                // a pixel is a single palette-index sample, so bpp is always 1 byte
                // (rounding up), regardless of how many bits that sample uses.
                BitDepth::B1 | BitDepth::B2 | BitDepth::B4 | BitDepth::B8 => 1,
                _ => unreachable!(),
            },
            ColorType::GrayScaleAlpha => match bit_depth {
                BitDepth::B8 => 2,
                BitDepth::B16 => 4,
                _ => unreachable!(),
            },
            ColorType::TrueColorAlpha => match bit_depth {
                BitDepth::B8 => 4,
                BitDepth::B16 => 8,
                _ => unreachable!(),
            },
        }
    }

    /// Reverses scanline filtering for a single width x height image (or, for
    /// an interlaced PNG, a single Adam7 pass), returning the reconstructed
    /// pixel bytes with the per-row filter-type bytes stripped out.
    fn unfilter_pass(
        bit_depth: BitDepth,
        color_type: ColorType,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Vec<u8> {
        let scan_line_len = Self::scanline_length_for(bit_depth, color_type, width);
        let stride = scan_line_len - 1; // pixel bytes per row, filter-type byte excluded
        let bpp = Self::get_bpp_for(bit_depth, color_type);
        assert!(data.len() == scan_line_len * height as usize);

        let mut output = Vec::with_capacity(stride * height as usize);

        for row_i in 0..height as usize {
            let row = &data[row_i * scan_line_len..(row_i + 1) * scan_line_len];
            let filter_type = row[0];
            let row = &row[1..];

            for i in 0..row.len() {
                // a = left, b = up, c = upper-left, all already-reconstructed bytes.
                let a = if i >= bpp {
                    output[output.len() - bpp]
                } else {
                    0
                };
                let b = if row_i > 0 {
                    output[output.len() - stride]
                } else {
                    0
                };
                let c = if row_i > 0 && i >= bpp {
                    output[output.len() - stride - bpp]
                } else {
                    0
                };

                let recon = match filter_type {
                    0 => row[i],
                    1 => row[i].wrapping_add(a),
                    2 => row[i].wrapping_add(b),
                    3 => row[i].wrapping_add(((a as u16 + b as u16) / 2) as u8),
                    4 => row[i].wrapping_add(Self::paeth_predictor(a, b, c)),
                    _ => unreachable!("invalid filter type, got {filter_type}"),
                };
                output.push(recon);
            }
        }

        output
    }

    /// Unfilters and bit-unpacks one width x height image worth of scanline
    /// data (the whole image for non-interlaced PNGs, or a single Adam7 pass)
    /// into row-major `u16` samples, `channels` per pixel.
    fn decode_pass(
        bit_depth: BitDepth,
        color_type: ColorType,
        channels: usize,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Vec<u16> {
        let packed = Self::unfilter_pass(bit_depth, color_type, width, height, data);
        let stride = Self::scanline_length_for(bit_depth, color_type, width) - 1;
        packed
            .chunks_exact(stride)
            .flat_map(|row| Self::unpack_channels(row, bit_depth, channels, width as usize))
            .collect()
    }

    /// (x_start, y_start, x_step, y_step) for each of the 7 Adam7 passes.
    const ADAM7_PASSES: [(u32, u32, u32, u32); 7] = [
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    ];

    /// Width/height of the sub-image sampled by one Adam7 pass; either may be
    /// 0 when the full image is smaller than the pass's starting offset.
    fn adam7_pass_dimensions(
        width: u32,
        height: u32,
        x_start: u32,
        y_start: u32,
        x_step: u32,
        y_step: u32,
    ) -> (u32, u32) {
        let pass_width = width.saturating_sub(x_start).div_ceil(x_step);
        let pass_height = height.saturating_sub(y_start).div_ceil(y_step);
        (pass_width, pass_height)
    }

    /// Decodes an Adam7-interlaced IDAT stream: unfilters and unpacks each of
    /// the 7 passes independently, then scatters their pixels into a
    /// full-resolution, row-major `u16` sample buffer.
    fn decode_adam7(header: &IHDRChunk, raw_channels: usize, inflated: &[u8]) -> Vec<u16> {
        let width = header.width as usize;
        let height = header.height as usize;
        let mut samples = vec![0u16; width * height * raw_channels];
        let mut cursor = 0usize;

        for &(x_start, y_start, x_step, y_step) in &Self::ADAM7_PASSES {
            let (pass_width, pass_height) = Self::adam7_pass_dimensions(
                header.width,
                header.height,
                x_start,
                y_start,
                x_step,
                y_step,
            );
            if pass_width == 0 || pass_height == 0 {
                continue;
            }

            let pass_bytes =
                Self::scanline_length_for(header.bit_depth, header.color_type, pass_width)
                    * pass_height as usize;
            let pass_data = &inflated[cursor..cursor + pass_bytes];
            cursor += pass_bytes;

            let pass_samples = Self::decode_pass(
                header.bit_depth,
                header.color_type,
                raw_channels,
                pass_width,
                pass_height,
                pass_data,
            );

            for row in 0..pass_height as usize {
                for col in 0..pass_width as usize {
                    let dst_x = x_start as usize + col * x_step as usize;
                    let dst_y = y_start as usize + row * y_step as usize;
                    let dst_idx = (dst_y * width + dst_x) * raw_channels;
                    let src_idx = (row * pass_width as usize + col) * raw_channels;
                    samples[dst_idx..dst_idx + raw_channels]
                        .copy_from_slice(&pass_samples[src_idx..src_idx + raw_channels]);
                }
            }
        }

        assert!(
            cursor == inflated.len(),
            "unexpected leftover Adam7 pass data"
        );
        samples
    }

    /// Number of channels/samples per pixel that `pixels()` emits.
    /// IndexedColor counts as 3 because `pixels` resolves each index through
    /// the palette into an RGB triple. A tRNS chunk adds one synthesized
    /// alpha channel for color types that don't already carry one.
    fn channels(&self) -> usize {
        let base = match self.header.color_type {
            ColorType::Grayscale => 1,
            ColorType::GrayScaleAlpha => 2,
            ColorType::TrueColor | ColorType::IndexedColor => 3,
            ColorType::TrueColorAlpha => 4,
        };
        base + if self.trns.is_some() { 1 } else { 0 }
    }

    /// Unpacks `width` pixels' worth of channel samples (`channels` per
    /// pixel) from one scanline's bytes, per `bit_depth`, widening every
    /// sample to `u16`. Bit depths below 8 only ever occur with a single
    /// channel (Grayscale, IndexedColor). Rows are byte-aligned, so for bit
    /// depths < 8 the last byte of a row may hold unused padding bits that
    /// must be dropped rather than read as extra samples.
    fn unpack_channels(row: &[u8], bit_depth: BitDepth, channels: usize, width: usize) -> Vec<u16> {
        match bit_depth {
            BitDepth::B1 | BitDepth::B2 | BitDepth::B4 => {
                assert!(channels == 1, "sub-byte bit depths only support 1 channel");
                let bits = match bit_depth {
                    BitDepth::B1 => 1,
                    BitDepth::B2 => 2,
                    BitDepth::B4 => 4,
                    _ => unreachable!(),
                };
                let samples_per_byte = 8 / bits;
                row.iter()
                    .flat_map(|&byte| {
                        (0..samples_per_byte).map(move |i| {
                            ((byte >> (8 - bits * (i + 1))) & ((1 << bits) - 1)) as u16
                        })
                    })
                    .take(width)
                    .collect()
            }
            BitDepth::B8 => row[..width * channels].iter().map(|&b| b as u16).collect(),
            BitDepth::B16 => row[..width * channels * 2]
                .chunks_exact(2)
                .map(|c| u16::from_be_bytes([c[0], c[1]]))
                .collect(),
        }
    }

    /// Assumed display gamma (roughly that of a CRT) used to gamma-correct
    /// stored samples per the file's gAMA chunk. PAM/PGM/PPM output carries
    /// no gamma metadata of its own, so without this, images that encode the
    /// same picture under different gAMA tags (as PngSuite's g0x* files do)
    /// would decode to visibly different sample values.
    const DISPLAY_GAMMA: f64 = 2.2;

    /// Gamma-corrects one color sample (never alpha, which is always
    /// linear) so that `sample^(1/(file_gamma * DISPLAY_GAMMA))` maps the
    /// stored value to the intensity a `DISPLAY_GAMMA`-gamma display would
    /// need to reproduce the original linear light.
    fn gamma_correct(sample: u16, max: u16, file_gamma: f64) -> u16 {
        let normalized = sample as f64 / max as f64;
        let corrected = normalized.powf(1.0 / (file_gamma * Self::DISPLAY_GAMMA));
        (corrected * max as f64).round().clamp(0.0, max as f64) as u16
    }

    /// Decodes this image into a flat, row-major array of pixel samples,
    /// `channels()` values per pixel, all widened to `u16` regardless of the
    /// source bit depth. IndexedColor samples are resolved through the
    /// palette into RGB triples rather than left as raw indices. A tRNS
    /// chunk (Grayscale/TrueColor/IndexedColor only) appends a synthesized
    /// alpha sample per pixel: fully transparent (0) for the one color/index
    /// tRNS designates as transparent, fully opaque otherwise. A gAMA chunk
    /// gamma-corrects every color sample (palette entries included), leaving
    /// alpha untouched.
    fn pixels(&self) -> Vec<u16> {
        let max = Self::max_sample_value(self.header.bit_depth);
        let correct = |v: u16| match self.gamma {
            Some(file_gamma) => Self::gamma_correct(v, max, file_gamma),
            None => v,
        };

        match self.header.color_type {
            ColorType::IndexedColor => {
                let palette = &self
                    .palette
                    .as_ref()
                    .expect("IndexedColor image must have a PLTE chunk")
                    .palettes;
                // Palette entries are always 8-bit RGB, regardless of the
                // index bit depth, so gamma-correct them against max=255.
                let correct_palette = |v: u8| match self.gamma {
                    Some(file_gamma) => Self::gamma_correct(v as u16, 255, file_gamma) as u8,
                    None => v,
                };
                match &self.trns {
                    Some(TRNSChunk::Indexed(alphas)) => self
                        .samples
                        .iter()
                        .flat_map(|&index| {
                            let Palette { r, g, b } = palette[index as usize];
                            let a = alphas.get(index as usize).copied().unwrap_or(255) as u16;
                            [
                                correct_palette(r) as u16,
                                correct_palette(g) as u16,
                                correct_palette(b) as u16,
                                a,
                            ]
                        })
                        .collect(),
                    _ => self
                        .samples
                        .iter()
                        .flat_map(|&index| {
                            let Palette { r, g, b } = palette[index as usize];
                            [
                                correct_palette(r) as u16,
                                correct_palette(g) as u16,
                                correct_palette(b) as u16,
                            ]
                        })
                        .collect(),
                }
            }
            ColorType::Grayscale => match &self.trns {
                Some(TRNSChunk::Gray(transparent)) => self
                    .samples
                    .iter()
                    .flat_map(|&v| [correct(v), if v == *transparent { 0 } else { max }])
                    .collect(),
                _ => self.samples.iter().map(|&v| correct(v)).collect(),
            },
            ColorType::TrueColor => match &self.trns {
                Some(TRNSChunk::TrueColor { r, g, b }) => self
                    .samples
                    .chunks_exact(3)
                    .flat_map(|px| {
                        let a = if px[0] == *r && px[1] == *g && px[2] == *b {
                            0
                        } else {
                            max
                        };
                        [correct(px[0]), correct(px[1]), correct(px[2]), a]
                    })
                    .collect(),
                _ => self
                    .samples
                    .chunks_exact(3)
                    .flat_map(|px| [correct(px[0]), correct(px[1]), correct(px[2])])
                    .collect(),
            },
            ColorType::GrayScaleAlpha => self
                .samples
                .chunks_exact(2)
                .flat_map(|px| [correct(px[0]), px[1]])
                .collect(),
            ColorType::TrueColorAlpha => self
                .samples
                .chunks_exact(4)
                .flat_map(|px| [correct(px[0]), correct(px[1]), correct(px[2]), px[3]])
                .collect(),
        }
    }

    /// Maximum sample value representable at `bit_depth` (i.e. `2^bits - 1`),
    /// used as the PAM header's MAXVAL.
    fn max_sample_value(bit_depth: BitDepth) -> u16 {
        match bit_depth {
            BitDepth::B1 => 1,
            BitDepth::B2 => 3,
            BitDepth::B4 => 15,
            BitDepth::B8 => 255,
            BitDepth::B16 => 65535,
        }
    }

    fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
        let (a, b, c) = (a as i32, b as i32, c as i32);
        let p = a + b - c;
        let pa = (p - a).abs();
        let pb = (p - b).abs();
        let pc = (p - c).abs();
        if pa <= pb && pa <= pc {
            a as u8
        } else if pb <= pc {
            b as u8
        } else {
            c as u8
        }
    }

    /// Writes this image as a binary PAM (P7, netpbm's arbitrary-depth
    /// format). Unlike PGM/PPM, PAM's DEPTH/TUPLTYPE header fields let one
    /// writer cover every ColorType, alpha channels included.
    fn write_to_pam(&self, filepath: &Path) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(filepath)?);
        let channels = self.channels();
        let maxval = match self.header.color_type {
            // Palette entries are always 8-bit RGB,wq regardless of index bit depth.
            ColorType::IndexedColor => 255,
            _ => Self::max_sample_value(self.header.bit_depth),
        };
        let has_alpha = self.trns.is_some()
            || matches!(
                self.header.color_type,
                ColorType::GrayScaleAlpha | ColorType::TrueColorAlpha
            );
        let tuple_type = match (self.header.color_type, has_alpha) {
            (ColorType::Grayscale | ColorType::GrayScaleAlpha, false) => "GRAYSCALE",
            (ColorType::Grayscale | ColorType::GrayScaleAlpha, true) => "GRAYSCALE_ALPHA",
            (ColorType::TrueColor | ColorType::IndexedColor | ColorType::TrueColorAlpha, false) => {
                "RGB"
            }
            (ColorType::TrueColor | ColorType::IndexedColor | ColorType::TrueColorAlpha, true) => {
                "RGB_ALPHA"
            }
        };

        writeln!(file, "P7")?;
        writeln!(file, "WIDTH {}", self.header.width)?;
        writeln!(file, "HEIGHT {}", self.header.height)?;
        writeln!(file, "DEPTH {channels}")?;
        writeln!(file, "MAXVAL {maxval}")?;
        writeln!(file, "TUPLTYPE {tuple_type}")?;
        writeln!(file, "ENDHDR")?;

        let samples = self.pixels();
        let bytes: Vec<u8> = if maxval > 255 {
            samples.iter().flat_map(|v| v.to_be_bytes()).collect()
        } else {
            samples.iter().map(|&v| v as u8).collect()
        };
        file.write_all(&bytes)?;

        Ok(())
    }
}

fn main() -> std::io::Result<()> {
    let test_files = File::open("tests.txt")?;
    let reader = BufReader::new(test_files);
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let path = format!("test/{line}.png");
        dbg!(&path);
        let f = File::open(path)?;
        let mut png_reader = BufReader::new(f);
        let png = Png::read_png(&mut png_reader)?;
        dbg!(&png);
        let mut output_path = PathBuf::new();
        output_path.push("out");
        output_path.push(format!("{}.pam", line));
        png.write_to_pam(&output_path)?;
    }

    Ok(())
}
