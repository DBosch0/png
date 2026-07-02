use std::{
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader, Read, Write},
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
#[derive(Debug)]
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

        assert!(
            matches!(interlace_method, InterlaceMethod::None),
            "Adam7 interlacing is not yet supported"
        );

        Ok(IHDRChunk {
            width,
            height,
            bit_depth,
            color_type,
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
    data: Vec<u8>,
}

impl Debug for Png {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Png")
            .field("header", &self.header)
            .field("palette", &self.palette)
            .field("gamma", &self.gamma)
            .field("sbit", &self.sbit)
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

        let (data, _checksum) = decompress(&data, zlib::Format::Zlib).unwrap();
        let data = Self::unfilter(&header, data)?;

        Ok(Png {
            header,
            palette,
            gamma,
            sbit,
            data,
        })
    }

    fn scanline_length(header: &IHDRChunk) -> usize {
        let mut scan_line_len = match header.bit_depth {
            BitDepth::B1 => (header.width + 7) / 8,
            BitDepth::B2 => (header.width + 3) / 4,
            BitDepth::B4 => (header.width + 1) / 2,
            BitDepth::B8 => header.width,
            BitDepth::B16 => header.width * 2,
        };
        scan_line_len *= match header.color_type {
            ColorType::Grayscale => 1,
            ColorType::TrueColor => 3,
            ColorType::IndexedColor => 1,
            ColorType::GrayScaleAlpha => 2,
            ColorType::TrueColorAlpha => 4,
        };
        scan_line_len as usize + 1 //1 added for the filter byte.
    }

    fn unfilter(header: &IHDRChunk, data: Vec<u8>) -> std::io::Result<Vec<u8>> {
        let scan_line_len = Self::scanline_length(header);
        let stride = scan_line_len - 1; // pixel bytes per row, filter-type byte excluded
        let bpp = Self::get_bpp(header);
        assert!(data.len() == scan_line_len * header.height as usize);

        let mut output = Vec::with_capacity(stride * header.height as usize);

        for row_i in 0..header.height as usize {
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

        Ok(output)
    }

    fn get_bpp(header: &IHDRChunk) -> usize {
        match header.color_type {
            ColorType::Grayscale => match header.bit_depth {
                BitDepth::B1 | BitDepth::B2 | BitDepth::B4 | BitDepth::B8 => 1,
                BitDepth::B16 => 2,
            },
            ColorType::TrueColor => match header.bit_depth {
                BitDepth::B8 => 3,
                BitDepth::B16 => 6,
                _ => unreachable!(),
            },
            ColorType::IndexedColor => match header.bit_depth {
                // a pixel is a single palette-index sample, so bpp is always 1 byte
                // (rounding up), regardless of how many bits that sample uses.
                BitDepth::B1 | BitDepth::B2 | BitDepth::B4 | BitDepth::B8 => 1,
                _ => unreachable!(),
            },
            ColorType::GrayScaleAlpha => match header.bit_depth {
                BitDepth::B8 => 2,
                BitDepth::B16 => 4,
                _ => unreachable!(),
            },
            ColorType::TrueColorAlpha => match header.bit_depth {
                BitDepth::B8 => 4,
                BitDepth::B16 => 8,
                _ => unreachable!(),
            },
        }
    }

    /// Unpacks `width` single-channel samples (grayscale intensities or
    /// palette indices) from one scanline's worth of bytes, per `bit_depth`.
    /// Rows are byte-aligned, so for bit depths < 8 the last byte of a row
    /// may hold unused padding bits that must be dropped rather than read as
    /// extra samples.
    fn unpack_samples(row: &[u8], bit_depth: BitDepth, width: usize) -> Vec<u8> {
        match bit_depth {
            BitDepth::B8 => row[..width].to_vec(),
            BitDepth::B1 | BitDepth::B2 | BitDepth::B4 => {
                let bits = match bit_depth {
                    BitDepth::B1 => 1,
                    BitDepth::B2 => 2,
                    BitDepth::B4 => 4,
                    _ => unreachable!(),
                };
                let samples_per_byte = 8 / bits;
                row.iter()
                    .flat_map(|&byte| {
                        (0..samples_per_byte)
                            .map(move |i| (byte >> (8 - bits * (i + 1))) & ((1 << bits) - 1))
                    })
                    .take(width)
                    .collect()
            }
            BitDepth::B16 => unreachable!("16-bit samples aren't bit-packed, nothing to unpack"),
        }
    }

    /// Maximum sample value representable at `bit_depth` (i.e. `2^bits - 1`),
    /// used as the PPM/PGM header's maxval for grayscale and true-color images.
    fn max_sample_value(bit_depth: BitDepth) -> u32 {
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

    fn write_to_ppm(&self, filepath: &Path) -> std::io::Result<()> {
        let mut file = File::create(filepath)?;
        match self.header.color_type {
            ColorType::Grayscale => {
                writeln!(file, "P2")?;
                writeln!(file, "{} {}", self.header.width, self.header.height)?;
                writeln!(file, "{}", Self::max_sample_value(self.header.bit_depth))?;
                match self.header.bit_depth {
                    BitDepth::B1 | BitDepth::B2 | BitDepth::B4 => {
                        let width = self.header.width as usize;
                        let stride = Self::scanline_length(&self.header) - 1;
                        for row in self.data.chunks_exact(stride) {
                            for sample in Self::unpack_samples(row, self.header.bit_depth, width) {
                                writeln!(file, "{sample}")?;
                            }
                        }
                    }
                    BitDepth::B8 => {
                        for byte in &self.data {
                            writeln!(file, "{}", byte)?;
                        }
                    }
                    BitDepth::B16 => {
                        for byte in self.data.chunks_exact(2) {
                            let v = (byte[0] as u16) << 8 | byte[1] as u16;
                            writeln!(file, "{}", v)?;
                        }
                    }
                }
            }
            ColorType::TrueColor => {
                writeln!(file, "P3")?;
                writeln!(file, "{} {}", self.header.width, self.header.height)?;
                writeln!(file, "{}", Self::max_sample_value(self.header.bit_depth))?;
                match self.header.bit_depth {
                    BitDepth::B8 => {
                        for byte in self.data.chunks_exact(3) {
                            writeln!(file, "{} {} {}", byte[0], byte[1], byte[2])?;
                        }
                    }
                    BitDepth::B16 => {
                        for byte in self.data.chunks_exact(6) {
                            let v1 = (byte[0] as u16) << 8 | byte[1] as u16;
                            let v2 = (byte[2] as u16) << 8 | byte[3] as u16;
                            let v3 = (byte[4] as u16) << 8 | byte[5] as u16;
                            writeln!(file, "{} {} {}", v1, v2, v3)?;
                        }
                    }
                    _ => unreachable!(),
                }
            }
            ColorType::IndexedColor => {
                let palette = &self
                    .palette
                    .as_ref()
                    .expect("IndexedColor image must have a PLTE chunk")
                    .palettes;

                writeln!(file, "P3")?;
                writeln!(file, "{} {}", self.header.width, self.header.height)?;
                writeln!(file, "255")?;

                let width = self.header.width as usize;
                let stride = Self::scanline_length(&self.header) - 1;
                for row in self.data.chunks_exact(stride) {
                    for index in Self::unpack_samples(row, self.header.bit_depth, width) {
                        let Palette { r, g, b } = palette[index as usize];
                        writeln!(file, "{r} {g} {b}")?;
                    }
                }
            }
            ColorType::GrayScaleAlpha => todo!(),
            ColorType::TrueColorAlpha => todo!(),
        }
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
        output_path.push(format!(
            "{}.{}",
            line,
            if line.contains('g') { "pgm" } else { "ppm" }
        ));
        png.write_to_ppm(&output_path)?;
    }

    Ok(())
}
