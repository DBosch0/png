use std::{
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::exit,
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
    filter_method: FilterMethod,
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
enum FilterMethod {
    None,
    Sub,
    Up,
    Average,
    Paeth,
}

impl From<u8> for FilterMethod {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Sub,
            2 => Self::Up,
            3 => Self::Average,
            4 => Self::Paeth,
            _ => unreachable!("invalid filtering method, expected 0, 1, 2, 3, 4, got {value}"),
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
        let filter_method = FilterMethod::from(read_u8(read)?);
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

        Ok(IHDRChunk {
            width,
            height,
            bit_depth,
            color_type,
            filter_method,
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
            _ = read_u32(read)?; //length of IHDR block
            let chunk_type = read_chunk_type(read)?;
            assert!(&chunk_type == b"IHDR", "png must start with header");
            let header = IHDRChunk::from_bytes(read)?;
            _ = read_u32(read)?; //crc
            header
        };

        let mut palette: Option<PLTEChunk> = None;
        let mut data: Vec<u8> = Vec::new();
        let mut gamma: Option<f64> = None;
        let mut sbit: Option<SBITChunk> = None;

        loop {
            let length = read_u32(read)?;
            let chunk_type = read_chunk_type(read)?;
            match &chunk_type {
                b"IHDR" => unreachable!("png can only have 1 header chunk"),
                b"gAMA" => {
                    assert!(palette.is_none(), "gAMA chunk must precede PLTE data");
                    assert!(data.is_empty(), "gAMA chunk must precede IDAT data");
                    let gamma_u32 = read_u32(read)?;
                    gamma = Some(gamma_u32 as f64 / 100000.0);
                }
                b"sBIT" => {
                    assert!(palette.is_none(), "gAMA chunk must precede PLTE data");
                    assert!(data.is_empty(), "gAMA chunk must precede IDAT data");
                    sbit = Some(SBITChunk::read_bytes(
                        read,
                        length as usize,
                        header.color_type,
                    )?);
                }
                b"PLTE" => {
                    if palette.is_none() {
                        palette = Some(PLTEChunk::from_bytes(read, length as usize)?)
                    } else {
                        unreachable!("png can only have 1 palette chunk");
                    }
                }
                b"IDAT" => {
                    let mut data_block = vec![0; length as usize];
                    read.read_exact(&mut data_block)?;
                    data.extend(&data_block);
                }
                b"IEND" => {
                    _ = read_u32(read)?; //crc         
                    break;
                }
                x => unimplemented!(
                    "unimplemented chunk type, {}",
                    str::from_utf8(x).expect("valid ascii string")
                ),
            }

            _ = read_u32(read)?; //crc 
        }

        let n = read.read(&mut [0])?;
        assert!(n == 0, "png has data remaing after IEND chunk");

        dbg!(&data.len());
        let mut decoder = flate2::write::ZlibDecoder::new(Vec::new());
        decoder.write_all(&data).unwrap();
        let data = decoder.finish().unwrap();

        // let (data, _checksum) = decompress(&data, zlib::Format::Zlib).unwrap();
        dbg!(&data.len());
        let data = Self::unfilter(&header, data)?;
        dbg!(data.len());

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
            ColorType::IndexedColor => 3,
            ColorType::GrayScaleAlpha => 2,
            ColorType::TrueColorAlpha => 4,
        };

        scan_line_len = match header.filter_method {
            FilterMethod::None => scan_line_len + 1,
            FilterMethod::Sub => todo!(),
            FilterMethod::Up => todo!(),
            FilterMethod::Average => todo!(),
            FilterMethod::Paeth => todo!(),
        };
        scan_line_len as usize
    }

    fn unfilter(header: &IHDRChunk, data: Vec<u8>) -> std::io::Result<Vec<u8>> {
        let mut output = Vec::new();
        let scan_line_len = Self::scanline_length(header);
        dbg!(scan_line_len);
        assert!(data.len() / scan_line_len == header.height as usize);
        match header.filter_method {
            FilterMethod::None => {
                for line in data.chunks(scan_line_len) {
                    output.extend_from_slice(&line[1..]);
                }
            }
            FilterMethod::Sub => todo!(),
            FilterMethod::Up => todo!(),
            FilterMethod::Average => todo!(),
            FilterMethod::Paeth => todo!(),
        }
        Ok(output)
    }

    fn write_to_ppm(&self, filepath: &Path) -> std::io::Result<()> {
        let mut file = File::create(filepath)?;
        match self.header.color_type {
            ColorType::Grayscale => {
                writeln!(file, "P2")?;
                writeln!(file, "{} {}", self.header.width, self.header.height)?;
                writeln!(
                    file,
                    "{}",
                    match self.header.bit_depth {
                        BitDepth::B1 => 1 << 1,
                        BitDepth::B2 => 1 << 2,
                        BitDepth::B4 => 1 << 4,
                        BitDepth::B8 => 1 << 8,
                        BitDepth::B16 => 1 << 16,
                    } - 1
                )?;
                match self.header.bit_depth {
                    BitDepth::B1 => {
                        // dbg!("here", self.data.len());
                        for byte in &self.data {
                            writeln!(
                                file,
                                "{} {} {} {} {} {} {} {}",
                                (byte >> 7) & 1,
                                (byte >> 6) & 1,
                                (byte >> 5) & 1,
                                (byte >> 4) & 1,
                                (byte >> 3) & 1,
                                (byte >> 2) & 1,
                                (byte >> 1) & 1,
                                (byte >> 0) & 1
                            )?;
                        }
                    }
                    BitDepth::B2 => {
                        for byte in &self.data {
                            writeln!(
                                file,
                                "{} {} {} {}",
                                (byte >> 6) & 0b11,
                                (byte >> 4) & 0b11,
                                (byte >> 2) & 0b11,
                                (byte >> 0) & 0b11
                            )?;
                        }
                    }
                    BitDepth::B4 => {
                        for byte in &self.data {
                            writeln!(file, "{} {}", (byte >> 4) & 0b1111, (byte >> 0) & 0b1111)?
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
            ColorType::TrueColor => todo!(),
            ColorType::IndexedColor => todo!(),
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
        // return Ok(());
    }

    Ok(())
}
