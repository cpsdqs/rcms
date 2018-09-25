use byteorder::{ByteOrder, ReadBytesExt, BE};
use cgmath::num_traits::{FromPrimitive, ToPrimitive};
use profile::Profile;
use std::collections::HashMap;
use std::{fmt, io, mem};
use types::{decode_date_time, tag_deserializer, tag_types, ICCDateTime};
use {ColorSpace, ICCTag, ICCTagType, Intent, ProfileClass, S15Fixed16};

pub enum DeserError {
    /// Magic number mismatch.
    MagicNumber,
    /// Something is invalid.
    Invalid,
    /// The file is too big.
    TooBig,
    /// The type is unsupported.
    UnsupportedType(u32),
    /// An error occurred while deserializing a tag.
    TagError(ICCTag, ICCTagType, io::Error),
    /// An IO error.
    IO(io::Error),
}

impl fmt::Debug for DeserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DeserError::MagicNumber => write!(f, "MagicNumber"),
            DeserError::Invalid => write!(f, "Invalid"),
            DeserError::TooBig => write!(f, "TooBig"),
            DeserError::UnsupportedType(x) => {
                use std::ffi::CStr;

                let a = (x >> 24) as u8;
                let b = ((x >> 16) & 0xFF_u32) as u8;
                let c = ((x >> 8) & 0xFF_u32) as u8;
                let d = (x & 0xFF) as u8;
                let bytes = [a, b, c, d, 0];
                match CStr::from_bytes_with_nul(&bytes) {
                    Ok(cstr) => write!(f, "UnsupportedType({})", cstr.to_string_lossy()),
                    Err(_) => write!(
                        f,
                        "UnsupportedType({:02x} {:02x} {:02x} {:02x})",
                        a, b, c, d
                    ),
                }
            }
            DeserError::TagError(tag, ty, err) => {
                write!(f, "TagError({:?}, {:?}, {:?})", tag, ty, err)
            }
            DeserError::IO(err) => write!(f, "IO({:?})", err),
        }
    }
}

impl From<io::Error> for DeserError {
    fn from(err: io::Error) -> DeserError {
        DeserError::IO(err)
    }
}

type Signature = u32;

/// ICC Platforms
enum Platform {
    /// `APPL`
    MacOS = 0x4150504C,
    /// `MSFT`
    Microsoft = 0x4D534654,
    /// `SUNW`
    Solaris = 0x53554E57,
    /// `SGI `
    SGI = 0x53474920,
    /// `TGNT`
    Taligent = 0x54474E54,
    /// `*nix` from argyll—not official
    Unices = 0x2A6E6978,
}

/// `acsp`
const MAGIC_NUMBER: u32 = 0x61637370;

/// Profile header -- is 32-bit aligned, so no issues expected with alignment
#[repr(C)]
struct ICCHeader {
    /// Profile size in bytes.
    size: u32,
    /// CMM for this profile.
    cmm_id: Signature,
    /// Format version number.
    version: u32,
    /// Type of profile.
    device_class: Signature,
    /// Color space of data.
    color_space: Signature,
    /// PCS; XYZ or Lab only.
    pcs: Signature,
    /// Date profile was created.
    date: ICCDateTime,
    /// Magic number to identify an ICC profile.
    magic: Signature,
    /// Primary platform.
    platform: Signature,
    /// Various bit settings.
    flags: u32,
    /// Device manufacturer.
    manufacturer: Signature,
    /// Device model number.
    model: u32,
    /// Device attributes.
    attributes: u64,
    /// Rendering intent.
    intent: u32,
    /// Profile illuminant.
    illuminant: EncodedXYZ,
    /// Profile creator.
    creator: Signature,
    /// Profile ID using MD5.
    profile_id: [u32; 4],
    /// Reserved for future use.
    reserved: [u8; 28],
}

/// ICC XYZ
#[repr(C)]
struct EncodedXYZ {
    x: S15Fixed16,
    y: S15Fixed16,
    z: S15Fixed16,
}

/// Profile ID as computed by MD5 algorithm
union ProfileID {
    u128: u128,
    u32: [u32; 4],
}

/// A tag entry
#[repr(C)]
struct TagEntry {
    signature: ICCTag,
    /// Start of tag
    offset: u32,
    /// Size in bytes
    size: u32,
}

#[derive(Debug)]
enum TagData<'a> {
    Data(&'a [u8]),
    Linked(ICCTag),
}

impl Profile {
    /// Deserializes an ICC profile from binary data.
    pub fn deser<T: io::Read>(input: &mut T) -> Result<Profile, DeserError> {
        // created with dummy values that’ll be filled in below
        let mut profile = Profile::new(ProfileClass::Abstract, ColorSpace::XYZ);
        let (header_size, tags) = read_header(input, &mut profile)?;

        let mut max_offset = 0;
        for tag in &tags {
            let offset = tag.offset + tag.size;
            if offset > max_offset {
                max_offset = offset;
            }
        }

        // hard limit at 16 MiB
        if max_offset > 16_777_216 {
            return Err(DeserError::TooBig);
        }

        let mut pool = Vec::with_capacity(max_offset as usize - header_size);
        pool.resize(max_offset as usize - header_size, 0);
        input.read_exact(&mut pool)?;

        let mut pool_tags = HashMap::new();
        let mut pool_tags_by_offset = HashMap::new();

        for tag in tags {
            let offset = tag.offset as usize - header_size;
            let mut end = offset + tag.size as usize;
            if pool_tags_by_offset.contains_key(&offset) {
                let link = pool_tags_by_offset.get(&offset).unwrap();
                pool_tags.insert(tag.signature, TagData::Linked(*link));
            } else {
                // fix for “untrustworthy” MLU size
                if let Some(ICCTagType::MLU) = ICCTagType::from_u32(BE::read_u32(&pool[offset..])) {
                    // just pass the whole thing
                    end = pool.len();
                }
                pool_tags.insert(tag.signature, TagData::Data(&pool[offset..end]));
                pool_tags_by_offset.insert(offset, tag.signature);
            }
        }

        for (tag, data) in pool_tags {
            match data {
                TagData::Linked(target) => profile.link_tag(tag, target),
                TagData::Data(buf) => if let Some(tag_types) = tag_types(tag) {
                    if buf.len() < 8 {
                        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
                    }
                    let data_type = BE::read_u32(buf);

                    let tag_type = match tag_types.iter().find(|x| x.to_u32() == Some(data_type)) {
                        Some(tt) => *tt,
                        None => return Err(DeserError::UnsupportedType(data_type)),
                    };

                    if let Some(deser_tag) = tag_deserializer(tag_type) {
                        profile.insert_tag_raw(
                            tag,
                            match deser_tag(&buf[8..]) {
                                Ok(data) => data,
                                Err(err) => return Err(DeserError::TagError(tag, tag_type, err)),
                            },
                        );
                    } else {
                        profile.insert_tag_raw_data(tag, buf.to_vec());
                    }
                },
            }
        }

        Ok(profile)
    }
}

/// Enforces that the profile version is per spec.
/// Operates on the big endian bytes from the profile.
/// Called before converting to platform endianness.
/// Byte 0 is BCD major version, so max 9.
/// Byte 1 is 2 BCD digits, one per nibble.
/// Reserved bytes 2 & 3 must be 0.
fn validated_version(version: u32) -> u32 {
    union VersionUnion {
        bytes: [u8; 4],
        version: u32,
    }
    let mut bytes = unsafe { VersionUnion { version }.bytes };
    if bytes[0] > 0x09 {
        bytes[0] = 0x09;
    }
    let mut tmp1 = bytes[1] & 0xf0;
    let mut tmp2 = bytes[1] & 0x0f;
    if tmp1 > 0x90 {
        tmp1 = 0x90;
    }
    if tmp2 > 0x09 {
        tmp2 = 0x09;
    }
    bytes[1] = tmp1 | tmp2;
    bytes[2] = 0;
    bytes[3] = 0;
    unsafe { VersionUnion { bytes }.version }
}

fn read_header<T: io::Read>(
    buf: &mut T,
    profile: &mut Profile,
) -> Result<(usize, Vec<TagEntry>), DeserError> {
    let mut header_buf = Vec::new();
    let mut header_size = mem::size_of::<ICCHeader>();
    header_buf.resize(mem::size_of::<ICCHeader>(), 0);
    buf.read_exact(&mut header_buf)?;

    let header = unsafe { &*(&*header_buf as *const [u8] as *const ICCHeader) };

    if u32::from_be(header.magic) != MAGIC_NUMBER {
        return Err(DeserError::MagicNumber);
    }

    macro_rules! from_be {
        ($enum:ty, $expr:expr, $err:expr) => {
            match <$enum>::from_u32(u32::from_be($expr)) {
                Some(x) => x,
                None => return Err($err),
            }
        };
    }

    profile.device_class = from_be!(ProfileClass, header.device_class, DeserError::Invalid);
    profile.color_space = from_be!(ColorSpace, header.color_space, DeserError::Invalid);
    profile.pcs = from_be!(ColorSpace, header.pcs, DeserError::Invalid);
    match profile.pcs {
        ColorSpace::XYZ | ColorSpace::Lab => (),
        _ => return Err(DeserError::Invalid),
    }
    profile.rendering_intent = from_be!(Intent, header.intent, DeserError::Invalid);
    profile.flags = u32::from_be(header.flags);
    profile.manufacturer = u32::from_be(header.manufacturer);
    profile.model = u32::from_be(header.model);
    profile.creator = u32::from_be(header.creator);

    profile.attributes = u64::from_be(header.attributes);
    profile.version = u32::from_be(validated_version(header.version));

    profile.created = decode_date_time(&header.date);

    profile.profile_id = unsafe {
        ProfileID {
            u32: header.profile_id,
        }.u128
    };

    // tag directory

    header_size += mem::size_of::<u32>();
    let tag_count = buf.read_u32::<BE>()?;
    let mut tags = Vec::new();
    for _ in 0..tag_count {
        header_size += 3 * mem::size_of::<u32>();
        let signature = buf.read_u32::<BE>()?;
        let signature = match ICCTag::from_u32(signature) {
            Some(tag) => tag,
            None => {
                // skip
                // TODO: handle non-lossily
                buf.read_u32::<BE>()?;
                buf.read_u32::<BE>()?;
                continue;
            }
        };
        let offset = buf.read_u32::<BE>()?;
        let size = buf.read_u32::<BE>()?;

        tags.push(TagEntry {
            signature,
            offset,
            size,
        });
    }

    Ok((header_size, tags))
}
