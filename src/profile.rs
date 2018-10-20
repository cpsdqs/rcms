//! Color profiles.

use cgmath::{Matrix, Matrix3, SquareMatrix};
use gamma::ToneCurve;
use mlu::MLU;
use named::NamedColorList;
use pcs::MAX_ENCODABLE_XYZ;
use pipe::{Pipeline, Stage};
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use time;
use white_point::{adaptation_matrix, D50};
use {CIExyYTriple, ColorSpace, ICCTag, Intent, ProfileClass, CIEXYZ};

#[derive(Debug, Clone)]
pub(crate) enum ProfileTagData {
    Linked(ICCTag),
    Data(Arc<Any + Send + Sync>),
    Raw(Vec<u8>),
}

#[derive(Debug, Clone)]
pub(crate) struct ProfileTag {
    save_as_raw: bool,
    data: ProfileTagData,
}

/// An ICC profile.
#[derive(Clone)]
pub struct Profile {
    // io_handler: ?,
    // context: ?, something about threads?
    /// Creation time
    pub created: time::Tm,
    pub(crate) version: u32,

    /// Device class.
    pub device_class: ProfileClass,

    /// Profile color space type.
    pub color_space: ColorSpace,

    /// Profile connection space.
    pub pcs: ColorSpace,
    pub rendering_intent: Intent,

    /// Flags?
    pub flags: u32,

    /// Manufacturer.
    pub manufacturer: u32,

    /// Model.
    pub model: u32,

    /// Attributes.
    pub attributes: u64,

    /// Creator.
    pub creator: u32,

    pub(crate) profile_id: u128,

    // pub profile_id: ProfileID?,
    pub(crate) tags: HashMap<ICCTag, ProfileTag>,
    // cmsUInt32Number          TagCount;
    // cmsTagSignature          TagNames[MAX_TABLE_TAG];
    // The tag to which is linked (0=none)
    // cmsTagSignature          TagLinked[MAX_TABLE_TAG];
    // Size on disk
    // cmsUInt32Number          TagSizes[MAX_TABLE_TAG];
    // cmsUInt32Number          TagOffsets[MAX_TABLE_TAG];
    // True to write uncooked
    // cmsBool                  TagSaveAsRaw[MAX_TABLE_TAG];
    // void *                   TagPtrs[MAX_TABLE_TAG];
    // cmsTagTypeHandler*       TagTypeHandlers[MAX_TABLE_TAG];
    // Same structure may be serialized on different types
    // depending on profile version, so we keep track of the
    // type handler for each tag in the list.
    /// ?
    pub is_write: bool,
}

// LUT tags
const DEVICE_TO_PCS_16: [ICCTag; 4] = [
    ICCTag::AToB0, // Perceptual
    ICCTag::AToB1, // Relative colorimetric
    ICCTag::AToB2, // Saturation
    ICCTag::AToB1, // Absolute colorimetric
];

const DEVICE_TO_PCS_FLOAT: [ICCTag; 4] = [
    ICCTag::DToB0, // Perceptual
    ICCTag::DToB1, // Relative colorimetric
    ICCTag::DToB2, // Saturation
    ICCTag::DToB3, // Absolute colorimetric
];

const PCS_TO_DEVICE_16: [ICCTag; 4] = [
    ICCTag::BToA0, // Perceptual
    ICCTag::BToA1, // Relative colorimetric
    ICCTag::BToA2, // Saturation
    ICCTag::BToA1, // Absolute colorimetric
];

const PCS_TO_DEVICE_FLOAT: [ICCTag; 4] = [
    ICCTag::BToD0, // Perceptual
    ICCTag::BToD1, // Relative colorimetric
    ICCTag::BToD2, // Saturation
    ICCTag::BToD3, // Absolute colorimetric
];

// Factors to convert from 1.15 fixed point to 0..1.0 range and vice-versa
const INP_ADJ: f64 = 1. / MAX_ENCODABLE_XYZ; // (65536.0/(65535.0*2.0))
const OUTP_ADJ: f64 = MAX_ENCODABLE_XYZ; // ((2.0*65535.0)/65536.0)

// How profiles may be used
// TODO: make this an enum
pub(crate) const USED_AS_INPUT: u32 = 0;
pub(crate) const USED_AS_OUTPUT: u32 = 1;
pub(crate) const USED_AS_PROOF: u32 = 2;

impl Profile {
    pub(crate) fn new(device_class: ProfileClass, color_space: ColorSpace) -> Profile {
        Profile {
            created: time::now_utc(),
            version: 0x02100000, // default version
            device_class,
            color_space,
            pcs: ColorSpace::XYZ,
            rendering_intent: Intent::Perceptual,
            flags: 0,
            manufacturer: 0,
            model: 0,
            attributes: 0,
            creator: 0,
            tags: HashMap::new(),
            is_write: false,
            profile_id: 0,
        }
    }

    pub(crate) fn encoded_version(&self) -> u32 {
        self.version
    }

    /// Returns the profile version as a float value (major.minor).
    pub fn version(&self) -> f64 {
        base_to_base(self.version >> 16, 16, 10) as f64 / 100.
    }

    /// Sets the profile version with a float value (major.minor).
    pub fn set_version(&mut self, version: f64) {
        self.version = base_to_base((version * 100. + 0.5).floor() as u32, 10, 16) << 16;
    }

    fn read_tag_recursive<T: Clone + 'static>(&self, sig: ICCTag, recursion: usize) -> Option<T> {
        if recursion > 30 {
            return None;
        }

        match self.tags.get(&sig) {
            Some(tag) => match tag.data {
                ProfileTagData::Linked(sig) => self.read_tag_recursive(sig, recursion + 1),
                ProfileTagData::Data(ref data) => match data.downcast_ref::<T>() {
                    Some(data) => Some(data.clone()),
                    None => None,
                },
                ProfileTagData::Raw(_) => None,
            },
            None => None,
        }
    }

    /// Reads a profile tag and attempts to cast it to T and then clones it if successful.
    pub fn read_tag_clone<T: Clone + Send + Sync + 'static>(&self, sig: ICCTag) -> Option<T> {
        self.read_tag_recursive(sig, 0)
    }

    /// Inserts a tag with an arbitrary value.
    pub fn insert_tag<T: Clone + Send + Sync + 'static>(&mut self, sig: ICCTag, data: T) {
        self.tags.insert(
            sig,
            ProfileTag {
                save_as_raw: false,
                data: ProfileTagData::Data(Arc::new(data)),
            },
        );
    }

    pub(crate) fn insert_tag_raw(&mut self, sig: ICCTag, data: Arc<Any + Send + Sync>) {
        self.tags.insert(
            sig,
            ProfileTag {
                save_as_raw: false,
                data: ProfileTagData::Data(data),
            },
        );
    }

    pub(crate) fn insert_tag_raw_data(&mut self, sig: ICCTag, buf: Vec<u8>) {
        self.tags.insert(
            sig,
            ProfileTag {
                save_as_raw: false,
                data: ProfileTagData::Raw(buf),
            },
        );
    }

    /// Links one tag’s value to another’s.
    pub fn link_tag(&mut self, sig: ICCTag, to: ICCTag) {
        self.tags.insert(
            sig,
            ProfileTag {
                save_as_raw: false,
                data: ProfileTagData::Linked(to),
            },
        );
    }

    /// Returns true if the key exists.
    pub fn has_tag(&self, tag: ICCTag) -> bool {
        self.tags.contains_key(&tag)
    }

    /// Returns a media white point fixing some issues found in certain old profiles.
    pub fn media_white_point(&self) -> CIEXYZ {
        match self.read_tag_clone(ICCTag::MediaWhitePoint) {
            Some(_) if self.version() < 4. && self.device_class == ProfileClass::Display => D50,
            Some(tag) => tag,
            None => D50,
        }
    }

    /// Returns the chromatic adaptation matrix. Fixes some issues as well.
    pub fn chad(&self) -> Matrix3<f64> {
        match self.read_tag_clone(ICCTag::ChromaticAdaptation) {
            Some(tag) => return tag,
            None => ()
        }

        // also support Vec<f64>
        match self.read_tag_clone::<Vec<f64>>(ICCTag::ChromaticAdaptation) {
            Some(ref tag) if tag.len() == 9 => {
                let mut mat_array = [0.; 9];
                mat_array.copy_from_slice(&tag);
                let mat: &Matrix3<_> = (&mat_array).into();
                return *mat;
            },
            _ => ()
        }

        if self.version() < 4. && self.device_class == ProfileClass::Display {
            match self.read_tag_clone(ICCTag::MediaWhitePoint) {
                Some(wp) => match adaptation_matrix(None, wp, D50) {
                    Some(mat) => mat,
                    None => Matrix3::identity(),
                },
                None => Matrix3::identity(),
            }
        } else {
            Matrix3::identity()
        }
    }

    /// Returns true if the profile is implemented as matrix-shaper.
    pub(crate) fn is_matrix_shaper(&self) -> bool {
        match self.color_space {
            ColorSpace::Gray => self.has_tag(ICCTag::GrayTRC),
            ColorSpace::RGB => {
                self.has_tag(ICCTag::RedColorant)
                    && self.has_tag(ICCTag::GreenColorant)
                    && self.has_tag(ICCTag::BlueColorant)
                    && self.has_tag(ICCTag::RedTRC)
                    && self.has_tag(ICCTag::GreenTRC)
                    && self.has_tag(ICCTag::BlueTRC)
            }
            _ => false,
        }
    }

    /// Returns true if the intent is implemented as CLUT
    pub(crate) fn is_clut(&self, intent: Intent, used_direction: u32) -> bool {
        // For devicelinks, the supported intent is that one stated in the header
        if self.device_class == ProfileClass::Link {
            return self.rendering_intent == intent;
        }

        let tag_table = match used_direction {
            USED_AS_INPUT => DEVICE_TO_PCS_16,
            USED_AS_OUTPUT => PCS_TO_DEVICE_16,
            USED_AS_PROOF => {
                return self.is_intent_supported(intent, USED_AS_INPUT)
                    && self.is_intent_supported(Intent::RelativeColorimetric, USED_AS_OUTPUT)
            }
            _ => return false,
        };

        self.has_tag(tag_table[intent as usize])
    }

    /// Returns info about supported intents
    pub(crate) fn is_intent_supported(&self, intent: Intent, used_direction: u32) -> bool {
        if self.is_clut(intent, used_direction) {
            return true;
        }

        // Is there any matrix-shaper? If so, the intent is supported. This is a bit odd, since V2 matrix shaper
        // does not fully support relative colorimetric because they cannot deal with non-zero black points, but
        // many profiles claims that, and this is certainly not true for V4 profiles. Lets answer "yes" no matter
        // the accuracy would be less than optimal in rel.col and v2 case.
        self.is_matrix_shaper()
    }

    /// Auxiliary, read colorants as a MAT3 structure. Used by any function that needs a matrix-shaper
    fn icc_matrix_rgb_to_xyz(&self) -> Option<Matrix3<f64>> {
        let red: Option<CIEXYZ> = self.read_tag_clone(ICCTag::RedColorant);
        let green: Option<CIEXYZ> = self.read_tag_clone(ICCTag::GreenColorant);
        let blue: Option<CIEXYZ> = self.read_tag_clone(ICCTag::BlueColorant);

        if let (Some(red), Some(green), Some(blue)) = (red, green, blue) {
            Some(Matrix3::from_cols(red.into(), green.into(), blue.into()).transpose())
        } else {
            None
        }
    }

    /// Reads the DToAX tag, adjusting the encoding of Lab or XYZ if needed.
    fn read_float_input_tag(&self, tag: ICCTag) -> Result<Pipeline, String> {
        let mut pipeline: Pipeline = match self.read_tag_clone(tag) {
            Some(p) => p,
            None => return Err(format!("Tag {:?} contains no pipeline", tag)),
        };
        let spc = self.color_space;
        let pcs = self.pcs;

        // input and output of transform are in lcms 0..1 encoding. If XYZ or Lab spaces are used,
        // these need to be normalized into the appropriate ranges (Lab = 100,0,0, XYZ=1.0,1.0,1.0)
        if spc == ColorSpace::Lab {
            pipeline.prepend_stage(Stage::new_normalize_to_lab_float());
        } else if spc == ColorSpace::XYZ {
            pipeline.prepend_stage(Stage::new_normalize_to_xyz_float());
        }

        if pcs == ColorSpace::Lab {
            pipeline.append_stage(Stage::new_normalize_from_lab_float());
        } else if pcs == ColorSpace::XYZ {
            pipeline.append_stage(Stage::new_normalize_from_xyz_float());
        }

        Ok(pipeline)
    }

    /// Reads the DToAX tag, adjusting the encoding of Lab or XYZ if needed
    fn read_float_output_tag(&self, tag: ICCTag) -> Result<Pipeline, String> {
        let mut pipeline: Pipeline = match self.read_tag_clone(tag) {
            Some(p) => p,
            None => return Err(format!("Tag {:?} contains no pipeline", tag)),
        };
        let spc = self.color_space;
        let pcs = self.pcs;

        // If PCS is Lab or XYZ, the floating point tag is accepting data in the space encoding,
        // and since the formatter has already accommodated to 0..1.0, we should undo this change
        if pcs == ColorSpace::Lab {
            pipeline.prepend_stage(Stage::new_normalize_to_lab_float());
        } else if pcs == ColorSpace::XYZ {
            pipeline.prepend_stage(Stage::new_normalize_to_xyz_float());
        }

        // the output can be Lab or XYZ, in which case normalization is needed on the end of the pipeline
        if spc == ColorSpace::Lab {
            pipeline.append_stage(Stage::new_normalize_from_lab_float());
        } else if spc == ColorSpace::XYZ {
            pipeline.append_stage(Stage::new_normalize_from_xyz_float());
        }

        Ok(pipeline)
    }

    /// Reads and creates a new LUT.
    ///
    /// All version-dependent things are adjusted here.
    /// Intent = 0xFFFFFFFF is added as a way to always read the matrix shaper, no matter what other LUTs are.
    pub(crate) fn read_input_lut(&self, intent: Intent) -> Result<Pipeline, String> {
        // On named color, take the appropriate tag
        if self.device_class == ProfileClass::NamedColor {
            let named_colors = match self.read_tag_clone::<NamedColorList>(ICCTag::NamedColor2) {
                Some(nc) => nc,
                None => return Err("No named color list".into()),
            };

            let mut lut = Pipeline::new(0, 0);
            lut.prepend_stage(Stage::new_named(named_colors, true));
            lut.append_stage(Stage::new_labv2_to_v4());
            return Ok(lut);
        }
        // TODO: the rest of this function

        // This is an attempt to reuse this function to retrieve the matrix-shaper as pipeline no
        // matter other LUT are present and have precedence. Intent = 0xffffffff can be used for that.
        if intent as u32 <= Intent::AbsoluteColorimetric as u32 {
            let mut tag_16 = DEVICE_TO_PCS_16[intent as usize];
            let tag_float = DEVICE_TO_PCS_FLOAT[intent as usize];

            if self.has_tag(tag_float) {
                // Float tag takes precedence
                // Floating point LUT are always V4, but the encoding range is no
                // longer 0..1.0, so we need to add an stage depending on the color space
                return self.read_float_input_tag(tag_float);
            }

            // Revert to perceptual if no tag is found
            if !self.has_tag(tag_16) {
                tag_16 = DEVICE_TO_PCS_16[0];
            }

            // Is there any LUT-Based table?
            if self.has_tag(tag_16) {
                // Check profile version and LUT type. Do the necessary adjustments if needed

                // First read the tag
                let _pipeline = match self.read_tag_clone(tag_16) {
                    Some(p) => p,
                    None => return Err("No table".into()),
                };

                unimplemented!()

                /*
                // After reading it, we have now info about the original type
                OriginalType =  _cmsGetTagTrueType(hProfile, tag16);

                // The profile owns the Lut, so we need to copy it
                Lut = cmsPipelineDup(Lut);

                // We need to adjust data only for Lab16 on output
                if (OriginalType != cmsSigLut16Type || cmsGetPCS(hProfile) != cmsSigLabData)
                    return Lut;

                // If the input is Lab, add also a conversion at the begin
                if (cmsGetColorSpace(hProfile) == cmsSigLabData &&
                    !cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageAllocLabV4ToV2(ContextID)))
                    goto Error;

                // Add a matrix for conversion V2 to V4 Lab PCS
                if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageAllocLabV2ToV4(ContextID)))
                    goto Error;

                return Lut; */
            }
        }

        // Lut was not found, try to create a matrix-shaper

        // Check if this is a grayscale profile.
        if self.color_space == ColorSpace::Gray {
            // if so, build appropriate conversion tables.
            // The tables are the PCS iluminant, scaled across GrayTRC
            unimplemented!()
            // return BuildGrayInputMatrixPipeline(hProfile);
        }

        // Not gray, create a normal matrix-shaper
        self.build_rgb_input_matrix_shaper()
    }

    // Create an output MPE LUT from agiven profile. Version mismatches are handled here
    pub(crate) fn read_output_lut(&self, intent: Intent) -> Result<Pipeline, String> {
        if intent as u32 <= Intent::AbsoluteColorimetric as u32 {
            let mut tag_16 = PCS_TO_DEVICE_16[intent as usize];
            let tag_float = PCS_TO_DEVICE_FLOAT[intent as usize];

            if self.has_tag(tag_float) {
                // Float tag takes precedence
                // Floating point LUT are always V4
                return self.read_float_output_tag(tag_float);
            }

            // Revert to perceptual if no tag is found
            if !self.has_tag(tag_16) {
                tag_16 = PCS_TO_DEVICE_16[0];
            }

            if self.has_tag(tag_16) {
                // TODO
                unimplemented!()
            }
            /*
            if (cmsIsTag(hProfile, tag16)) { // Is there any LUT-Based table?

                // Check profile version and LUT type. Do the necessary adjustments if needed

                // First read the tag
                cmsPipeline* Lut = (cmsPipeline*) cmsReadTag(hProfile, tag16);
                if (Lut == NULL) return NULL;

                // After reading it, we have info about the original type
                OriginalType =  _cmsGetTagTrueType(hProfile, tag16);

                // The profile owns the Lut, so we need to copy it
                Lut = cmsPipelineDup(Lut);
                if (Lut == NULL) return NULL;

                // Now it is time for a controversial stuff. I found that for 3D LUTS using
                // Lab used as indexer space,  trilinear interpolation should be used
                if (cmsGetPCS(hProfile) == cmsSigLabData)
                    ChangeInterpolationToTrilinear(Lut);

                // We need to adjust data only for Lab and Lut16 type
                if (OriginalType != cmsSigLut16Type || cmsGetPCS(hProfile) != cmsSigLabData)
                    return Lut;

                // Add a matrix for conversion V4 to V2 Lab PCS
                if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageAllocLabV4ToV2(ContextID)))
                    goto Error;

                // If the output is Lab, add also a conversion at the end
                if (cmsGetColorSpace(hProfile) == cmsSigLabData)
                    if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageAllocLabV2ToV4(ContextID)))
                        goto Error;

                return Lut;
    Error:
                cmsPipelineFree(Lut);
                return NULL;
            } */
        }

        // Lut not found, try to create a matrix-shaper

        // Check if this is a grayscale profile.
        if self.color_space == ColorSpace::Gray {
            // if so, build appropriate conversion tables.
            // The tables are the PCS iluminant, scaled across GrayTRC
            // return BuildGrayOutputPipeline(hProfile);
            unimplemented!()
        }

        // Not gray, create a normal matrix-shaper, which only operates in XYZ space
        self.build_rgb_output_matrix_shaper()
    }

    fn build_rgb_input_matrix_shaper(&self) -> Result<Pipeline, String> {
        // XYZ PCS in encoded in 1.15 format, and the matrix output comes in 0..0xffff range, so
        // we need to adjust the output by a factor of (0x10000/0xffff) to put data in
        // a 1.16 range, and then a >> 1 to obtain 1.15. The total factor is (65536.0)/(65535.0*2)
        let mat = match self.icc_matrix_rgb_to_xyz() {
            Some(m) => m * INP_ADJ,
            None => return Err("Failed to read colorants as matrix".into()),
        };

        let shapes: [ToneCurve; 3] = [
            match self.read_tag_clone(ICCTag::RedTRC) {
                Some(t) => t,
                None => return Err("No red tone curve".into()),
            },
            match self.read_tag_clone(ICCTag::GreenTRC) {
                Some(t) => t,
                None => return Err("No green tone curve".into()),
            },
            match self.read_tag_clone(ICCTag::BlueTRC) {
                Some(t) => t,
                None => return Err("No blue tone curve".into()),
            },
        ];

        let mut pipeline = Pipeline::new(3, 3);

        pipeline.append_stage(Stage::new_tone_curves(3, &shapes));
        pipeline.append_stage(Stage::new_matrix(
            3,
            3,
            unsafe { &*(&mat as *const _ as *const [f64; 9]) },
            None,
        ));

        // Note that it is certainly possible a single profile would have a LUT based
        // tag for output working in lab and a matrix-shaper for the fallback cases.
        // This is not allowed by the spec, but this code is tolerant to those cases
        if self.pcs == ColorSpace::Lab {
            pipeline.append_stage(Stage::new_xyz_to_lab());
        }

        Ok(pipeline)
    }

    fn build_rgb_output_matrix_shaper(&self) -> Result<Pipeline, String> {
        let mat = match self.icc_matrix_rgb_to_xyz() {
            Some(m) => m,
            None => return Err("Failed to read colorants as matrix".into()),
        };

        // XYZ PCS in encoded in 1.15 format, and the matrix input should come in 0..0xffff range, so
        // we need to adjust the input by a << 1 to obtain a 1.16 fixed and then by a factor of
        // (0xffff/0x10000) to put data in 0..0xffff range. Total factor is (2.0*65535.0)/65536.0.
        let inv = match mat.invert() {
            Some(m) => m,
            None => return Err("Could not invert matrix".into()),
        } * OUTP_ADJ;

        let shapes: [ToneCurve; 3] = [
            match self.read_tag_clone(ICCTag::RedTRC) {
                Some(t) => t,
                None => return Err("No red tone curve".into()),
            },
            match self.read_tag_clone(ICCTag::GreenTRC) {
                Some(t) => t,
                None => return Err("No green tone curve".into()),
            },
            match self.read_tag_clone(ICCTag::BlueTRC) {
                Some(t) => t,
                None => return Err("No blue tone curve".into()),
            },
        ];

        let inv_shapes: [ToneCurve; 3] = [
            shapes[0].reverse()?,
            shapes[1].reverse()?,
            shapes[2].reverse()?,
        ];

        let mut pipeline = Pipeline::new(3, 3);

        // Note that it is certainly possible a single profile would have a LUT based
        // tag for output working in lab and a matrix-shaper for the fallback cases.
        // This is not allowed by the spec, but this code is tolerant to those cases
        if self.pcs == ColorSpace::Lab {
            pipeline.append_stage(Stage::new_lab_to_xyz());
        }

        pipeline.append_stage(Stage::new_matrix(
            3,
            3,
            unsafe { &*(&inv as *const _ as *const [f64; 9]) },
            None,
        ));
        pipeline.append_stage(Stage::new_tone_curves(3, &inv_shapes));

        Ok(pipeline)
    }

    // Read the AToD0 tag, adjusting the encoding of Lab or XYZ if neded
    pub(crate) fn read_float_devicelink_tag(&self, tag_float: ICCTag) -> Option<Pipeline> {
        let mut pipeline: Pipeline = self.read_tag_clone(tag_float)?;
        let pcs = self.pcs;
        let spc = self.color_space;

        if spc == ColorSpace::Lab {
            pipeline.prepend_stage(Stage::new_normalize_to_lab_float());
        } else if spc == ColorSpace::XYZ {
            pipeline.prepend_stage(Stage::new_normalize_to_xyz_float());
        }

        if pcs == ColorSpace::Lab {
            pipeline.append_stage(Stage::new_normalize_from_lab_float());
        } else if pcs == ColorSpace::XYZ {
            pipeline.append_stage(Stage::new_normalize_from_xyz_float());
        }

        Some(pipeline)
    }

    /// This one includes abstract profiles as well. Matrix-shaper cannot be obtained on that device class. The
    /// tag name here may default to AToB0
    pub(crate) fn read_devicelink_lut(&self, intent: Intent) -> Result<Pipeline, String> {
        if intent as u32 > Intent::AbsoluteColorimetric as u32 {
            return Err("Intent is not ICC".into());
        }

        let mut tag_16 = DEVICE_TO_PCS_16[intent as usize];
        let mut tag_float = DEVICE_TO_PCS_FLOAT[intent as usize];

        // On named color, take the appropriate tag
        if self.device_class == ProfileClass::NamedColor {
            let named_colors: NamedColorList = match self.read_tag_clone(ICCTag::NamedColor2) {
                Some(nc) => nc,
                None => return Err("No named color list in profile".into()),
            };

            let mut pipeline = Pipeline::new(0, 0);

            pipeline.prepend_stage(Stage::new_named(named_colors, false));
            if self.color_space == ColorSpace::Lab {
                pipeline.append_stage(Stage::new_labv2_to_v4());
            }

            return Ok(pipeline);
        }

        if self.has_tag(tag_float) {
            // Float tag takes precedence
            // Floating point LUT are always V
            if let Some(pipeline) = self.read_float_devicelink_tag(tag_float) {
                return Ok(pipeline);
            }
        }

        tag_float = DEVICE_TO_PCS_FLOAT[0];
        if let Some(pipeline) = self.read_tag_clone(tag_float) {
            return Ok(pipeline);
        }

        if self.has_tag(tag_16) {
            // Is there any LUT-Based table?
            tag_16 = DEVICE_TO_PCS_16[0];
        }

        // Check profile version and LUT type. Do the necessary adjustments if needed

        // Read the tag
        let mut pipeline: Pipeline = match self.read_tag_clone(tag_16) {
            Some(pipeline) => pipeline,
            None => return Err("Profile has no LUT-based table".into()),
        };

        // Now it is time for some controversial stuff. I found that for 3D LUTS using
        // Lab used as indexer space, trilinear interpolation should be used
        if self.pcs == ColorSpace::Lab {
            // TODO
            unimplemented!()
            // ChangeInterpolationToTrilinear(Lut);
        }

        /*
        // After reading it, we have info about the original type
        OriginalType = _cmsGetTagTrueType(hProfile, tag16);

        // We need to adjust data for Lab16 on output
        if (OriginalType != cmsSigLut16Type) return Lut;
        */

        // Here it is possible to get Lab on both sides

        if self.color_space == ColorSpace::Lab {
            pipeline.prepend_stage(Stage::new_labv4_to_v2());
        }

        if self.pcs == ColorSpace::Lab {
            pipeline.append_stage(Stage::new_labv2_to_v4());
        }

        Ok(pipeline)
    }

    // cmsio1.c
    // TODO: fn build_gray_input_pipeline(&self) -> Pipeline
    // TODO: _cmsReadFloatInputTag
    // TODO: BuildGrayOutputPipeline
    // TODO: _cmsReadFloatOutputTag
    // TODO: _cmsReadOutputLUT
    // TODO: cmsIsMatrixShaper
    // TODO: cmsIsCLUT
    // TODO: cmsIsIntentSupported
    // TODO: _cmsReadProfileSequence
    // TODO: _cmsWriteProfileSequence
    // TODO: GetMLUFromProfile
    // TODO: _cmsCompileProfileSequence
    // TODO: GetInfo
    // TODO: cmsGetProfileInfo
    // TODO: cmsGetProfileInfoASCII
}

impl fmt::Debug for Profile {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Profile {{ {:?} version {:x?}, {:?} class, pcs {:?}, {:?} intent",
            self.color_space, self.version, self.device_class, self.pcs, self.rendering_intent
        )?;
        for (k, v) in &self.tags {
            write!(f, ", {:?}: ", k)?;
            macro_rules! def_tag {
                ($($ty:ty),+) => {
                    match v.data {
                        ProfileTagData::Data(ref val) => {
                            if false {}
                            $(
                            else if let Some(d) = val.downcast_ref::<$ty>() {
                                write!(f, "{:?}", d)?;
                            }
                            )+
                            else {
                                write!(f, "<~>")?;
                            }
                        }
                        ProfileTagData::Linked(key) => write!(f, "(-> {:?})", key)?,
                        ProfileTagData::Raw(ref buf) => {
                            write!(f, "<raw blob of size {}>", buf.len())?
                        }
                    }
                }
            }
            match k {
                | ICCTag::ProfileDescription
                | ICCTag::Copyright
                | ICCTag::DeviceModelDesc
                | ICCTag::DeviceMfgDesc => def_tag!(MLU),
                ICCTag::MediaWhitePoint => def_tag!(CIEXYZ),
                ICCTag::MediaBlackPoint => def_tag!(CIEXYZ),
                ICCTag::ChromaticAdaptation => def_tag!(Matrix3<f64>, Vec<f64>),
                ICCTag::RedColorant | ICCTag::GreenColorant | ICCTag::BlueColorant => {
                    def_tag!(CIEXYZ)
                }
                ICCTag::RedTRC | ICCTag::GreenTRC | ICCTag::BlueTRC | ICCTag::GrayTRC => {
                    def_tag!(ToneCurve)
                }
                ICCTag::Chromaticity => def_tag!(CIExyYTriple),
                _ => write!(f, "<~>")?,
            }
        }
        write!(f, " }}")?;
        Ok(())
    }
}

/// Get an hexadecimal number with same digits as v
fn base_to_base(mut n: u32, base_in: u32, base_out: u32) -> u32 {
    let mut buff = Vec::new();

    while n > 0 {
        buff.push(n % base_in);
        n /= base_in;
    }

    let mut out = 0;
    for i in 0..buff.len() {
        out = out * base_out + buff[buff.len() - i - 1];
    }

    out
}
