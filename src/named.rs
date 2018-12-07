use crate::pipe::{Stage, StageData, StageType};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamedColor {
    name: String,
    pcs: [u16; 3],
    device_colorant: [u16; 16],
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamedColorList {
    pub colorants: usize,
    pub prefix: [u8; 32],
    pub prefix_end: u8,
    pub suffix: [u8; 32],
    pub suffix_end: u8,
    pub colors: Vec<NamedColor>,
}

impl Stage {
    pub(crate) fn new_named(color_list: NamedColorList, use_pcs: bool) -> Stage {
        Stage::alloc(
            StageType::NamedColor,
            1,
            if use_pcs { 3 } else { color_list.colorants },
            if use_pcs {
                eval_named_color_pcs
            } else {
                eval_named_color
            },
            StageData::NamedColorList(color_list),
        )
    }
}

// TODO
fn eval_named_color(_: &[f32], _: &mut [f32], _: &Stage) {
    unimplemented!()
}
fn eval_named_color_pcs(_: &[f32], _: &mut [f32], _: &Stage) {
    unimplemented!()
}
