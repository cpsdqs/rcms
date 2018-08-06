use lut::{Stage, StageData, StageType};
use transform::NamedColorList;

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
