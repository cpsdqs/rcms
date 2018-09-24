use pipe::Pipeline;
use profile::Profile;
use Intent;

impl Pipeline {
    pub(crate) fn new_gamut_check(
        _profiles: &[Profile],
        _bpc: &[bool],
        _intents: &[Intent],
        _adaptation_states: &[f64],
    ) -> Pipeline {
        // TODO
        unimplemented!()
    }
}
