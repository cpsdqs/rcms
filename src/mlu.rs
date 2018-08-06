//! Multilocalized unicode objects.

use std::collections::HashMap;

fn str_to_16(s: &str) -> u16 {
    let mut bytes = s.bytes();
    (((bytes.next().unwrap() as u16) << 8) | bytes.next().unwrap() as u16)
}

/*
fn str_from_16(n: u16) -> String {
    unsafe { String::from_utf8_unchecked(vec![(n >> 8) as u8, n as u8]) }
}
 */

type Lang = u16;
type Country = u16;

/// Multilocalized unicode object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MLU {
    entries: HashMap<(Lang, Country), String>,
}

#[allow(non_camel_case_types)]
type wchar_t = u16;

impl MLU {
    pub fn new() -> MLU {
        MLU {
            entries: HashMap::new(),
        }
    }

    /// Adds an entry.
    pub fn set(&mut self, lang: &str, country: &str, string: &str) {
        let lang = str_to_16(lang);
        let country = str_to_16(country);

        self.entries.insert((lang, country), string.to_string());
    }

    pub fn get(&self, lang: &str, country: &str) -> Option<&str> {
        self.entries
            .get(&(str_to_16(lang), str_to_16(country)))
            .map(|x| &**x)
    }
}
