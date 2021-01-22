//! SIMD buffer types.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/RustCrypto/meta/master/logo_small.png",
    html_root_url = "https://docs.rs/simd-buffers/0.1.0"
)]
#![warn(rust_2018_idioms)] // TODO: missing_docs

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", test)))]
mod portable;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86::{U128x8, U128, U256};

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub use portable::{U128x8, U128};

use core::ops;

pub struct LengthError;

/// Unsigned integer trait
pub trait Unsigned: Default + ops::BitXor + ops::BitXorAssign {
    /// Return the zero value
    fn zero() -> Self {
        Self::default()
    }
}
