//! x86 SIMD buffers

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::LengthError;
use core::{
    convert::TryFrom,
    mem,
    ops::{BitXor, BitXorAssign, Index, IndexMut},
};
use super::Unsigned;

macro_rules! simd_int {
    (
        $name:ident,
        $inner:ty,
        $bs:expr,
        $zero:ident,
        $loadu:ident,
        $storeu:ident,
        $xor:ident,
        $doc:expr
    ) => {
        #[doc=$doc]
        #[derive(Copy, Clone, Debug)]
        #[repr(transparent)]
        pub struct $name($inner);

        impl Default for $name {
            #[inline(always)]
            fn default() -> Self {
                Self(unsafe { $zero() })
            }
        }

        impl From<$inner> for $name {
            #[inline(always)]
            fn from(x: $inner) -> $name {
                Self(x)
            }
        }

        impl From<$name> for $inner {
            #[inline(always)]
            fn from(x: $name) -> $inner {
                x.0
            }
        }

        impl From<[u8; $bs]> for $name {
            #[inline(always)]
            fn from(bytes: [u8; $bs]) -> $name {
                Self(unsafe { $loadu(bytes.as_ptr() as *const $inner) })
            }
        }

        impl From<$name> for [u8; $bs] {
            #[inline(always)]
            fn from(x: $name) -> [u8; $bs] {
                unsafe {
                    let mut bytes: [u8; $bs] = mem::zeroed();
                    $storeu(bytes.as_mut_ptr() as *mut $inner, x.0);
                    bytes
                }
            }
        }

        impl BitXor for $name {
            type Output = Self;

            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                // TODO(tarcieri): runtime CPU feature detection
                unsafe { self.$xor(rhs) }
            }
        }

        impl BitXorAssign for $name {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = *self ^ rhs;
            }
        }

        impl Unsigned for $name {}
    };
}

simd_int!(
    U128,
    __m128i,
    16,
    _mm_setzero_si128,
    _mm_loadu_si128,
    _mm_storeu_si128,
    bitxor_sse2,
    "128-bit SIMD integer"
);

impl U128 {
    #[target_feature(enable = "sse2")]
    pub unsafe fn bitxor_sse2(self, rhs: Self) -> Self {
        Self(_mm_xor_si128(self.0, rhs.0))
    }
}

impl From<u128> for U128 {
    #[inline(always)]
    fn from(x: u128) -> U128 {
        Self(unsafe { *(&x as *const _ as *const __m128i) })
    }
}

impl From<U128> for u128 {
    #[inline(always)]
    fn from(x: U128) -> u128 {
        unsafe { *(&x.0 as *const _ as *const u128) }
    }
}

impl TryFrom<&[u8]> for U128 {
    type Error = LengthError;

    fn try_from(bytes: &[u8]) -> Result<U128, LengthError> {
        match bytes.len() {
            0..=15 => {
                let mut buf = [0u8; 16];
                buf[..bytes.len()].copy_from_slice(bytes);
                Ok(buf.into())
            }
            16 => Ok(Self(unsafe {
                _mm_loadu_si128(bytes.as_ptr() as *const __m128i)
            })),
            _ => Err(LengthError),
        }
    }
}

simd_int!(
    U256,
    __m256i,
    16,
    _mm256_setzero_si256,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    bitxor_avx2,
    "256-bit SIMD integer"
);

impl U256 {
    #[target_feature(enable = "avx")]
    pub unsafe fn from_u128x2_avx(lo: U128, hi: U128) -> Self {
        // TODO(tarcieri): compare lo/hi ordering and properties vs `_mm256_set_m128i`
        Self(_mm256_setr_m128i(lo.0, hi.0))
    }

    #[target_feature(enable = "avx")]
    #[target_feature(enable = "sse2")]
    pub unsafe fn load_u128x2_avx(lo: &U128, hi: &U128) -> Self {
        Self(_mm256_loadu2_m128i(&lo.0, &hi.0))
    }

    #[target_feature(enable = "avx")]
    #[target_feature(enable = "sse2")]
    pub unsafe fn store_u128x2_avx(self, lo: &mut U128, hi: &mut U128) {
        _mm256_storeu2_m128i(&mut hi.0, &mut lo.0, self.0);
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn bitxor_avx2(self, rhs: Self) -> Self {
        Self(_mm256_xor_si256(self.0, rhs.0))
    }
}

impl From<(U128, U128)> for U256 {
    fn from(pair: (U128, U128)) -> U256 {
        // TODO(tarcieri): runtime AVX feature detection
        unsafe { U256::from_u128x2_avx(pair.0, pair.1) }
    }
}

impl From<U256> for (U128, U128) {
    fn from(x: U256) -> (U128, U128) {
        let mut lo = U128::default();
        let mut hi = U128::default();

        // TODO(tarcieri): runtime AVX feature detection
        unsafe { x.store_u128x2_avx(&mut lo, &mut hi) }
        (lo, hi)
    }
}

/// 1024-bit SIMD buffer decomposed into 8 x [`U128`] values.
#[derive(Clone, Debug, Default)]
#[repr(align(128))]
pub struct U128x8([U128; 8]);

impl TryFrom<&[u8]> for U128x8 {
    type Error = LengthError;

    fn try_from(bytes: &[u8]) -> Result<U128x8, LengthError> {
        let mut chunks = bytes.chunks_exact(16);
        let mut result: [U128; 8] = unsafe { mem::zeroed() };
        let mut i = 0;

        for chunk in &mut chunks {
            if i == 8 {
                return Err(LengthError);
            }

            result[i] = unsafe { _mm_loadu_si128(chunk.as_ptr() as *const __m128i) }.into();
            i += 1;
        }

        let rem = chunks.remainder();

        if !rem.is_empty() {
            if i < 8 {
                result[i] = U128::try_from(rem)?;
                i += 1;
            } else {
                return Err(LengthError);
            }
        }

        while i < 8 {
            result[i] = U128::zero();
            i += 1;
        }

        Ok(Self(result))
    }
}

impl BitXorAssign for U128x8 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        // TODO(tarcieri): runtime AVX2 feature detection
        unsafe { self.bitxor_assign_avx2(&rhs) }
    }
}

impl Index<usize> for U128x8 {
    type Output = U128;

    fn index(&self, idx: usize) -> &U128 {
        self.0.index(idx)
    }
}

impl IndexMut<usize> for U128x8 {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.0.index_mut(idx)
    }
}

impl U128x8 {
    /// Perform a [`BitXorAssign`] operation using AVX2 intrinsics.
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "sse2")]
    pub unsafe fn bitxor_assign_avx2(&mut self, rhs: &U128x8) {
        for (lhs, rhs) in self.0.chunks_exact_mut(2).zip(rhs.0.chunks_exact(2)) {
            let out = _mm256_xor_si256(
                _mm256_setr_m128i(lhs[0].0, lhs[1].0),
                _mm256_setr_m128i(rhs[0].0, rhs[1].0),
            );

            _mm256_storeu2_m128i(&mut lhs[0].0, &mut lhs[1].0, out);
        }
    }
}
