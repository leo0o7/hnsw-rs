use serde::{Deserialize, Serialize};

pub trait Distance<const D: usize> {
    fn distance(&self, a: &[f32; D], b: &[f32; D]) -> f32;
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct L2Squared;

impl<const D: usize> Distance<D> for L2Squared {
    #[inline(always)]
    fn distance(&self, a: &[f32; D], b: &[f32; D]) -> f32 {
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut s4 = 0.0f32;
        let mut s5 = 0.0f32;
        let mut s6 = 0.0f32;
        let mut s7 = 0.0f32;

        let mut i = 0;

        while i + 8 <= D {
            let d0 = a[i] - b[i];
            let d1 = a[i + 1] - b[i + 1];
            let d2 = a[i + 2] - b[i + 2];
            let d3 = a[i + 3] - b[i + 3];
            let d4 = a[i + 4] - b[i + 4];
            let d5 = a[i + 5] - b[i + 5];
            let d6 = a[i + 6] - b[i + 6];
            let d7 = a[i + 7] - b[i + 7];

            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
            s4 += d4 * d4;
            s5 += d5 * d5;
            s6 += d6 * d6;
            s7 += d7 * d7;

            i += 8;
        }

        while i < D {
            let d = a[i] - b[i];
            s0 += d * d;
            i += 1;
        }

        (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
    }
}
