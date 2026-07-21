use std::f32::consts::PI;

use crate::shape::Region2D;
use crate::shape::regions::rejection_sample;

/// Lanczos approximation of the Gamma function for `x > 0`.
///
/// Accurate to ~15 significant digits for typical inputs.
/// Coefficients from Numerical Recipes 3rd ed., §6.1.
fn gamma(mut x: f32) -> f32 {
    if x < 0.5 {
        // Reflection: Γ(x) = π / (sin(πx) · Γ(1 − x))
        return PI / ((PI * x).sin() * gamma(1.0 - x));
    }
    x -= 1.0;
    const G: usize = 7;
    const C: [f32; G + 2] = [
        0.999_999_999_999_809_9,
        676.520_4,
        -1_259.139_2,
        771.323_4,
        -176.615_04,
        12.507_343,
        -0.138_571_1,
        9.984_369e-6,
        1.505_632_7e-7,
    ];
    let mut ag = C[0];
    for (i, ci) in C.iter().enumerate().skip(1) {
        ag += ci / (x + i as f32);
    }
    let t = x + G as f32 + 0.5;
    (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * ag
}

/// Region type for a superellipse `|a|^n + |b|^n ≤ 1`.
///
/// `n = 2` is a circle (area = π), `n → ∞` approaches a square (area → 4).
/// The exponent must be positive.
#[derive(Clone)]
pub struct SuperellipseRegion {
    pub n: f32,
}

impl SuperellipseRegion {
    pub fn new(n: f32) -> Self {
        debug_assert!(n > 0.0, "superellipse exponent must be positive");
        Self { n }
    }
}

impl Region2D for SuperellipseRegion {
    fn contains(&self, a: f32, b: f32) -> bool {
        a.abs().powf(self.n) + b.abs().powf(self.n) <= 1.0
    }

    fn area(&self) -> f32 {
        // Closed form: 4 · Γ(1 + 1/n)² / Γ(1 + 2/n)
        4.0 * gamma(1.0 + 1.0 / self.n).powi(2) / gamma(1.0 + 2.0 / self.n)
    }

    fn bounding_box_area(&self) -> f32 {
        4.0 // uniform over [-1,1]²
    }

    fn sample(&self, u: f32, v: f32) -> (f32, f32) {
        let (a, b) = rejection_sample(u, v, self);
        (a, b)
    }
}
