use std::f64::consts::PI;

use crate::planar::Region2D;

/// Region type for a superellipse `|a|^n + |b|^n ≤ 1`.
///
/// `n = 2` is a circle (area = π), `n → ∞` approaches a square (area → 4).
/// The exponent must be positive.
#[derive(Clone)]
pub struct SuperellipseRegion {
    pub n: f64,
}

impl SuperellipseRegion {
    pub fn new(n: f64) -> Self {
        debug_assert!(n > 0.0, "superellipse exponent must be positive");
        Self { n }
    }
}

impl Region2D for SuperellipseRegion {
    fn contains(&self, a: f64, b: f64) -> bool {
        a.abs().powf(self.n) + b.abs().powf(self.n) <= 1.0
    }

    fn area(&self) -> f64 {
        // Closed form: 4 · Γ(1 + 1/n)² / Γ(1 + 2/n)
        4.0 * gamma(1.0 + 1.0 / self.n).powi(2) / gamma(1.0 + 2.0 / self.n)
    }

    fn bounding_box_area(&self) -> f64 {
        4.0 // uniform over [-1,1]²
    }

    fn sample(&self, u: f64, v: f64) -> (f64, f64) {
        let mut u = u;
        let mut v = v;
        for _ in 0..32 {
            let a = u * 2.0 - 1.0;
            let b = v * 2.0 - 1.0;
            if self.contains(a, b) {
                return (a, b);
            }
            u = (u + 0.618033988749895).fract();
            v = (v + 0.618033988749895).fract();
        }
        (0.0, 0.0)
    }
}

/// Lanczos approximation of the Gamma function for `x > 0`.
///
/// Accurate to ~15 significant digits for typical inputs.
/// Coefficients from Numerical Recipes 3rd ed., §6.1.
fn gamma(mut x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: Γ(x) = π / (sin(πx) · Γ(1 − x))
        return PI / ((PI * x).sin() * gamma(1.0 - x));
    }
    x -= 1.0;
    const G: usize = 7;
    const C: [f64; G + 2] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let mut ag = C[0];
    for (i, ci) in C.iter().enumerate().skip(1) {
        ag += ci / (x + i as f64);
    }
    let t = x + G as f64 + 0.5;
    (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * ag
}
