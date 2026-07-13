// ================================================================
// § Sample1D — sum type for continuous vs discrete sampling results
//
// Reference: luxrays/mcdistribution.h lines 105-135
// ================================================================

/// Result of sampling a 1D distribution.
///
/// Explicitly distinguishes continuous from discrete sampling,
/// eliminating the out-parameter pattern.
#[derive(Clone, Copy, Debug)]
pub enum Sample1D {
    /// Continuous sample at position `x` ∈ [0, 1), its PDF, and the bucket index.
    Continuous { x: f64, pdf: f64, offset: usize },
    /// Discrete bucket at `index`, its PDF, and fractional remainder within the bucket.
    Discrete { index: usize, pdf: f64, du: f64 },
}

impl Sample1D {
    /// Extract the PDF value regardless of variant.
    pub fn pdf(&self) -> f64 {
        match self {
            Sample1D::Continuous { pdf, .. } | Sample1D::Discrete { pdf, .. } => *pdf,
        }
    }
}

/// 1D piecewise-constant distribution with CDF-based sampling.
/// Used internally by Dist2D for the marginal and conditional distributions.
pub struct Dist1D {
    /// Cumulative distribution function (CDF) values, length n+1.
    cdfs: Vec<f64>,
    /// Normalized function values (weights ≥ 0).
    funcs: Vec<f64>,
    /// Sum of all function values. Zero if all weights are zero (uniform fallback).
    total: f64,
}

impl Dist1D {
    /// Build a 1D distribution from raw weight values.
    /// Non-positive values are clamped to zero; a zero-total distribution samples uniformly.
    pub fn new(values: &[f64]) -> Self {
        let n = values.len();
        let mut funcs = values.to_vec();

        let total = funcs.iter_mut().fold(0., |mut acc, value| {
            let weight = value.max(0.0);
            *value = weight;
            acc += weight;
            acc
        });

        let mut cdfs = vec![0.; n + 1];
        if total == 0. {
            (0..=n).for_each(|i| {
                cdfs[i] = i as f64 / n as f64;
            })
        } else {
            for i in 1..=n {
                cdfs[i] = cdfs[i - 1] + funcs[i - 1] / total;
            }
            cdfs[n] = 1.0;
        }

        Self { cdfs, funcs, total }
    }

    /// Sample the distribution with a unit-random value `u` ∈ [0, 1).
    /// Returns (index, PDF_value) where PDF_value uses the [0, 1] sample-space measure.
    pub fn sample(&self, u: f64) -> (usize, f64) {
        let u_clamp = &u.clamp(0., 1.0 - 1e-10);
        let offset = self.cdfs.binary_search_by(|&val| {
            if val <= *u_clamp {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });
        let index = offset.unwrap_or_else(|idx| idx - 1);
        (index, self.pdf(index))
    }

    /// Evaluate the PDF at a given index. Returns 1.0 for the uniform fallback (zero total).
    pub fn pdf(&self, index: usize) -> f64 {
        if self.total == 0. {
            return 1.0;
        }

        (self.funcs[index] * self.count() as f64) / self.total
    }

    /// Number of bins in the distribution.
    pub fn count(&self) -> usize {
        self.funcs.len()
    }

    /// Sample a discrete bucket, returning a `Sample1D::Discrete`.
    pub fn sample_discrete(&self, u: f64) -> Sample1D {
        let (index, pdf) = self.sample(u);
        Sample1D::Discrete {
            index,
            pdf,
            du: 0.0,
        }
    }

    /// Sample a continuous position in [0, 1), returning a `Sample1D::Continuous`.
    ///
    /// Uses the same CDF as [`sample()`](Self::sample) but returns the
    /// interpolated continuous position within the bucket.
    pub fn sample_continuous(&self, u: f64) -> Sample1D {
        let n = self.count();
        if u <= 0.0 {
            return Sample1D::Continuous {
                x: 0.0,
                pdf: self.pdf(0),
                offset: 0,
            };
        }
        if u >= 1.0 - 1e-15 {
            return Sample1D::Continuous {
                x: 1.0 - 1e-15,
                pdf: self.pdf(n - 1),
                offset: n - 1,
            };
        }
        let pos = self
            .cdfs
            .binary_search_by(|&val| {
                if val <= u {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .unwrap_or_else(|idx| idx - 1)
            .min(n - 1);
        let du = if self.cdfs[pos + 1] > self.cdfs[pos] {
            ((u - self.cdfs[pos]) / (self.cdfs[pos + 1] - self.cdfs[pos])).min(1.0 - 1e-15)
        } else {
            0.0
        };
        let x = ((pos as f64 + du) / n as f64).min(1.0 - 1e-15);
        let pdf = if self.total == 0.0 {
            1.0
        } else {
            (self.funcs[pos] * n as f64) / self.total
        };
        Sample1D::Continuous {
            x,
            pdf,
            offset: pos,
        }
    }
}

/// 2D piecewise-constant distribution using a product of marginal + conditional 1D distributions.
/// Samples from the 2D CDF are drawn by first sampling the marginal (rows), then the conditional
/// (columns within the chosen row).
pub struct Dist2D {
    marginal: Dist1D,
    conditional: Vec<Dist1D>,
}

impl Dist2D {
    /// Build a 2D distribution from a flat array of shape (nv, nu) in row-major order.
    /// `nu` = columns (u-axis), `nv` = rows (v-axis).
    pub fn new(values: &[f64], nu: usize, nv: usize) -> Self {
        let mut row_sums = vec![0.; nv];
        for j in 0..nv {
            (0..nu).for_each(|i| {
                row_sums[j] += values[j * nu + i];
            });
        }
        let marginal = Dist1D::new(&row_sums);
        let conditional = (0..nv)
            .map(|j| {
                let row_start = j * nu;
                let row_end = row_start + nu;
                Dist1D::new(&values[row_start..row_end])
            })
            .collect();

        Self {
            marginal,
            conditional,
        }
    }

    /// Sample the 2D distribution with two unit-random values (u, v).
    /// Returns (column, row, PDF_value). `u` selects the column within the row,
    /// `v` selects the row from the marginal distribution.
    pub fn sample(&self, u: f64, v: f64) -> (usize, usize, f64) {
        let (row, marginal_pdf) = self.marginal.sample(v);

        let (col, conditional_pdf) = self.conditional[row].sample(u);

        let pdf = marginal_pdf * conditional_pdf;

        (col, row, pdf)
    }

    /// Evaluate the PDF at pixel (i, j) in the [0, 1]² sample-space measure.
    pub fn pdf(&self, i: usize, j: usize) -> f64 {
        self.marginal.pdf(j) * self.conditional[j].pdf(i)
    }
}
