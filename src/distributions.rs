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
