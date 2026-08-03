/// Result of sampling a 1D distribution.
///
/// Explicitly distinguishes continuous from discrete sampling,
/// eliminating the out-parameter pattern.
///
/// Reference: luxrays/mcdistribution.h lines 105-135
#[derive(Clone, Copy, Debug)]
pub enum Sample1D {
    /// Continuous sample at position `x` ∈ [0, 1), its PDF, and the bucket index.
    Continuous { x: f32, pdf: f32, offset: usize },
    /// Discrete bucket at `index`, its PDF, and fractional remainder within the bucket.
    Discrete { index: usize, pdf: f32, du: f32 },
}

impl Sample1D {
    /// Extract the PDF value regardless of variant.
    pub fn pdf(&self) -> f32 {
        match self {
            Sample1D::Continuous { pdf, .. } | Sample1D::Discrete { pdf, .. } => *pdf,
        }
    }

    /// Extract the bucket index (continuous offset or discrete index).
    pub fn index(&self) -> usize {
        match self {
            Sample1D::Continuous { offset, .. } => *offset,
            Sample1D::Discrete { index, .. } => *index,
        }
    }
}

/// Stack-allocated 1D piecewise-constant distribution with `N` bins known at compile time.
/// Reference: luxrays Distribution1DFixed, pbrt-v4
#[derive(Clone, Debug)]
pub struct FixedDist1D<const N: usize> {
    /// Raw function values (clamped to ≥ 0).  Un-normalized.
    func: [f32; N],
    /// Cumulative distribution: `cdf[i]` = sum of normalized func for [0..=i].
    cdf: [f32; N],
    /// Sum of raw input weights.  Zero-total triggers uniform fallback.
    func_int: f32,
    /// 1 / N — precomputed.
    inv_count: f32,
}

impl<const N: usize> FixedDist1D<N> {
    /// Build from raw weight values.  Non-positive weights are clamped to zero.
    pub fn new(f: &[f32; N]) -> Self {
        let inv_count = 1.0 / N as f32;
        let mut func = [0.0f32; N];
        let mut total = 0.0f32;
        for (i, &v) in f.iter().enumerate() {
            let w = v.max(0.0);
            func[i] = w;
            total += w;
        }

        let mut cdf = [0.0f32; N];
        if total > 0.0 {
            let inv_total = 1.0 / total;
            let mut running = 0.0f32;
            for i in 0..N {
                running += func[i] * inv_total;
                cdf[i] = running;
            }
        } else {
            // Uniform fallback
            for i in 0..N {
                func[i] = 1.0;
                cdf[i] = (i + 1) as f32 * inv_count;
            }
        }

        FixedDist1D {
            func,
            cdf,
            func_int: total,
            inv_count,
        }
    }

    /// Integral of the original function over [0, 1).
    #[inline]
    pub fn integral(&self) -> f32 {
        self.func_int
    }

    /// Bucket index for `u` ∈ [0, 1).
    #[inline]
    pub fn offset(&self, u: f32) -> usize {
        ((u * N as f32) as usize).min(N - 1)
    }

    /// Continuous PDF at position `u` ∈ [0, 1) — the density of the piecewise-constant
    /// distribution, which integrates to 1 over [0, 1).
    ///
    /// For a uniform (zero-total) distribution returns 1.0.
    #[inline]
    pub fn pdf_continuous(&self, u: f32) -> f32 {
        self.pdf_discrete(self.offset(u))
    }

    /// PDF (continuous density) for bucket at `offset`.
    ///
    /// Returns the same value as [`pdf_continuous`](Self::pdf_continuous) for any position inside
    /// the same bucket — the piecewise-constant value of the density over that bin.
    ///
    /// For a uniform (zero-total) distribution returns 1.0.
    #[inline]
    pub fn pdf_discrete(&self, offset: usize) -> f32 {
        if self.func_int == 0.0 {
            1.0
        } else {
            self.func[offset] * (N as f32) / self.func_int
        }
    }

    /// Sample a continuous position `x` ∈ [0, 1) from this distribution.
    pub fn sample_continuous(&self, u: f32) -> Sample1D {
        let n = N;
        if u <= 0.0 {
            return Sample1D::Continuous {
                x: 0.0,
                pdf: self.pdf_discrete(0),
                offset: 0,
            };
        }
        if u >= 1.0 {
            return Sample1D::Continuous {
                x: 1.0 - 1e-15,
                pdf: self.pdf_discrete(n - 1),
                offset: n - 1,
            };
        }
        let pos = self
            .cdf
            .partition_point(|&c| c <= u)
            .saturating_sub(1)
            .min(n - 1);
        let cdf_low = if pos == 0 { 0.0 } else { self.cdf[pos - 1] };
        let cdf_high = self.cdf[pos];
        let du = if cdf_high > cdf_low {
            ((u - cdf_low) / (cdf_high - cdf_low)).min(1.0 - 1e-15)
        } else {
            0.0
        };
        let x = ((pos as f32 + du) * self.inv_count).min(1.0 - 1e-15);
        Sample1D::Continuous {
            x,
            pdf: self.pdf_discrete(pos),
            offset: pos,
        }
    }

    /// Sample a discrete bucket index from this distribution.
    pub fn sample_discrete(&self, u: f32) -> Sample1D {
        let n = N;
        if u <= 0.0 {
            return Sample1D::Discrete {
                index: 0,
                pdf: self.pdf_discrete(0),
                du: 0.0,
            };
        }
        if u >= 1.0 {
            return Sample1D::Discrete {
                index: n - 1,
                pdf: self.pdf_discrete(n - 1),
                du: 1.0,
            };
        }
        let pos = self
            .cdf
            .partition_point(|&c| c <= u)
            .saturating_sub(1)
            .min(n - 1);
        let cdf_low = if pos == 0 { 0.0 } else { self.cdf[pos - 1] };
        let cdf_high = self.cdf[pos];
        let du = if cdf_high > cdf_low {
            ((u - cdf_low) / (cdf_high - cdf_low)).min(1.0)
        } else {
            0.0
        };
        Sample1D::Discrete {
            index: pos,
            pdf: self.pdf_discrete(pos),
            du,
        }
    }
}

// ================================================================

/// Stack-allocated 2D piecewise-constant distribution with `NU × NV`
/// bins known at compile time.
/// Composed of a marginal `FixedDist1D<NV>` (rows) and NV conditional
/// `FixedDist1D<NU>` (columns within each row).
///
/// Reference: luxrays Distribution2D, pbrt-v4
#[derive(Clone, Debug)]
pub struct FixedDist2D<const NU: usize, const NV: usize> {
    /// Marginal distribution over rows (v-axis).
    marginal: FixedDist1D<NV>,
    /// Per-row conditional distributions over columns (u-axis).
    conditional: [FixedDist1D<NU>; NV],
}

impl<const NU: usize, const NV: usize> FixedDist2D<NU, NV> {
    /// Build from a flat array of shape `(NV, NU)` in row-major order.
    ///
    /// # Panics Panics if `values.len() != NU * NV`.
    pub fn new(values: &[f32]) -> Self {
        assert_eq!(values.len(), NU * NV);
        // Row sums for the marginal (NV rows)
        let row_sums: [f32; NV] = core::array::from_fn(|j| {
            let start = j * NU;
            values[start..start + NU].iter().sum()
        });
        let marginal = FixedDist1D::<NV>::new(&row_sums);

        // Per-row conditional distributions (NU columns each)
        let conditional: [FixedDist1D<NU>; NV] = core::array::from_fn(|j| {
            let start = j * NU;
            let mut row = [0.0f32; NU];
            row.copy_from_slice(&values[start..start + NU]);
            FixedDist1D::<NU>::new(&row)
        });

        FixedDist2D {
            marginal,
            conditional,
        }
    }

    /// Joint PDF (continuous density) at pixel `(col, row)`.
    ///
    /// This is the value of the 2D piecewise-constant density over [0, 1)² — the product of
    /// marginal and conditional densities, integrating to 1 over the unit square.
    pub fn pdf(&self, col: usize, row: usize) -> f32 {
        self.marginal.pdf_discrete(row) * self.conditional[row].pdf_discrete(col)
    }

    /// Sample a pixel index `(col, row)` from the distribution.
    ///
    /// Returns `(col, row, pdf)` where `pdf` is the continuous density value for the sampled pixel.
    pub fn sample(&self, u: f32, v: f32) -> (usize, usize, f32) {
        let row_sample = self.marginal.sample_discrete(v);
        let col_sample = self.conditional[row_sample.index()].sample_discrete(u);
        (
            col_sample.index(),
            row_sample.index(),
            row_sample.pdf() * col_sample.pdf(),
        )
    }
}

// ================================================================

/// Heap-allocated 1D piecewise-constant distribution with runtime-sized
/// bins. Use [`FixedDist1D`] when N is known at compile time.
///
/// Reference: luxrays Distribution1D, pbrt-v4
pub struct Dist1D {
    /// Cumulative distribution function (CDF) values, length n + 1.
    cdfs: Vec<f32>,
    /// Raw function values (weights ≥ 0).  Un-normalized.
    funcs: Vec<f32>,
    /// Sum of all function values.  Zero-total triggers uniform fallback.
    total: f32,
}

impl Dist1D {
    /// Build from raw weight values.  Non-positive values are clamped to zero; a zero-total
    /// distribution samples uniformly.
    pub fn new(values: &[f32]) -> Self {
        let n = values.len();
        let mut funcs = values.to_vec();

        let total = funcs.iter_mut().fold(0.0, |mut acc, value| {
            let weight = value.max(0.0);
            *value = weight;
            acc += weight;
            acc
        });

        let mut cdfs = vec![0.0; n + 1];
        if total == 0.0 {
            (0..=n).for_each(|i| {
                cdfs[i] = i as f32 / n as f32;
            })
        } else {
            for i in 1..=n {
                cdfs[i] = cdfs[i - 1] + funcs[i - 1] / total;
            }
            cdfs[n] = 1.0;
        }

        Self { cdfs, funcs, total }
    }

    /// Integral of the original function over [0, 1).
    pub fn integral(&self) -> f32 {
        self.total
    }

    /// Number of bins in the distribution.
    pub fn count(&self) -> usize {
        self.funcs.len()
    }

    /// Continuous PDF value (density) for bin `index`.
    ///
    /// This is the value of the piecewise-constant density for any point inside bin `index`, such
    /// that `∫₀¹ p(x) dx = 1`. For a uniform (zero-total) distribution returns 1.0.
    #[inline]
    pub fn pdf_discrete(&self, index: usize) -> f32 {
        if self.total == 0.0 {
            return 1.0;
        }
        (self.funcs[index] * self.count() as f32) / self.total
    }

    /// Continuous PDF at position `u` ∈ [0, 1).
    #[inline]
    pub fn pdf_continuous(&self, u: f32) -> f32 {
        let n = self.count();
        let idx = (u * n as f32).min((n - 1) as f32) as usize;
        self.pdf_discrete(idx)
    }

    /// Sample a discrete bucket from the distribution.
    ///
    /// Returns `Sample1D::Discrete` with the bucket index and the continuous PDF value for any
    /// point in that bucket.
    pub fn sample_discrete(&self, u: f32) -> Sample1D {
        let (index, pdf) = self.sample_internal(u);
        Sample1D::Discrete {
            index,
            pdf,
            du: 0.0,
        }
    }

    /// Sample a continuous position in [0, 1), returning `Sample1D::Continuous`.
    pub fn sample_continuous(&self, u: f32) -> Sample1D {
        let n = self.count();
        if u <= 0.0 {
            return Sample1D::Continuous {
                x: 0.0,
                pdf: self.pdf_discrete(0),
                offset: 0,
            };
        }
        if u >= 1.0 - 1e-15 {
            return Sample1D::Continuous {
                x: 1.0 - 1e-15,
                pdf: self.pdf_discrete(n - 1),
                offset: n - 1,
            };
        }
        // f32 1e-10 rounds to 1.0, so clamp strictly below 1.0.
        let u_clamp = u.clamp(0.0, 1.0 - f32::EPSILON);
        let pos = self
            .cdfs
            .binary_search_by(|&val| {
                if val <= u_clamp {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .unwrap_or_else(|idx| idx.saturating_sub(1))
            .min(n - 1);
        let du = if self.cdfs[pos + 1] > self.cdfs[pos] {
            ((u_clamp - self.cdfs[pos]) / (self.cdfs[pos + 1] - self.cdfs[pos])).min(1.0 - 1e-15)
        } else {
            0.0
        };
        let x = ((pos as f32 + du) / n as f32).min(1.0 - 1e-15);
        Sample1D::Continuous {
            x,
            pdf: self.pdf_discrete(pos),
            offset: pos,
        }
    }

    /// Internal: sample a bucket, returning `(index, pdf)`.
    #[inline]
    fn sample_internal(&self, u: f32) -> (usize, f32) {
        if self.funcs.is_empty() {
            return (0, 1.0);
        }
        let u_clamp = u.clamp(0.0, 1.0 - f32::EPSILON);
        let offset = self.cdfs.binary_search_by(|&val| {
            if val <= u_clamp {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });
        let index = offset
            .unwrap_or_else(|idx| idx.saturating_sub(1))
            .min(self.funcs.len() - 1);
        (index, self.pdf_discrete(index))
    }
}

/// Heap-allocated 2D piecewise-constant distribution with runtime-sized bins.
///
/// Use [`FixedDist2D`] when NU, NV are known at compile time.
///
/// Composed of a marginal `Dist1D` (rows) and per-row conditional `Dist1D` (columns within each
/// row), matching the luxrays / pbrt-v4 design: marginal selects the row, then the conditional for
/// that row selects the column.
pub struct Dist2D {
    /// Marginal distribution over rows (v-axis).
    marginal: Dist1D,
    /// Per-row conditional distributions over columns (u-axis).
    conditional: Vec<Dist1D>,
}

impl Dist2D {
    /// Build from a flat array of shape `(nv, nu)` in row-major order.
    /// `nu` = columns (u-axis), `nv` = rows (v-axis).
    pub fn new(values: &[f32], nu: usize, nv: usize) -> Self {
        let mut row_sums = vec![0.0; nv];
        for j in 0..nv {
            for i in 0..nu {
                row_sums[j] += values[j * nu + i];
            }
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

    /// Joint PDF (continuous density) at pixel `(col, row)`.
    ///
    /// Product of marginal and conditional densities, integrating to 1 over [0, 1)².
    pub fn pdf(&self, col: usize, row: usize) -> f32 {
        self.marginal.pdf_discrete(row) * self.conditional[row].pdf_discrete(col)
    }

    /// Sample a pixel index from the distribution.
    ///
    /// Returns `(col, row, pdf)` where `pdf` is the continuous density value for the sampled pixel.
    pub fn sample(&self, u: f32, v: f32) -> (usize, usize, f32) {
        let row_sample = self.marginal.sample_discrete(v);
        let col_sample = self.conditional[row_sample.index()].sample_discrete(u);
        (
            col_sample.index(),
            row_sample.index(),
            row_sample.pdf() * col_sample.pdf(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist1d_boundaries_do_not_panic() {
        let values: Vec<f32> = (0..2048)
            .map(|i| (i as f32 + 1.0).sin().abs() + 0.001)
            .collect();
        let dist = Dist1D::new(&values);
        for u in [0.0f32, 0.5, 1.0 - f32::EPSILON, 1.0, 1.5] {
            let s = dist.sample_discrete(u);
            assert!(
                s.index() < dist.count(),
                "index {} out of bounds",
                s.index()
            );
            assert!(s.pdf().is_finite() && s.pdf() >= 0.0, "bad pdf");
            let s = dist.sample_continuous(u);
            assert!(
                s.index() < dist.count(),
                "index {} out of bounds",
                s.index()
            );
            assert!(s.pdf().is_finite() && s.pdf() >= 0.0, "bad pdf");
        }
    }

    #[test]
    fn dist2d_boundaries_do_not_panic() {
        let nu = 128;
        let nv = 64;
        let data: Vec<f32> = (0..nv * nu)
            .map(|k| ((k % nu) as f32 + 1.0).sin().abs() + 0.001)
            .collect();
        let dist = Dist2D::new(&data, nu, nv);
        for (u, v) in [(0.0f32, 0.0f32), (1.0, 1.0), (1.5, 0.7), (0.3, 1.5)] {
            let (col, row, pdf) = dist.sample(u, v);
            assert!(col < nu, "col {col} out of bounds");
            assert!(row < nv, "row {row} out of bounds");
            assert!(pdf.is_finite() && pdf >= 0.0, "bad pdf {pdf}");
        }
    }
}
