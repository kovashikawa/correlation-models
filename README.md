# Advanced correlation measures

Modern dependence measures that go beyond Pearson's rho, with clean
numpy/scipy implementations and a benchmark against classic nonlinear
counterexamples. Companion repo for the blog post "Measuring Dependence
Beyond Pearson's rho".

## Measures included

| Measure | Detects | Range | Zero iff independent | Key reference |
|---|---|---|---|---|
| Pearson rho | linear only | [-1, 1] | no | Pearson (1895) |
| Spearman rho | monotone | [-1, 1] | no | Spearman (1904) |
| Kendall tau | monotone | [-1, 1] | no | Kendall (1938) |
| Chatterjee xi | any dependence | [0, 1] | yes | Chatterjee (2021) |
| Distance correlation | any dependence | [0, 1] | yes | Szekely, Rizzo, Bakirov (2007) |
| HSIC | any dependence (kernel) | [0, inf) | yes | Gretton et al. (2005) |
| KSG mutual information | any dependence | [0, inf) | yes | Kraskov, Stogbauer, Grassberger (2004) |
| MIC | any dependence | [0, 1] | yes | Reshef et al. (2011), caveats in Kinney and Atwal (2014) |
| Tail dependence lambda | joint extremes | [0, 1] | no | Sibuya (1959), Joe (1993) |

## Install

```bash
uv venv
uv pip install -r requirements.txt
```

Optional extras for cross-checking:

```bash
uv pip install dcor        # reference distance correlation implementation
uv pip install minepy      # reference MIC implementation
```

## Usage

```python
import numpy as np
from correlation_models import measures as m

rng = np.random.default_rng(0)
x = rng.normal(size=2000)
y = np.abs(x)                       # dependent, but Pearson = 0

print(f"Pearson          {m.pearson(x, y):.3f}")
print(f"Spearman         {m.spearman(x, y):.3f}")
print(f"Kendall          {m.kendall(x, y):.3f}")
print(f"Chatterjee xi    {m.chatterjee_xi(x, y):.3f}")
print(f"Distance corr    {m.distance_correlation(x, y):.3f}")
print(f"HSIC             {m.hsic(x, y):.4f}")
print(f"KSG mutual info  {m.mutual_information_ksg(x, y):.4f}")
```

## Benchmark

```bash
uv run python scripts/benchmark.py
```

Runs all measures on eight synthetic datasets (linear, quadratic, absolute
value, sine, circle, xor, independent, heavy tails) and prints a markdown
table. The point is visual: Pearson reports ~0 on |X|, circle, xor, and
quadratic, while every modern measure scores them as dependent.

## Notes on the measures

### Distance correlation (Szekely, Rizzo, Bakirov 2007)

Distance covariance is a norm on the difference between the joint
characteristic function and the product of the marginals. The estimator is
simple: double-center the pairwise distance matrices, take the mean product.
dCor = 0 if and only if independence, in any dimension, for distributions
with finite first moments. Cost is O(n^2) memory and time.

### Chatterjee's xi (2021)

A rank-based coefficient with a one-line interpretation: how well knowing X
narrows the conditional distribution of Y. Xi = 0 iff independence, xi = 1
iff Y is a measurable function of X. Asymmetric by construction, so order
matters: xi(X, Y) measures Y as a function of X.

### HSIC (Gretton et al. 2005)

Maps each variable into a reproducing kernel Hilbert space and measures the
norm of the cross-covariance operator between the two embeddings. With a
characteristic kernel (RBF here), HSIC = 0 iff independence. No density
estimation needed, which is why it is the workhorse of kernel feature
selection (Song et al. 2012).

### KSG mutual information (Kraskov, Stogbauer, Grassberger 2004)

k-nearest-neighbor estimator of mutual information, adaptive resolution in
both margins. I(X; Y) = 0 iff independence. Values are in bits and depend on
the marginal entropies, so it is a poor cross-dataset comparability measure
but an excellent detector. scikit-learn exposes it as
`mutual_info_regression`.

### MIC (Reshef et al. 2011)

Maximal information coefficient: maximize normalized mutual information over
all grid binning schemes, capped by sample size. Equitability claims were
contested, see Kinney and Atwal (2014), so treat MIC as one more detector,
not as a calibrated strength scale. `minepy` is the canonical implementation.

### Tail dependence (Joe 1993)

For risk work, global measures hide the tails. Tail dependence answers: given
that one variable is above its q-quantile, how likely is the other? The
Gaussian copula famously has zero tail dependence for any rho < 1, which is
why Pearson-only thinking understates joint tail risk. The estimator here is
the empirical version at a chosen quantile.

## When to use what

- Bivariate, monotone, want a signed measure: Spearman or Kendall.
- Bivariate, any shape, want a [0, 1] strength: distance correlation or
  Chatterjee xi.
- High-dimensional or multivariate vectors: distance correlation, HSIC.
- Feature selection, nonlinear screening: HSIC, KSG MI.
- Portfolio/risk: tail dependence on top of a global measure.
- Rule of thumb: never ship a dependence claim built on Pearson alone.

## References

1. Szekely, G. J., Rizzo, M. L., and Bakirov, N. K. (2007). Measuring and
   testing dependence by correlation of distances. Annals of Statistics.
2. Szekely, G. J. and Rizzo, M. L. (2009). Brownian distance covariance.
   Annals of Applied Statistics.
3. Chatterjee, S. (2021). A new coefficient of correlation. Journal of the
   American Statistical Association.
4. Gretton, A., Bousquet, O., Smola, A., and Scholkopf, B. (2005). Measuring
   statistical dependence with Hilbert-Schmidt norms. ALT.
5. Kraskov, A., Stogbauer, H., and Grassberger, P. (2004). Estimating mutual
   information. Physical Review E.
6. Reshef, D. N. et al. (2011). Detecting novel associations in large data
   sets. Science.
7. Kinney, J. B. and Atwal, G. S. (2014). Equitability, mutual information,
   and the maximal information coefficient. PNAS.
8. Joe, H. (1993). Multivariate models and dependence concepts. Chapman and
   Hall.
9. Song, L., Smola, A., Gretton, A., Bedo, J., and Borgwardt, K. (2012).
   Feature selection via dependence maximization. JMLR.
10. Edelmann, D., Móri, T. F., and Székely, G. J. (2021). On relationships
    between the Pearson and the distance correlation coefficients. Statistics
    and Probability Letters.
