# Experiments

Numerical study for the latent-space DRO paper (`../latex/main.tex`).

```
python3 p1_portfolio.py          # P1 sweep  -> results/p1_rows.json
python3 p2_rbdo.py               # P2 sweep  -> results/p2_rows.json
python3 p2_optima.py             # P2 per-shift optima (for the regret metric)
python3 analyze.py               # Fig 1, Fig 2, tables
python3 diagnostics.py theory    # Fig 4
python3 diagnostics.py manifold  # Fig 3
python3 diagnostics.py oracle    # Fig 5
```

Add `--quick` to either sweep for a fast single-seed run. Everything is CPU-only and
float64; a full pass is about 40 min per problem.

## Layout

| file | what it holds |
|---|---|
| `dro/flows.py` | the generator `G_theta`: a RealNVP-style flow with base `N(0, I)` |
| `dro/transforms.py` | latent transforms `T_eta` and their KL to the base (shift, diagonal affine, full affine, coupling) |
| `dro/surrogate.py` | the branch--trunk surrogate of Section 4 |
| `dro/method.py` | Algorithm 1, the trust-region inner solve, the shared optimizer helper |
| `dro/baselines.py` | every comparison method |
| `dro/problems.py` | P1 (portfolio) and P2 (short column), their shifts, and exact reference densities |
| `dro/plotting.py` | figure style; color encodes the *family* of method |

## The two problems

**P1, mean-risk portfolio.** Esfahani--Kuhn factor model, `xi_i = psi + zeta_i`. Convex,
exact loss, no surrogate, so differences between methods are attributable to the ambiguity
set alone. Three departures from the published numbers, each forced by a degeneracy we ran
into and each documented in `PortfolioProblem`:

* A risk-free asset is needed. Without it the weights sum to one over the risky assets
  alone, a common downward shift of every mean is a *constant* added to the loss, and
  since that shift is also the most divergence-efficient move available, the adversary
  spends its whole budget somewhere that cannot move the optimum.
* Reward-to-risk has to vary across assets. In the original every asset has ratio 1.2,
  which puts the optimum at nearly equal weights with nothing to exploit differentially.
* The objective has to be neither positively homogeneous (mean-CVaR is, which corners the
  cash/risky split) nor infinite under heavy tails (exponential utility is). Return
  penalized by shortfall against a fixed target is neither.

**P2, RBDO short column.** Kuschel--Rackwitz limit state; the objective is
`area + lam * P_f`, so the loss is an indicator and the sigmoid-link surrogate is what
supplies differentiability in the design. The limit state is analytic, so `P_f`, the
worst-case distribution and the robust design are all checkable by brute force. Shifts
move the two bending moments differentially, which changes the optimal *aspect ratio* and
not merely the optimal size -- without that, every form of robustness has the same
response ("make it bigger") and no ambiguity set can be told from another.

## Two traps worth knowing about

**Adam must not be used under a simplex projection.** Its update is sign-like, so when
every coordinate of the gradient shares a sign the step is a near-uniform translation --
exactly what projection onto the simplex annihilates. The iterate then never leaves its
starting point, silently. `make_optimizer` picks momentum SGD for simplex-constrained
designs and Adam for box-constrained ones.

**The chi-square DRO weights need two parameters, not one.** The maximizer is
`w_i propto (1 + (l_i - eta)/beta)_+`; fixing `eta` at the mean gives a family that
converges to `w propto (l - lbar)_+` and therefore cannot reach a large radius at all,
which quietly caps the baseline instead of erroring.

## Evaluation

Every method is tuned only through its own single radius knob, and we report the whole
frontier rather than a chosen point. The headline metric is worst-case **regret** over a
family of held-out shifts of the true data-generating process -- shifts no method's
adversary ever saw. Regret subtracts, per shift, the best achievable objective on that
shift, which removes the component of a shift that hurts every design equally and would
otherwise make the metric a restatement of nominal performance.
