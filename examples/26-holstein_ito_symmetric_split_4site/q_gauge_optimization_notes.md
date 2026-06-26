# Periodic optimization of the scalar diffusion gauge $q$

Companion derivation for the implementation in
`ipie/addons/free_projection/propagation/ito_second_order_fp.py`
(`ItoSymmSplitImportancePropagatorFP`). It translates §11 ("Optimizing the
diffusion gauge: a Green–Kubo objective") of `noise_gauges.pdf`, specialised to
Einstein phonons in §11.5, into concrete per-walker quantities that the code can
accumulate, and proves that refreshing $q$ from the population leaves the
estimator unbiased.

Throughout, the walker is the single-D2 (or, for the Toyozawa trial, the
lattice-contracted *dressed* single-D2 of §10.4) component
$\lvert\psi,f\rangle$. Modes are indexed by $\mu,\rho=1,\dots,N_m$; for Holstein
the mode index equals the lattice site and $\omega_\mu\equiv\omega_0$ is flat.
We work on the **zero-noise manifold** (Prop. 7.3),
$$
\lambda = B + Q\bar A,\qquad M\bar M^{\mathsf T}=Q,\qquad \bar a = M^{\mathsf T}A,
$$
and restrict to a **scalar** gauge $Q=qI$, $q>0$, i.e. $M=\sqrt q\,I$ and
$\lambda = B + q\,\bar A$. This is exactly the parametrisation already in the
code: `delta_lambda = B + q * A.conj()`, `M = sqrt(q) I`, and the force bias is
evaluated in the gauge-transformed variables $\widetilde A=\sqrt q\,A$,
$\widetilde B=(B-\lambda)/\sqrt q$.

---

## 1. The objective and its scalar optimum

For a time-averaged estimator the long-run error is the zero-frequency
Green–Kubo weight (§11.1). On the zero-noise manifold the weight is quiet at
leading order, so the error flows entirely through *trajectory* fluctuations of
the walker coordinates. §11.2 collects these into two Hermitian-PSD cost metrics
and a single objective
$$
J(Q)=\operatorname{tr}(W_{\mathrm{ph}}\,Q)+\operatorname{tr}(V\,Q^{-1}),
\qquad V\equiv W_{\mathrm{el}}^{\mathsf T},
$$
with the closed-form matrix-geometric-mean optimum of Thm. 11.6. For a **scalar**
$Q=qI$ this collapses to a one-dimensional convex problem.

**Proposition 1 (scalar optimum).**
*Let $a:=\operatorname{tr}W_{\mathrm{ph}}>0$ and $b:=\operatorname{tr}V>0$. Then*
$$
J(q)=a\,q+\frac{b}{q}\quad(q>0)
$$
*is strictly convex with unique minimiser*
$$
\boxed{\,q^\star=\sqrt{b/a}=\sqrt{\operatorname{tr}V/\operatorname{tr}W_{\mathrm{ph}}}\,},
\qquad J(q^\star)=2\sqrt{ab},
$$
*and the suboptimality of the neutral gauge $q=1$ is*
$$
\frac{J(1)}{J(q^\star)}=\frac{a+b}{2\sqrt{ab}}=\frac{1+r}{2\sqrt r},\qquad r=b/a.
$$

*Proof.* $J'(q)=a-b/q^2$, vanishing at $q^2=b/a$; $J''(q)=2b/q^3>0$ on $q>0$, so
the critical point is the unique global minimum. Substituting gives
$J(q^\star)=a\sqrt{b/a}+b\sqrt{a/b}=2\sqrt{ab}$, and $J(1)=a+b$. $\square$

This is the Holstein scalar gauge of §11.5: $q^\star=\sqrt{\operatorname{tr}V/\operatorname{tr}W_{\mathrm{ph}}}$
with gain $(1+r)/2\sqrt r$. The implementation accumulates $a$ and $b$ over the
walker population and sets $q\leftarrow q^\star$.

---

## 2. The phonon cost $\operatorname{tr}W_{\mathrm{ph}}$

§11.2 gives $W_{\mathrm{ph}}=\Omega^{-1}\,s\,s^\dagger\,\Omega^{-1}$ with
$\Omega=\operatorname{diag}(\omega_\mu)$ and phonon sensitivities
$s_\mu=\partial E_L/\partial f_\mu$ (the OU coordinate of §11.1 is the walker
displacement $f$). Hence, per walker,
$$
\operatorname{tr}W_{\mathrm{ph}}
=\operatorname{tr}(\Omega^{-1}ss^\dagger\Omega^{-1})
=s^\dagger\Omega^{-2}s
=\sum_\mu\frac{\lvert s_\mu\rvert^2}{\omega_\mu^2}
\;\xrightarrow[\text{Einstein}]{}\;
\frac{\lVert s\rVert^2}{\omega_0^2}.
$$

**Proposition 2 (phonon sensitivity).**
*For a single-D2 trial with overlap $O=(\phi^\dagger\psi)\,e^{\beta^\dagger f}$,
local energy*
$$
E_L=\langle h_{\mathrm{eff}}(f)\rangle_{\phi\psi}
+\sum_\nu\omega_\nu f_\nu\beta^*_\nu
+\sum_\nu\beta^*_\nu\langle G_\nu^\dagger\rangle_{\phi\psi},
\qquad
h_{\mathrm{eff}}(f)=h+\sum_\nu f_\nu G_\nu,
$$
*(with $\langle X\rangle_{\phi\psi}=\phi^\dagger X\psi/\phi^\dagger\psi$), one has*
$$
\boxed{\,s_\mu=\frac{\partial E_L}{\partial f_\mu}
=\omega_\mu\,\beta^*_\mu+\langle G_\mu\rangle_{\phi\psi}
=\omega_\mu A_\mu+\langle G_\mu\rangle_{\phi\psi}\,}.
$$
*If in addition $G_\mu=G_\mu^\dagger$ (Hermitian coupling, e.g. Holstein
$G_\mu=g\,c_\mu^\dagger c_\mu$), then $\langle G_\mu\rangle_{\phi\psi}=\langle G_\mu^\dagger\rangle_{\phi\psi}=B_\mu$, so*
$$
s_\mu=\omega_\mu A_\mu+B_\mu .
$$

*Proof.* $\phi,\psi$ are independent of $f$, so only the two explicit
$f$-dependences contribute: $\partial_{f_\mu}\langle h_{\mathrm{eff}}\rangle
=\langle\partial_{f_\mu}h_{\mathrm{eff}}\rangle=\langle G_\mu\rangle$, and
$\partial_{f_\mu}\!\sum_\nu\omega_\nu f_\nu\beta^*_\nu=\omega_\mu\beta^*_\mu$. The
creation term $\sum_\nu\beta^*_\nu\langle G_\nu^\dagger\rangle$ is $f$-independent.
Finally $A_\mu=\beta^*_\mu$ (Eq. 1.6 of the notes) and, for Hermitian $G_\mu$,
the operator identity $G_\mu=G_\mu^\dagger$ makes the two mixed matrix elements
literally equal. $\square$

Both $A_\mu$ and $B_\mu$ are already produced by
`trial.calc_ito_log_derivatives`. The code therefore forms
$s=\omega_0 A+B$ for Holstein, and falls back to the general
$s=\omega A+\langle G\rangle$ (with $\langle G\rangle$ obtained from the full,
non-residual coupling tensor) when the coupling is non-Hermitian.

> **Dressed (Toyozawa) trial.** For the momentum-projected dD2 trial the same
> formula holds verbatim with $A,B$ the $\rho$-weighted derivatives of Eq. (10.6)
> — i.e. exactly what `calc_ito_log_derivatives` returns — because §10.4 shows
> the dD2 overlap is a single-D2 overlap with an $f$-dependent dressed bra
> $\widetilde\phi$, and $\langle\cdot\rangle$ is then the dressed mixed
> expectation. The $f$-dependence of $\widetilde\phi$ produces the shift already
> baked into $A$; it does not add a term to $s_\mu$ at the level used here (we
> neglect $\partial\widetilde\phi/\partial f$ in the cost metric, consistent with
> the frozen-coefficient approximation of §11.2).

---

## 3. The electron cost $\operatorname{tr}V$

§11.2: $W_{\mathrm{el}}\approx u\,u^\dagger/\Delta^2$ with channel sensitivities
$$
u_\rho=\sum_i\frac{\partial E_L}{\partial\psi_i}\,[(G_\rho^\dagger-\lambda_\rho)\psi]_i,
$$
$\Delta$ the spectral gap of $h_{\mathrm{eff}}$, and $V=W_{\mathrm{el}}^{\mathsf T}$.
Then $\operatorname{tr}V=\operatorname{tr}W_{\mathrm{el}}=\lVert u\rVert^2/\Delta^2$.

**Proposition 3 (the channel sensitivity is a connected correlator and is
$\lambda$-independent).**
*Write the electronic local-energy operator*
$$
\widehat O := h_{\mathrm{eff}}(f)+\sum_\nu A_\nu\,G_\nu^\dagger,
\qquad
E_{L,\mathrm{el}}:=\langle\widehat O\rangle_{\phi\psi},
$$
*so that $E_L=E_{L,\mathrm{el}}+\sum_\nu\omega_\nu f_\nu\beta^*_\nu$ and
$\partial E_L/\partial\psi_i=\partial E_{L,\mathrm{el}}/\partial\psi_i$. Then for
a single-electron determinant ($O=\phi^\dagger\psi$)*
$$
\boxed{\,u_\rho=\langle\widehat O\,G_\rho^\dagger\rangle_{\phi\psi}
-\langle\widehat O\rangle_{\phi\psi}\,\langle G_\rho^\dagger\rangle_{\phi\psi}
=\operatorname{Cov}_{\phi\psi}\!\big(\widehat O,\,G_\rho^\dagger\big)\,},
$$
*which is independent of the splitting $\lambda$.*

*Proof.* With $O=\phi^\dagger\psi$ a scalar,
$E_{L,\mathrm{el}}=\phi^\dagger\widehat O\psi/\phi^\dagger\psi$ and
$$
\frac{\partial E_{L,\mathrm{el}}}{\partial\psi_i}
=\frac{(\phi^\dagger\widehat O)_i}{\phi^\dagger\psi}
-\frac{(\phi^\dagger\widehat O\psi)(\phi^\dagger)_i}{(\phi^\dagger\psi)^2}
=\frac{1}{\phi^\dagger\psi}\Big[(\phi^\dagger\widehat O)_i-E_{L,\mathrm{el}}(\phi^\dagger)_i\Big].
$$
Contract with $[(G_\rho^\dagger-\lambda_\rho)\psi]_i$ and use
$\phi^\dagger(G_\rho^\dagger-\lambda_\rho)\psi/\phi^\dagger\psi=B_\rho-\lambda_\rho$
and $\phi^\dagger\widehat O(G_\rho^\dagger-\lambda_\rho)\psi/\phi^\dagger\psi
=\langle\widehat O G_\rho^\dagger\rangle-\lambda_\rho E_{L,\mathrm{el}}$:
$$
u_\rho=\big(\langle\widehat O G_\rho^\dagger\rangle-\lambda_\rho E_{L,\mathrm{el}}\big)
-E_{L,\mathrm{el}}\big(B_\rho-\lambda_\rho\big)
=\langle\widehat O G_\rho^\dagger\rangle-E_{L,\mathrm{el}}B_\rho,
$$
and $B_\rho=\langle G_\rho^\dagger\rangle$, giving the connected correlator. The
$\lambda_\rho$ terms cancel identically. $\square$

This is the rigorous content of "the kick$\cdot(\partial E_L/\partial\psi)$
channel": the $\lambda$-dependence of the *kick* drops out of the *sensitivity*,
leaving a state functional (the connected correlator of the local-energy
operator with the channel creation operator) — exactly the connected structure
flagged in §9.3. For one electron every $\langle\cdot\rangle$ is a plain
$\phi^\dagger(\cdots)\psi/\phi^\dagger\psi$ matrix element, so
$$
u_\rho=\frac{\phi^\dagger\widehat O\,G_\rho^\dagger\psi}{\phi^\dagger\psi}
-\Big(\frac{\phi^\dagger\widehat O\psi}{\phi^\dagger\psi}\Big)
\Big(\frac{\phi^\dagger G_\rho^\dagger\psi}{\phi^\dagger\psi}\Big),
$$
needs no two-particle reduced density matrix and is what the code computes.

> **Multi-electron fallback.** For $>1$ electron, $u_\rho$ remains
> $\operatorname{Cov}_{\phi\psi}(\widehat O,G_\rho^\dagger)$ but requires the
> mixed 2-RDM (Wick) to evaluate $\langle\widehat O G_\rho^\dagger\rangle$. Since
> *any* adapted $q$ is unbiased (§4), the code uses the channel-kick magnitude
> $\lVert(G_\rho^\dagger-\lambda_\rho)\psi\rVert$ — the seesaw carrier
> $\mathbb E[\eta\eta^\dagger]=\bar Q^{-1}d\tau$ of §9.3 and the electron weight
> used in the diagonal recipe (Cor. 11.7) — as a robust proxy. This is a
> variance choice, never a bias.

**The gap $\Delta$.** $h_{\mathrm{eff}}(f)=T+\sum_\mu f_\mu G_\mu$ is non-Hermitian
(complex $f$), so its eigenvalues are complex; the imaginary-time contraction
rate is governed by the real parts. We take
$\Delta=\mathrm{Re}\,E_1-\mathrm{Re}\,E_0$, the gap between the two
lowest-real-part eigenvalues, floored at a small $\varepsilon_\Delta$ to avoid
division blow-up at near-degeneracies (the trust region of §4 absorbs the rest).

---

## 4. Population average and freedom from bias

Per walker $w$ define $a_w=\sum_\mu\lvert s_{w\mu}\rvert^2/\omega_\mu^2$ and
$b_w=\lVert u_w\rVert^2/\Delta_w^2$. Minimising the **population-summed**
objective $\sum_w J_w(q)=q\sum_w a_w+q^{-1}\sum_w b_w$ gives, by Prop. 1,
$$
\boxed{\,q^\star=\sqrt{\dfrac{\sum_w b_w}{\sum_w a_w}}\,},
$$
with the two sums reduced across MPI ranks. (Summing the costs, rather than
averaging $q_w^\star$, is the correct aggregation: it is the exact optimum of the
total Green–Kubo objective of the ensemble.)

**Proposition 4 (the refresh does not bias the estimator).**
*Refreshing $q$ every $S$ steps from the pre-noise walker population, then holding
it fixed for the block, leaves every walker expectation exact up to the same
$O(1/N_w)$ ensemble coupling as the energy-shift estimate.*

*Proof.* Thm. 4.6 (finite-step exactness) states that for any $(M,\lambda)$ that
is **adapted** — measurable with respect to the information available *before*
the step's noise $\Delta\widetilde Z$ is drawn — and held fixed within the step,
$$
\mathbb E_{\Delta Z}\big[\lvert\psi',f'\rangle\big]=e^{-sH_+}\lvert\psi,f\rangle
$$
exactly, **with no weight compensation** (Rmk. 4.2: the splitting needs none, and
the mixing changes only the quadratic variation, which was never constrained).
A scalar $q$ computed from the walker ensemble at the start of step $k$ is a
deterministic function of $\{\psi_w,f_w\}$ *before* step $k$'s noise, hence
adapted; holding it fixed across the block keeps it adapted at every step. The
only departure from a per-walker gauge is that $q$ now depends on *other*
walkers' coordinates. This is the ensemble coupling of Rmk. 11.10: it enters at
$O(1/N_w)$, identically to estimating $E_T$ from the population, and vanishes in
the large-population limit. Correlation *across time* is what would bias the
generator (the increments must keep zero conditional mean given the past); a
gauge that is constant over the block and recomputed only from the current state
introduces none. $\square$

Consequences for validation (the ladder of §13): **extrapolated energies must
not move**; only error bars and $\tau_{\mathrm{int}}$ may. Practical guards
(all variance-only, never bias): an absolute clamp $q\in[q_{\min},q_{\max}]$,
optional geometric smoothing $q\leftarrow q^{1-\alpha}(q^\star)^\alpha$ (a trust
region on $\lvert\log q\rvert$), and the existing per-walker $\lambda$ cap
`split_gauge_max_norm`. The OU sanity check
$2\omega_\mu\langle\lvert f_\mu\rvert^2\rangle/(q\,)\to1$ should hold once $q$ has
equilibrated.

---

## 5. Consistency with the empirical small-$\omega_0/g$ finding

The diagonal recipe (Cor. 11.7) reads
$q^\star_{\mu}\propto\omega_\mu\,(\lVert\mathrm{kick}_\mu\rVert/\Delta)/\lvert s_\mu\rvert$;
its scalar trace version is $q^\star\propto\omega_0\,(\lVert u\rVert/\Delta)/\lVert s\rVert$.
In the adiabatic corner $\omega_0/t\ll1$ — the classically hard regime — the
explicit factor $\omega_0$ drives $q^\star$ **small**: the optimum starves the
slow phonons and routes noise through the fast-relaxing electron. This is exactly
the empirical observation that small $q$ stabilises long-time propagation at
small $\omega_0/g$, and that $q=1$ fails first there. The optimiser should
therefore recover, automatically and quantitatively, the hand-tuned small-$q$
choice — and predict the gain $(1+r)/2\sqrt r$ *a priori* from a short $q=1$ run.

---

## 6. Algorithm (per refresh)

1. Compute $A_\mu,B_\mu$ from `trial.calc_ito_log_derivatives` (full, non-residual
   coupling for $\langle G\rangle$).
2. Phonon: $s_\mu=\omega_\mu A_\mu+\langle G_\mu\rangle$ (Holstein:
   $\langle G_\mu\rangle=B_\mu$); accumulate
   $a_w=\sum_\mu\lvert s_{w\mu}\rvert^2/\omega_\mu^2$.
3. Electron: build $\widehat O_w=h_{\mathrm{eff}}(f_w)+\sum_\nu A_{w\nu}G_\nu^\dagger$;
   for one electron $u_{w\rho}=\langle\widehat O G_\rho^\dagger\rangle-\langle\widehat O\rangle B_\rho$
   (else kick-norm proxy). Gap $\Delta_w$ from $\mathrm{Re}\,\mathrm{spec}(h_{\mathrm{eff}}(f_w))$.
   Accumulate $b_w=\lVert u_w\rVert^2/\Delta_w^2$.
4. MPI-reduce $\sum_w a_w$, $\sum_w b_w$; set $q^\star=\sqrt{\sum b/\sum a}$;
   clamp/smooth; update `split_gauge_q` and `split_gauge_sqrt_q`.
