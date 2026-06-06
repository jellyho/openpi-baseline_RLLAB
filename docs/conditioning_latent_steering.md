# CoLaS: Conditioning-Latent Steering for Stable Reinforced Fine-Tuning of Multi-NFE Flow Policies

*Working note / paper draft. Extends Latent Policy Steering (LPS) to multi-step flow
policies while structurally guaranteeing on-manifold behavior. Method name **CoLaS**
(Conditioning-Latent Steering); alternatives: LoCoS / LCLS.*

---

## Abstract

Reinforced fine-tuning of pretrained flow policies is unstable: differentiating
the multi-step sampling chain dilutes the learning signal, and critic errors are
amplified into **destructive drift** away from the pretrained prior. Latent Policy
Steering (LPS) elegantly sidesteps drift by steering the policy's **input noise** —
since every prior latent decodes to a valid action, steering *never leaves the
data manifold by construction* and requires **no regularization temperature**. We
show that this guarantee silently breaks in high dimension: when the noise prior is
much higher-dimensional than the action manifold, the sphere contains directions
*orthogonal to the manifold* that decode off-distribution, and gradient-based
steering reliably finds them. We propose **Low-Dimensional Conditioning-Latent
Steering (LCLS)**: instead of steering the high-dimensional initial noise, we steer
a **low-dimensional latent `z_c` injected as conditioning at every integration
step** of a multi-NFE flow. Three properties follow: (i) the conditioning injection
yields a **non-vanishing** steering gradient (no chain dilution); (ii) the low
dimension makes the reachable set a **bounded, in-support manifold**, so steering
cannot leave the prior — structurally, with **no temperature**; (iii) the base
flow stays multi-NFE, preserving expressivity. We prove that LCLS enjoys **linear**
critic-error sensitivity and an **unconditional** deviation bound, in contrast to
the **exponential** amplification of exponentially-tilted methods (QAM/TRQAM,
their Lemma 1). The trade is a base-quality-dependent *reachability* gap, which
vanishes for strong pretrained policies.

---

## 1. Introduction

Pretrained flow / flow-matching policies model rich, multimodal behavior and are an
attractive prior for off-policy RL fine-tuning. The central difficulty is that a
flow policy is defined *implicitly* through a multi-step denoising ODE, so
gradient-based improvement must differentiate the sampling chain — expensive and
unstable — and any critic error tends to push the fine-tuned policy off the
pretrained prior (*destructive drift*).

**Latent Policy Steering (LPS)** offers a clean answer: freeze the flow and run
actor–critic over its **input noise** `z` (the prior latent on a hypersphere).
Because the decoder was trained to map the *whole* prior to the data manifold,
every steered `z` decodes to an on-manifold action — so steering stays in support
**by construction**, with **no KL/temperature** to tune. This is exactly the
property practitioners want, and LPS works well in low-dimensional control (e.g.
OGBench).

**This note's starting observation.** In higher-dimensional settings (long action
chunks, high-DoF robots), the same noise-space LPS *exploits the critic*: the
steered action chunk explodes — typically the last and some middle chunk steps —
in a way **random sampling never produces**, even though the latent stays on the
sphere. We trace this to a precise failure of the LPS premise in high dimension,
and propose a structural fix that restores the premise for **multi-NFE** flows.

---

## 2. Background and the LPS premise

**Flow policy.** A velocity field `v_θ(x, τ; s)` transports noise `x_0 ∼ N(0,I)` to
an action `x_1 ∼ π_base(·|s)` via `dx_τ = v_θ(x_τ, τ; s) dτ`. LPS steers by choosing
the initial `x_0` (projected to the sphere of radius `√d`, the Gaussian typical
set), keeping `v_θ` frozen.

**The LPS premise (why it needs no temperature).**

> *Every point of the prior decodes to an on-manifold action.* Hence steering in the
> prior latent space cannot leave the manifold — manifold preservation is
> **structural**, not a penalty.

**Families of flow-policy fine-tuning** (and their limits, following the TRQAM
taxonomy):

| Family | Mechanism | Limit |
|---|---|---|
| Residual (action-space) | frozen policy + additive action residual | corrects only at action level; ignores generative dynamics |
| **Noise-space (LPS / DSRL)** | actor–critic over input noise | **bounded by the frozen policy's expressivity** |
| Adjoint / SOC (QAM, TRQAM) | learned per-step drift via stochastic optimal control | critic errors amplified (drift); needs VJP/adjoint |

LPS is the noise-space family. Its "expressivity bound" — staying within the base
support — is precisely the *guarantee* we want; the problem we solve is its
high-dimensional *exploitation*.

---

## 3. Diagnosis: why high-dimensional noise-space LPS breaks

Write the LPS premise as a requirement on the decode map `g_s : Z → A`:

> **Premise.** `g_s` maps the prior `Z` onto the data manifold, bijectively and
> with controlled (Lipschitz) conditioning.

In high dimension this fails on two fronts.

**(D1) Dimension excess: on-sphere ≠ on-manifold.** The noise prior is
`dim(Z) = H·d_a` (e.g. `50×32 = 1600`), but smooth action chunks have a much lower
*intrinsic* dimension. The decode therefore has **directions orthogonal to the
manifold**: moving `z` along them moves the action *off* the manifold even though
`‖z‖` is unchanged. Random `z` never excites these measure-zero directions; a
**gradient-based actor finds them deterministically**.

**(D2) Expansive one-step decode.** With a 1-NFE (MeanFlow) decoder `a = z − u(z)`,
the velocity `u` is an unbounded network, so `‖a‖` is **not** bounded by `‖z‖`. A
single jump can be highly expansive in some directions; gradient search drives `z`
there and `a` blows up.

**Why DDPG specifically exploits this.** The critic `Q̂` overestimates off-manifold
actions (extrapolation), so `∇_a Q̂` points *out* of the manifold; the actor follows
`∇_a Q̂ · ∂a/∂z` into exactly the (D1)/(D2) cracks. Random sampling averages over the
sphere and never targets high-`Q̂` directions, which is why **only steering**
explodes. The spikes concentrate at the **chunk boundary** (least-constrained causal
critic token / flow endpoint) and at **horizon seams** — both high-frequency.

The correct objective is the value-tilted prior `π^\*(a|s) ∝ π_base(a|s) e^{Q/α}`,
whose support is `⊆ supp(π_base)` (the tilt **reweights**, it cannot **relocate**
mass); with bounded critic `Q∈[-1,0]` the tilt factor is bounded by 1, so it only
*suppresses*. DDPG/argmax dropped the `α·KL(π‖π_base)` term whose infinite barrier at
the support boundary is what prevents OOD. The question is how to re-impose this
**structurally** (no `α`) while keeping multi-NFE expressivity and a strong
gradient.

---

## 4. Method: Low-Dimensional Conditioning-Latent Steering (LCLS)

**Idea.** Move the steering variable from *the high-dimensional initial condition of
the ODE* to *a low-dimensional parameter of the ODE, re-injected at every step*:

```
v_θ(z_τ, τ, s, z_c),        z_c ∈ low-dim sphere S^{d'-1}(r),  d' ≪ H·d_a
a = ODE_solve(z₁_fixed, v_θ(·,·,s,z_c), num_steps = N).
```

`z_c` modulates the whole trajectory (e.g. via AdaLN / FiLM, or a conditioning token
the action tokens cross-attend to). The high-dimensional noise `z₁` is **fixed** at
steering time (a typical draw), so `a = F_s(z_c)` is a deterministic, expressive map
of the low-dim style latent.

### 4.1 Why conditioning: non-vanishing gradient (no BPTT dilution)

Let `S_τ = ∂z_τ/∂(steer)`. Differentiating the ODE gives the sensitivity ODE
`dS_τ/dτ = J_τ S_τ + B_τ`, `J_τ = ∂v/∂z_τ`, `B_τ = ∂v/∂z_c`, whose solution is

```
da/dz_c = ∫ Φ(0,τ) B_τ dτ          (forcing injected at EVERY τ)
da/dz₁  = Φ(0,1)                     (LPS: fully-propagated initial condition, B≡0)
```

The conditioning case receives a **forcing term `B_τ = ∂v/∂z_c` at every step**,
including the un-attenuated late steps (`Φ(0,τ)→I` as `τ→0`). The initial-noise case
only gets the full transition `Φ(0,1)`, which accumulates all the attenuation — this
is the empirically observed *halving of performance under multi-step BPTT*. The
conditioning latent is a **skip connection to every step**: like text/class
conditioning in diffusion, its gradient does not vanish, while the input noise's
does. (ACT's CVAE "style" latent is exactly such an injected conditioning variable.)

### 4.2 Why low-dim: structural manifold preservation, decoupled from expressivity

Matching `d'` to the action manifold's intrinsic dimension removes the orthogonal
(off-manifold) directions of (D1): every direction of the low-dim sphere is a
direction *along* the data manifold, so steering — even greedy DDPG — **cannot leave
it**. Expressivity is *not* sacrificed: the high-dimensional `z₁` (fixed at steering)
still supplies within-mode detail; `z_c` only carries the value-relevant **mode**.
Thus

```
low-dim z_c (safe steering handle)   +   high-dim noise (expressivity)
```

Crucially, low dimension is required for the **exploitation** fix, not the gradient
fix: a *high-dimensional* conditioning latent would also give a non-vanishing
gradient (4.1) but would reintroduce the (D1) adversarial directions. Both
properties are needed, and they are orthogonal.

### 4.3 Training objectives

**Phase 1 — Conditional Flow-Matching VAE (base; frozen afterwards).** Learn the
conditional velocity `v_θ` and an encoder `q_φ(z_c|a,s)` (a hyperspherical / vMF
posterior on the low-dim sphere):

```
z_c ~ q_φ(·|a,s);   τ~U[0,1];  ε~N(0,I);   z_τ = (1-τ)a + τε
L₁ = ‖ v_θ(z_τ, τ, s, z_c) − (ε − a) ‖²   +   β · KL( q_φ(z_c|a,s) ‖ Uniform(S^{d'-1}) )
```

The encoder forces `z_c` to be informative (carry the action mode); the KL provides
**coverage** of the low-dim sphere so the decoder is valid everywhere the actor may
go. `β` is a one-time base-training knob (not a steering temperature); the low
dimension makes coverage easy and `β` forgiving. *Posterior collapse* — `z_c` ignored
because the noise alone explains the action — is the main risk and is mitigated by
fixing `z₁` at steering, free-bits/annealed KL, and small `d'`.

**Phase 2 — RL steering (base frozen).** Train critic `Q_w` (distributional, with TD
target network, ensemble-min pessimism, CalQL anchor) and a latent actor
`π_ψ(z_c|s)` that **projects to the low-dim sphere**:

```
Critic : L_critic = HL-Gauss( Q_w(s, a_data),  R_chunk + γ^H Q_target(s', Decode(s', π_ψ(s'))) )
Actor  : L_actor  = − Q_w( s, Decode(s, π_ψ(s)) )      (z_c on-sphere; critic detached)
```

No KL / temperature appears in Phase 2: the support constraint is the **sphere
projection** of `π_ψ`'s output. The actor gradient flows through the multi-NFE
decode but is non-vanishing (4.1); memory is controlled by `jax.checkpoint`
(rematerialized scan) or a continuous adjoint — and is *cheaper than fine-tuning the
flow*, because the flow is frozen and only a small actor and a low-dim gradient are
involved.

### 4.4 Figures

**Figure 1 — Why the conditioning gradient survives multi-NFE (sensitivity ODE).**

```
 LPS  (steer the initial noise z₁):
       z₁ ─► z_{τ} ─► z_{τ-h} ─► … ─► a ──► Q
       └────────── da/dz₁ = Φ(0,1)  (one fully-propagated chain) ──────────┘
                          ▲  product of N step-Jacobians ⇒ attenuates / vanishes

 CoLaS (steer conditioning z_c, re-injected at EVERY step):
              z_c        z_c        z_c        z_c
               │∂v/∂z_c   │          │          │          (skip-connection to every τ)
               ▼          ▼          ▼          ▼
       z₁ ─► z_{1} ─► z_{1-h} ─► … ─────────► a ──► Q
               └ da/dz_c = ∫ Φ(0,τ)·∂v/∂z_c dτ  (forcing at every τ) ┘
                          ▲ late terms τ→0 have Φ(0,τ)≈I  ⇒ NON-vanishing
```

**Figure 2 — Suboptimality decomposition: reachability vs stability.**

```
  Q*   ▲
       │   ●  global optimum                ┐
       │   ┊                                │ Δ_reach  (CoLaS cannot cross; TRQAM can,
       │   ┊                                │           at exponential critic-error risk)
       │   ●  max over M_s   ← CoLaS ceiling ┘
       │   ┊  } 2ε_in + ζ    (CoLaS regret: LINEAR, guaranteed, machinery-free)
       │   ●  CoLaS achieved
       │
       │   ●  base policy π_base
       └────────────────────────────────────────────────────────────►
            strong base ⇒ Δ_reach → 0  ⇒  CoLaS ≈ global optimum, for free
```

**Figure 3 — Method overview.**

```
            ┌──────────────── Phase 1 (CFM-VAE, then frozen) ────────────────┐
   (a,s) ─► encoder q_φ ─► z_c ∈ S^{d'-1}(r) ─┐
                                              ├─► v_θ(z_τ,τ,s,z_c)  ─► FM loss + βKL
   noise z_τ, τ ───────────────────────────--┘        (z_c via AdaLN at every step)

            ┌──────────────── Phase 2 (steer, base frozen) ─────────────────┐
   s ─► actor π_ψ ─► z_c (on-sphere) ─► Decode_N(s,z_c) ─► a ─► critic Q_w
                         ▲ DDPG: maximize Q_w(s,a);  z_c low-dim ⇒ on-manifold
```

---

## 5. Theory: structural stability

Let `g_s : Z → A` be the frozen decode of the low-dim sphere `Z = S^{d'-1}(r)`,
reachable set `M_s := g_s(Z)`, critic `Q̂ ≈ Q^\*`.

**Assumptions.**
- **(A1) Compactness.** `Z` compact, `g_s` continuous ⟹ `M_s` compact, `diam(M_s)=:D_s<∞`.
- **(A2) Lipschitz decode.** `‖g_s(z)−g_s(z')‖ ≤ L_g‖z−z'‖` (well-conditioned multi-NFE flow), hence `D_s ≤ 2 r L_g`.
- **(A3) Domain-restricted critic error.** `ε_in(s) := sup_{a∈M_s} |Q̂(s,a) − Q^\*(s,a)|`. Since `M_s ⊆ supp(π_base)`, this is the **in-distribution** error (small).
- **(A4) ζ-approximate maximization.** `Q̂(s, â) ≥ sup_{a∈M_s} Q̂(s,a) − ζ` (`ζ` = optimization gap of DDPG).

**Theorem 1 (Stability of low-dim in-support steering).** Under (A1)–(A4), with
`a^\* := argmax_{a∈M_s} Q^\*(s,a)`:

```
(i)  value regret:   Q^*(s,a^*) − Q^*(s,â) ≤ 2 ε_in(s) + ζ          (linear in ε_in)
(ii) deviation:      ‖â − a_base‖ ≤ D_s ≤ 2 r L_g      for ANY critic Q̂.
```

*Proof.* (i) `Q^*(â) ≥ Q̂(â) − ε_in ≥ sup_M Q̂ − ζ − ε_in ≥ Q̂(a^*) − ζ − ε_in ≥
Q^*(a^*) − 2ε_in − ζ`. (ii) `â ∈ M_s` by construction; `M_s` compact ⟹
`‖â − a‖ ≤ D_s` for all `a ∈ M_s`; `D_s ≤ L_g·diam(Z) = 2rL_g` by (A2). ∎

**Lemma 2 (Comparison with the tilt family, TRQAM Lemma 1).** For the
exponentially-tilted family `π_Q ∝ π_base e^{βQ}` (the optimizer of the SOC / KL
objective, hence QAM/TRQAM),

```
TV(π_Q, π_{Q̃}) ≤ ½ ( e^{2βε} − 1 ),     ε = ‖Q − Q̃‖_{∞, full support}.
```

**Contrast (two independent axes).**

| | amplification | operating error | deviation bound |
|---|---|---|---|
| Tilt (QAM/TRQAM) | `e^{2βε}` (exponential) | `ε` over **full** support (incl. OOD ⟹ large) | requires active control of `β`/`λ` (conditional) |
| **LCLS (ours)** | `2 ε_in` (**linear**) | `ε_in` **in-distribution** (small) | `D_s`, **unconditional** in `Q̂` |

LCLS wins on *both* the amplification and the operating-error axes, and its deviation
bound holds for an **arbitrary, even adversarial, critic** — no Girsanov, dual
descent, or adjoint required.

**Suboptimality decomposition (the honest trade).**

```
total suboptimality(s) = Δ_reach(s)                  + ( 2 ε_in(s) + ζ )
                         └ reachability gap ┘           └ stability term ┘
   Δ_reach(s) := max_{a∈A} Q^*(s,a) − max_{a∈M_s} Q^*(s,a)  ≥ 0.
```

- **LCLS:** pays a structural `Δ_reach` (cannot exceed `M_s`), but the stability term
  is **small and linear**, guaranteed.
- **Tilt:** `Δ_reach = 0` (can relocate mass anywhere), but the stability term is the
  **exponential** `e^{2βε}` exposure that TRQAM spends its entire machinery to bound.

For a **strong** pretrained policy (`M_s` covers the good modes), `Δ_reach ≈ 0`: LCLS
attains near-global optimality *and* the structural stability for free. For a weak /
narrow base, `Δ_reach > 0` — this is the regime where controlled-deviation methods
(TRQAM) pay off.

---

## 6. Computation: BPTT vs adjoint

Adjoint matching (TRQAM) **is** backpropagation through time — in its
memory-efficient continuous form: the backward adjoint ODE is reverse-mode AD,
costing one **VJP per step** (TRQAM's stated limitation). Per update both methods are
`O(N)` flow evaluations.

The decisive difference is *what is updated*:

| | flow | per-update gradient | memory |
|---|---|---|---|
| TRQAM | **fine-tuned** (whole velocity net) | adjoint VJP for **all** flow params + optimizer state | `O(N)` states (lean adjoint) |
| **LCLS** | **frozen** | input-gradient to a **low-dim `z_c`** + small actor | `O(N)` activations naively → `O(N)` small carries with `jax.checkpoint` |

LCLS carries no full-model parameter gradient or optimizer state for the flow.
Gradient checkpointing (rematerialized scan) or a continuous adjoint give LCLS the
same lean-memory regime as TRQAM, after which LCLS is **strictly cheaper per update**
(frozen flow, small actor, low-dim gradient). Stability of the chain gradient is
obtained by TRQAM via the adjoint-matching *regression* reformulation, and by LCLS via
the *non-vanishing conditioning gradient* plus the *low dimension*.

---

## 7. Relation to prior work

- **LPS / DSRL (noise-space).** LCLS is their structural fix: low-dim **conditioning**
  instead of high-dim **noise** removes both the exploitation (D1) and the BPTT
  dilution (4.1), while retaining the "stay in support" guarantee they are praised
  (and, in TRQAM, critiqued) for.
- **Residual / action-space (LPSD, TD3+BC).** Correct only at the action level; LCLS
  steers through the generative dynamics and needs no behavior-cloning temperature.
- **QAM / TRQAM (SOC / adjoint).** Same diagnosis (drift = our exploitation; their
  Lemma 1 is our exponential-amplification statement; the tilted prior is the shared
  target). Opposite cure: they **bound a controlled deviation** with adaptive `λ`;
  we **forbid deviation structurally** with a low-dim support. We avoid VJP/adjoint
  and any temperature, at the cost of reachability.
- **PLAS / BCQ (latent-action offline RL).** Empirical evidence that *DDPG in a
  low-dim latent* works where high-dim does not — the same algorithm, the right space.
- **EWFM / FlowQ / QGPO (energy-weighted FM, learned guidance).** Train a value-tilted
  flow via per-sample weighting (no VJP) — strong but they *re-train* the flow and
  abandon frozen-base steering; LCLS keeps the frozen base + cheap latent steer.
- **ACT (CVAE action chunking).** Validates the architecture: a low-dim conditioning
  "style" latent injected into a chunk generator; LCLS adds critic-driven steering of
  that latent.

---

## 8. Experimental design (proposed)

The experiments must establish two things: **(S) structural stability** (CoLaS does
not drift / collapse where tilt methods do), and **(R) the reachability claim**
(`Δ_reach → 0` for strong bases, so the stability is *free*, not bought with
performance). We deliberately span the **weak-base** regime (where TRQAM wins) and
the **strong-base** regime (our target).

**Environments.**
- **OGBench** (50 tasks) — direct comparison to TRQAM's headline numbers; *weak/narrow*
  bases, large `Δ_reach` expected (adversarial to our claim — report honestly).
- **Robomimic (lift, can)** — the setting where TRQAM shows QAM/QAM-E **collapse**
  (their Fig. 2); used for the stability experiment (E1).
- **Robot manipulation with a large pretrained flow VLA (e.g. π0 / π0.5 tabletop)** —
  the *strong-base* regime where `Δ_reach ≈ 0` is plausible; the regime CoLaS targets.

**Baselines.**
- Noise-space: **DSRL** (= LPS) — the method we fix.
- Flow-RL: **FQL**, **IFQL**, **CGQL-L**.
- Adjoint/SOC: **QAM**, **QAM-E**, **TRQAM** (SOTA).
- Energy-weighted (re-trains flow): **FlowQ / EWFM** (no-VJP point of comparison).

**Ablations (isolate each claim).**
- **A1 low-dim vs high-dim `z_c`** — high-dim conditioning keeps the non-vanishing
  gradient but should re-introduce exploitation ⇒ isolates that *low-dim* (not
  conditioning) prevents OOD.
- **A2 conditioning vs initial-noise steering** at multi-NFE — should reproduce the
  observed *performance halving* of noise steering ⇒ isolates the BPTT-dilution fix.
- **A3 NFE sweep** `N ∈ {1,4,10}` — 1-NFE CoLaS vs multi-NFE; expressivity vs cost.
- **A4 `d'` sweep** — diversity (too small) vs adversarial surface (too large).
- **A5 critic stabilizers** — ensemble-min / target / CalQL on–off.

**Metrics.**
- Task **success rate / return** (primary).
- **Stability**: deviation `‖a_steered − a_base‖` vs critic-error proxy; **adjoint/grad
  norm** over training (replicate TRQAM Fig. 2: show CoLaS stays flat where QAM diverges
  to 10²⁰); **off-manifold fraction** measured by the FK/projection overlay
  (`examples/tabletop_sim`, `--action-overlay`): fraction of steered chunks leaving the
  base sample cloud.
- **`Δ_reach` estimate**: `max_{a∈M_s} Q*(s,a)` (best of N base samples re-scored, IDQL
  oracle) minus the achieved value; track as base strength varies (E2).
- **Compute**: per-update wall-clock and **peak memory** vs TRQAM (frozen-flow +
  checkpoint vs full-flow adjoint).

**Key experiments.**
- **E1 (stability under imperfect critic).** Robomimic, identical critic/seed sweep as
  TRQAM Fig. 2. *Claim:* CoLaS adjoint/grad norm and success stay stable where
  QAM/QAM-E collapse — **without** a trust-region controller (Theorem 1(ii)).
- **E2 (reachability vs base strength).** Train base policies of increasing
  quality/coverage; plot CoLaS−TRQAM performance gap vs base strength. *Claim:* the
  gap shrinks to ≈0 as the base strengthens (`Δ_reach → 0`), validating the central
  trade decomposition.
- **E3 (no BPTT dilution).** CoLaS vs noise-space LPS as `N` grows. *Claim:* CoLaS
  retains performance at `N=10` where noise steering halves (Fig. 1 / sensitivity ODE).
- **E4 (efficiency).** Per-update cost / peak memory vs TRQAM. *Claim:* frozen flow +
  low-dim `z_c` + checkpointing ⇒ strictly cheaper per update at equal NFE.

**Falsifiable predictions.** (i) On strong bases CoLaS matches TRQAM at lower cost and
with zero collapses; (ii) on weak bases (OGBench) CoLaS trails TRQAM by exactly the
measured `Δ_reach`; (iii) removing *low-dim* (A1) restores exploding actions while
removing *conditioning* (A2) restores BPTT dilution — the two failures are separable.

---

## 9. Limitations and open questions

1. **Reachability ceiling.** LCLS is capped at `max_{a∈M_s} Q^\*`. The central
   empirical claim — that a strong pretrained `M_s` is rich enough that `Δ_reach ≈ 0`
   — must be demonstrated; this is the question a reviewer will press.
2. **Posterior collapse.** If `z_c` is ignored, there is nothing to steer; needs the
   `z₁`-fixing, KL coverage, and small `d'` mitigations, and monitoring of the
   `z_c`↔action mutual information.
3. **Base retraining.** The conditioning latent must be part of the base (Phase 1) —
   a frozen noise-only policy cannot accept it post-hoc; an adapter fine-tune is a
   lighter alternative.
4. **Choice of `d'`.** Set near the intrinsic dimension of the *steerable mode*
   variation; too small caps diversity, too large reintroduces adversarial
   directions.

---

## 10. Conclusion

High-dimensional noise-space LPS breaks not because steering leaves the sphere, but
because the sphere is **larger than the action manifold** and a gradient actor finds
the orthogonal, off-manifold directions. The fix is not a temperature or a trust
region, but a **change of steering space**: a **low-dimensional latent injected as
conditioning** into a multi-NFE flow. This restores the original LPS guarantee in
high dimension — *structurally, with no temperature* — while keeping a strong,
non-vanishing steering gradient and full generative expressivity. The resulting
method has **linear** critic-error sensitivity and an **unconditional** deviation
bound, where exponentially-tilted SOC methods (QAM/TRQAM) have **exponential**
sensitivity controlled only by active machinery. The price — a reachability gap that
vanishes for strong pretrained policies — is exactly the regime of modern large
flow-based robot policies.

---

### Appendix B. Implementation map (openpi / π0-alphaflow)

The conditioning hook already exists: the action expert is modulated by **AdaRMS**
(`adarms_cond`) at *every layer* (`gemma.py: RMSNorm(...)(x, adarms_cond[i])`, lines
~336/351), and `adarms_cond` is rebuilt at every flow step. Injecting `z_c` there
makes it a conditioning forcing term at every layer × every step (Fig. 1) **for free**
— no new attention path.

**Where each piece goes:**

1. **`z_c` injection (1 line of math).** In `pi0_alphaflow.embed_suffix_with_r`
   (`pi0_alphaflow.py:195–214`), the suffix conditioning is
   `adarms_cond + r_emb`. Add a projection of `z_c`:
   ```python
   # new module:  self.zc_proj = nnx.Linear(d', action_expert_width)   # zero-init out
   return tokens, mask, ar_mask, adarms_cond + r_emb + self.zc_proj(z_c)
   ```
   Zero-init `zc_proj` so the conditioned model starts identical to the base.

2. **Thread `z_c` through the velocity call.** `_action_velocity`
   (`pi0_alphaflow.py:293–317`) and the decode/`steer`/`sample` paths in
   `pi0_lps_rft.py` (`_decode`, `steer_actions`, `_loss_*`) take an extra `z_c`
   argument and pass it to `embed_suffix_with_r`. The multi-step integrator (NFE>1)
   wraps its step in `jax.checkpoint` (§6).

3. **Encoder `q_φ(z_c|a,s)`** (Phase-1 only). New small module: tokenize the action
   chunk + reuse the frozen prefix KV (state/image context), pool, project to a
   low-dim **sphere** (vMF / projected-Gaussian; reuse `_to_sphere` with radius
   `√d'`). Lives next to `latent_out_proj` in `pi0_lps_rft.py`.

4. **Phase-1 loss (CFM-VAE).** In the base `compute_loss` (the flow-matching branch of
   `pi0_alphaflow`/`pi0`): sample `z_c ~ q_φ`, condition `v_θ`, add
   `β·KL(q_φ ‖ Uniform-sphere)`. Use **true multi-step FM** (not 1-NFE MeanFlow) so the
   base is expressive. `β` is annealed; `d'` small. New config: `z_c_dim`, `kl_weight`,
   `nfe`.

5. **Phase-2 (steer).** Reuse the existing LPS-RFT critic + latent actor
   (`pi0_lps_rft.py`), but: the latent actor's `latent_out_proj` now outputs a
   **`d'`-dim** vector → `_to_sphere(√d')` → fed as `z_c` (not the per-step noise). The
   noise `z₁` is held fixed (a typical draw). DDPG `−Q_w(s, Decode_N(s, z_c))`, with the
   critic ensemble-min / target / CalQL already implemented. **No KL/temperature in
   Phase 2** — the sphere projection is the only constraint.

6. **Config.** Add a `Pi0CoLaSConfig` (extends `Pi0LPSRFTConfig`) with `z_c_dim`,
   `kl_weight`, `nfe`, reusing `get_freeze_filter` (the encoder is Phase-1-trainable,
   frozen in Phase 2; `zc_proj` is part of the frozen base after Phase 1).

**Status.** Hook points verified against the current code; the change is additive
(zero-init `zc_proj` ⇒ base-identical at init). Needs a GPU smoke test (one Phase-1
step + one Phase-2 step) before any run — untested here by design (training GPUs are
off-limits).

---

### Appendix A. Notation

| Symbol | Meaning |
|---|---|
| `π_base(·|s)` | pretrained (frozen) flow policy |
| `v_θ(z_τ,τ,s,z_c)` | conditional velocity field |
| `z_c ∈ S^{d'-1}(r)` | low-dim steering (style) latent on a sphere |
| `z₁` | high-dim initial noise (fixed at steering) |
| `g_s, M_s = g_s(Z)` | decode map and its (compact, in-support) reachable set |
| `Q^\*, Q̂` | true / learned critic; `ε_in` = in-distribution critic error on `M_s` |
| `Δ_reach` | reachability gap (global optimum − best in `M_s`) |
| `N` | number of flow integration steps (NFE) |
