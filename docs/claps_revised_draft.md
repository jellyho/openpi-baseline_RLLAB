# CLaPS: Conditioning-Latent Policy Steering for Stable Adaptation of Multi-Step Flow Policies

*Working note / paper draft. Extends Latent Policy Steering (LPS) from one-step MeanFlow policies to expressive multi-step Flow Matching policies by redesigning the steering interface.*

---

## 0. Executive Summary

Latent Policy Steering (LPS) is attractive because it freezes a pretrained generative behavior policy and optimizes only a latent actor. This avoids directly fine-tuning the base policy and reduces destructive policy drift. However, extending LPS from one-step MeanFlow policies to multi-step Flow Matching policies is not straightforward.

A direct extension that steers only the high-dimensional initial flow noise is weak for two reasons:

1. **Signal attenuation.** The critic gradient must propagate through the entire multi-step sampling trajectory.
2. **High-dimensional exploitation.** Even if the latent remains on a Gaussian typical sphere, a critic-guided actor may discover rare decoder directions that random prior sampling almost never visits.

We propose **CLaPS**, short for **Conditioning-Latent Policy Steering**. Instead of steering the full initial noise, CLaPS introduces a compact low-dimensional conditioning latent \(z_c\) that is injected into every flow integration step. The base flow policy is trained once with this conditioning interface and then frozen. During downstream RL, only a lightweight latent actor that outputs \(z_c\) is optimized.

The design aims to provide:

- a shorter and repeatedly injected gradient path than initial-noise steering,
- a compact steering interface that is easier to cover and regularize,
- preservation of multi-step flow expressivity,
- frozen-base downstream adaptation without direct velocity-field fine-tuning.

The core claim is **not** that low dimensionality automatically guarantees perfect support preservation. Rather, low dimensionality makes the reachable steering set substantially easier to cover, validate, and constrain, reducing critic exploitation while retaining useful downstream controllability.

**Contributions.**

1. We identify *why* one-step LPS does not extend to multi-step flows: initial-noise steering suffers both gradient attenuation through the sampling chain and high-dimensional decoder exploitation.
2. We propose a **steerable conditional flow architecture** built on a deliberate **role separation** — high-dimensional detail noise \(\epsilon\) vs. a compact, repeatedly-injected steering latent \(z_c\) — turning RL adaptation into *mode selection over a learned compact interface*, with the base flow frozen.
3. We give a **restricted-set robustness** analysis: a value regret that is *linear* in the in-support critic error (Prop. 1) and an *unconditional* deviation bound \(\mathrm{diam}(\mathcal{M}_s)\) (Prop. 2), together with an honest reachability decomposition that separates the structural from the empirical.
4. We predict and test the property that makes LPS viable for expressive policies: **conditioning-steering does not degrade as NFE grows, while initial-noise steering does**.

---

## 1. Motivation

Pretrained Flow Matching policies model rich multimodal behavior and are attractive priors for offline and offline-to-online RL. However, fine-tuning these models is challenging because their output action is produced implicitly by a multi-step ODE solver.

Let a pretrained flow policy be

\[
\frac{d x_\tau}{d\tau}
=
v_\theta(x_\tau,\tau;s),
\qquad
x_0 \sim \mathcal{N}(0,I),
\qquad
x_1 = a.
\]

Direct policy optimization requires differentiating through the entire sampling trajectory. Existing methods address this in different ways:

| Family | Mechanism | Main limitation |
|---|---|---|
| Action residual methods | Add an action-space correction on top of a frozen policy | Ignores the internal generative trajectory |
| Noise-space steering methods | Optimize the initial latent noise of a frozen decoder | Weak or ill-conditioned signal for long multi-step flows |
| Adjoint / SOC methods | Fine-tune the velocity field through adjoint-style control | More expressive, but requires careful drift control |
| **CLaPS** | Steer a compact conditioning latent injected throughout the frozen flow | Requires a steerable conditioning interface during base training |

The target setting of CLaPS is a strong pretrained flow policy whose behavior coverage is already broad enough for downstream tasks, but whose adaptation interface should remain conservative and efficient.

---

## 2. Recap: Why LPS Works Well for One-Step Policies

In one-step LPS, a frozen decoder maps a latent variable to an action chunk:

\[
a = G_\beta(s,z).
\]

A latent actor outputs

\[
z_\phi = \pi_\phi(s),
\]

and is trained using

\[
\mathcal{L}_{\mathrm{LPS}}(\phi)
=
-
\mathbb{E}_{s\sim\mathcal{D}}
\left[
Q_\psi(s,G_\beta(s,\pi_\phi(s)))
\right].
\]

Because the decoder is frozen, adaptation does not directly modify the pretrained policy parameters. For a spherical latent parameterization,

\[
z_\phi
=
\sqrt{d}
\frac{l_\phi(s)}
{\|l_\phi(s)\|_2},
\]

the actor is constrained to query the Gaussian typical shell rather than arbitrary high-norm latents.

This is a strong practical bias. However, it is important to state the guarantee carefully:

> Remaining on the latent sphere does not automatically imply that every decoded action is perfectly in-distribution. It restricts the actor to the typical latent set seen during base training, which empirically reduces but does not mathematically eliminate decoder exploitation.

This distinction becomes increasingly important as the latent dimension grows.

---

## 3. Why Direct Multi-Step Noise Steering Is Insufficient

A direct multi-step extension of LPS would optimize the initial flow noise:

\[
a
=
F_\beta(s,z_0),
\]

where \(F_\beta\) is the frozen ODE solution map induced by \(v_\beta\).

The actor objective becomes

\[
\mathcal{L}_{\mathrm{noise\text{-}LPS}}
=
-
Q_\psi(s,F_\beta(s,\pi_\phi(s))).
\]

This is mathematically valid, but can be weak in practice.

### 3.1 Gradient attenuation through the full flow trajectory

Let

\[
J_{z_0}
=
\frac{\partial F_\beta(s,z_0)}{\partial z_0}.
\]

The actor receives

\[
\nabla_{z_0}
Q_\psi(s,F_\beta(s,z_0))
=
J_{z_0}^{\top}
\nabla_a Q_\psi(s,a).
\]

For a multi-step flow, \(J_{z_0}\) is a product of step-wise Jacobians. Consequently, the signal can become small or poorly conditioned as the number of integration steps increases.

### 3.2 High-dimensional noise-space exploitation

For long action chunks or high-DoF control, the initial noise dimension may be very large:

\[
d_{\mathrm{noise}}
=
H \cdot d_a.
\]

Even when \(z_0\) remains on a sphere, the actor may discover rare directions that produce abnormal decoded chunks. Random prior sampling may almost never reach these directions, while gradient-based optimization can target them systematically.

Therefore:

\[
\text{on-sphere}
\not\Rightarrow
\text{empirically safe decode}.
\]

The problem is not that the Gaussian typical shell is meaningless. The problem is that a high-dimensional steering interface is difficult to cover densely and easy for an imperfect critic to exploit.

---

## 4. Core Idea: Conditioning-Latent Policy Steering

CLaPS replaces high-dimensional initial-noise steering with a compact conditioning latent.

The conditional flow policy is

\[
\frac{d x_\tau}{d\tau}
=
v_\theta(x_\tau,\tau;s,z_c),
\]

where

\[
z_c
\in
\mathcal{Z}_c
=
\sqrt{d_c}S^{d_c-1},
\qquad
d_c
\ll
H d_a.
\]

The generated action is

\[
a
=
F_\theta(s,\epsilon,z_c),
\qquad
\epsilon \sim \mathcal{N}(0,I).
\]

Here:

- \(\epsilon\) carries high-dimensional stochastic detail,
- \(z_c\) exposes a compact, steerable behavioral mode interface,
- the conditional flow model remains multi-step and expressive.

During downstream RL, the conditional base policy is frozen and only a latent actor is trained:

\[
z_c
=
\pi_\phi(s).
\]

The RL action is

\[
a_\phi
=
F_\theta(s,\epsilon,\pi_\phi(s)).
\]

The actor objective is

\[
\mathcal{L}_{\mathrm{actor}}
=
-
\mathbb{E}
\left[
Q_\psi(s,a_\phi)
\right].
\]

---

## 5. Why Conditioning Helps

### 5.1 Repeated injection mitigates gradient attenuation

Let

\[
S_\tau
=
\frac{\partial x_\tau}{\partial z_c}.
\]

Differentiating the ODE gives

\[
\frac{dS_\tau}{d\tau}
=
J_\tau S_\tau
+
B_\tau,
\]

where

\[
J_\tau
=
\frac{\partial v_\theta}{\partial x_\tau},
\qquad
B_\tau
=
\frac{\partial v_\theta}{\partial z_c}.
\]

The solution has the form

\[
\frac{\partial a}{\partial z_c}
=
\int_0^1
\Phi(1,\tau) B_\tau d\tau.
\]

By contrast, initial-noise steering receives only

\[
\frac{\partial a}{\partial z_0}
=
\Phi(1,0).
\]

The conditioning latent therefore receives a forcing contribution at every integration step, including late steps whose propagation path is short.

The correct claim is:

> Repeated conditioning injection provides shorter gradient routes and can mitigate attenuation relative to initial-noise steering.

The stronger statement that the gradient is always non-vanishing would require additional assumptions, such as a lower bound on \(\|B_\tau\|\) and limited cancellation across timesteps.

### 5.2 Compactness makes coverage easier

The low-dimensional sphere is not automatically safe. However, it is much easier to cover and validate than a high-dimensional noise space.

The goal is to learn a decoder such that

\[
g_s :
\mathcal{Z}_c
\rightarrow
\mathcal{A},
\qquad
g_s(z_c)
=
F_\theta(s,\epsilon,z_c),
\]

induces a compact reachable set

\[
\mathcal{M}_s
=
g_s(\mathcal{Z}_c)
\]

that remains close to demonstrated behavior.

Low dimension helps because:

1. the conditioning space is easier to cover during base training,
2. interpolation between nearby \(z_c\) values is easier to inspect,
3. actor optimization has fewer adversarial directions,
4. the downstream RL problem becomes mode selection over a compact interface.

The claim should be framed as:

> CLaPS makes support preservation easier to enforce and empirically validate by restricting RL to a compact learned steering interface.

---

## 6. Phase 1: Train a Steerable Conditional Flow Policy

The base model must expose a useful conditioning latent before RL begins.

### 6.1 Conditional Flow Matching objective

Let an encoder produce a posterior

\[
q_\varphi(z_c|a,s).
\]

Sample:

\[
z_c
\sim
q_\varphi(\cdot|a,s),
\]

\[
\epsilon
\sim
\mathcal{N}(0,I),
\]

\[
\tau
\sim
\mathcal{U}[0,1],
\]

and form an OT interpolation

\[
x_\tau
=
(1-\tau)\epsilon + \tau a.
\]

The velocity target is

\[
u_\tau
=
a-\epsilon.
\]

Train the conditional flow with

\[
\mathcal{L}_{\mathrm{CFM}}
=
\mathbb{E}
\left[
\left\|
v_\theta(x_\tau,\tau;s,z_c)
-
(a-\epsilon)
\right\|_2^2
\right].
\]

### 6.2 Latent coverage objective

To encourage broad usage of the conditioning sphere:

\[
\mathcal{L}_{\mathrm{KL}}
=
D_{\mathrm{KL}}
\left(
q_\varphi(z_c|a,s)
\|
\mathrm{Unif}
\left(
\sqrt{d_c}S^{d_c-1}
\right)
\right).
\]

The Phase-1 objective is

\[
\mathcal{L}_{\mathrm{base}}
=
\mathcal{L}_{\mathrm{CFM}}
+
\beta
\mathcal{L}_{\mathrm{KL}}.
\]

The coefficient \(\beta\) is a base-training hyperparameter, not an RL steering temperature.

### 6.3 Posterior collapse risk

Because the decoder also observes \(s\), \(\epsilon\), and \(x_\tau\), it may ignore \(z_c\). This is the central implementation risk.

Monitor:

\[
\left\|
\frac{\partial a}{\partial z_c}
\right\|,
\]

and the decode diversity:

\[
\mathbb{E}_{z_c,z_c'}
\left[
\left\|
g_s(z_c)-g_s(z_c')
\right\|
\right].
\]

Practical mitigations:

- use a small \(d_c\),
- use KL warmup or free bits,
- inject \(z_c\) into every action-expert layer,
- verify that changing \(z_c\) changes meaningful action modes,
- compare fixed-noise and random-noise evaluations.

---

## 7. Phase 2: Freeze the Base and Steer Only the Conditioning Latent

After Phase 1, freeze:

- the conditional velocity field \(v_\theta\),
- the encoder \(q_\varphi\),
- the \(z_c\) injection layers.

Train:

- an action-space critic \(Q_\psi\),
- a lightweight latent actor \(\pi_\phi\).

The latent actor outputs

\[
\tilde z_c
=
l_\phi(s),
\]

then projects to the conditioning sphere:

\[
z_c
=
\sqrt{d_c}
\frac{\tilde z_c}
{\|\tilde z_c\|_2}.
\]

The actor objective is

\[
\mathcal{L}_{\mathrm{actor}}
=
-
\mathbb{E}
\left[
Q_\psi
\left(
s,
F_\theta
\left(
s,\epsilon,\pi_\phi(s)
\right)
\right)
\right].
\]

The critic can use a standard chunked TD objective:

\[
\mathcal{L}_{Q}
=
\mathbb{E}
\left[
\left(
Q_\psi(s_t,a_{t:t+H})
-
y_t
\right)^2
\right],
\]

with

\[
y_t
=
r_{t:t+H}
+
\gamma^H
Q_{\bar\psi}
\left(
s_{t+H},
F_\theta
\left(
s_{t+H},
\epsilon',
\pi_{\bar\phi}(s_{t+H})
\right)
\right).
\]

Recommended stabilizers:

- target critic,
- EMA target latent actor,
- critic ensemble minimum,
- actor update delay,
- actor-gradient clipping,
- Cal-QL-style anchoring if needed.

---

## 8. Deterministic vs Stochastic Detail Noise

CLaPS separates:

\[
z_c
\quad
\text{from}
\quad
\epsilon.
\]

This **role separation** — \(\epsilon\) carrying within-mode stochastic detail, \(z_c\) carrying the steerable behavioral mode — is a core design contribution and not merely an implementation detail. It is what allows a *compact* latent to hold the value-relevant degree of freedom while the multi-step flow retains full generative detail; it is also what keeps the RL-facing interface low-dimensional regardless of how high-dimensional the action chunk is. The separation creates two evaluation modes.

### 8.1 Fixed detail noise

\[
\epsilon
=
\epsilon_{\mathrm{fixed}}.
\]

Pros:

- deterministic decoding,
- easier critic optimization,
- easier debugging.

Cons:

- may reduce base diversity.

### 8.2 Random detail noise

\[
\epsilon
\sim
\mathcal{N}(0,I).
\]

Pros:

- preserves within-mode detail diversity,
- closer to the pretrained generative model.

Cons:

- introduces variance into the actor objective.

Recommended protocol:

1. start with fixed \(\epsilon\) for debugging,
2. evaluate both fixed and random \(\epsilon\),
3. optionally train with a small set of fixed detail-noise anchors.

---

## 9. Theory: Restricted-Set Robustness

The clean theoretical story is not that low dimension automatically proves support preservation. The clean story is that CLaPS restricts optimization to a compact learned reachable set.

Let

\[
g_s :
\mathcal{Z}_c
\rightarrow
\mathcal{A}
\]

be the frozen conditional decoder for a fixed detail noise anchor, and define

\[
\mathcal{M}_s
=
g_s(\mathcal{Z}_c).
\]

Assume:

1. \(\mathcal{Z}_c\) is compact.
2. \(g_s\) is continuous.
3. The critic error is bounded on \(\mathcal{M}_s\):

\[
\epsilon_{\mathrm{in}}(s)
=
\sup_{a\in\mathcal{M}_s}
\left|
\hat Q(s,a)-Q^\star(s,a)
\right|.
\]

4. The actor finds a \(\zeta\)-approximate maximizer over \(\mathcal{M}_s\):

\[
\hat Q(s,\hat a)
\ge
\sup_{a\in\mathcal{M}_s}
\hat Q(s,a)
-
\zeta.
\]

Define

\[
a^\star_{\mathcal{M}}
=
\arg\max_{a\in\mathcal{M}_s}
Q^\star(s,a).
\]

Then:

### Proposition 1: Restricted-set critic robustness

\[
Q^\star(s,a^\star_{\mathcal{M}})
-
Q^\star(s,\hat a)
\le
2\epsilon_{\mathrm{in}}(s)
+
\zeta.
\]

### Reachability decomposition

\[
\boxed{
\mathrm{Regret}(s)
\le
\Delta_{\mathrm{reach}}(s)
+
2\epsilon_{\mathrm{in}}(s)
+
\zeta
}
\]

where

\[
\Delta_{\mathrm{reach}}(s)
=
\max_{a\in\mathcal{A}}
Q^\star(s,a)
-
\max_{a\in\mathcal{M}_s}
Q^\star(s,a).
\]

This is the honest trade-off:

- CLaPS pays a reachability gap,
- but restricts critic-guided optimization to a compact learned interface.

### Proposition 2: Unconditional bounded deviation

For **any** critic \(\hat Q\) (even adversarial), since \(\hat a \in \mathcal{M}_s\) and \(\mathcal{M}_s\) is compact,

\[
\| \hat a - a_{\mathrm{base}} \|
\le
\mathrm{diam}(\mathcal{M}_s),
\qquad
\forall\, a_{\mathrm{base}} \in \mathcal{M}_s.
\]

The bound is *structural*: it holds with no trust-region controller and no temperature. What low dimensionality does **not** by itself guarantee is that \(\mathrm{diam}(\mathcal{M}_s)\) is small; rather, a small \(d_c\) plus adequate Phase-1 coverage makes a small reachable diameter *easy to enforce and to validate empirically*. Thus the deviation is always bounded; its tightness is the empirical content.

### Remark: error exposure vs. exponentially-tilted methods

Methods whose optimum is an exponential tilt \(\pi \propto \pi_{\mathrm{base}}\, e^{\beta Q}\) (e.g. QAM/TRQAM) amplify critic error as

\[
\mathrm{TV}(\pi_Q,\pi_{\tilde Q})
\le
\tfrac12\!\left(e^{2\beta\epsilon}-1\right),
\]

where \(\epsilon\) is the critic error over the **full** reachable support, including off-distribution regions. CLaPS instead incurs \(2\epsilon_{\mathrm{in}}\) over a **restricted in-support** set \(\mathcal{M}_s\). This is a difference in *error exposure*, not a universal dominance claim: tilt-based methods can reach high-value actions outside \(\mathcal{M}_s\) that CLaPS cannot. The two are complementary points on a stability–reachability spectrum.

---

## 10. Relation to QAM and TRQAM

QAM and TRQAM fine-tune the sampling dynamics. CLaPS does not.

| Method | Trainable downstream object | Base flow | Main stabilization mechanism |
|---|---|---|---|
| QAM | velocity field | fine-tuned | adjoint matching |
| TRQAM | velocity field + trust-region control | fine-tuned | path-space KL regulation |
| Noise-space LPS | initial noise actor | frozen | latent restriction |
| **CLaPS** | compact conditioning-latent actor | frozen | restricted learned steering interface |

The comparison should be framed as complementary rather than strictly dominant.

*(Distinct lineage, one line: latent-action offline RL — e.g. policy optimization in a learned action latent — targets standard MLP policies, not the multi-step sampling process of a flow; CLaPS instead steers a conditioning latent injected throughout a frozen multi-step flow. Optional to include.)*

QAM/TRQAM ask:

> How can we safely allow the pretrained sampling dynamics to move?

CLaPS asks:

> Can we design a compact interface that makes downstream flow-policy steering effective without moving the pretrained dynamics at all?

TRQAM can exceed the reachable set of a frozen base policy. CLaPS cannot. This is a deliberate trade:

\[
\text{controlled deviation}
\quad
\text{vs}
\quad
\text{restricted reachability}.
\]

Avoid claiming that CLaPS universally dominates QAM/TRQAM. Instead, test whether strong pretrained policies make

\[
\Delta_{\mathrm{reach}}
\approx 0.
\]

---

## 11. Experimental Plan

The experiments should establish four claims:

1. conditioning injection provides a stronger adaptation signal than initial-noise steering,
2. compact \(z_c\) reduces exploitation relative to high-dimensional steering,
3. strong base policies reduce the reachability gap,
4. frozen-base steering can be cheaper and more stable than velocity-field fine-tuning.

### 11.1 Environments

Use three regimes.

#### A. OGBench

Purpose:

- compare against QAM/TRQAM headline results,
- stress-test the weak-base regime,
- measure whether reachability limits matter.

#### B. Robomimic

Purpose:

- reproduce QAM/QAM-E collapse settings,
- measure gradient stability and action drift.

#### C. Strong pretrained VLA manipulation

Purpose:

- test the intended regime,
- evaluate whether broad pretrained behavior makes

\[
\Delta_{\mathrm{reach}}
\approx 0.
\]

### 11.2 Baselines

- BC flow policy
- noise-space LPS / DSRL
- FQL
- IFQL
- QAM
- QAM-E
- TRQAM
- action-residual baseline
- conditional-flow BC without RL
- high-dimensional conditioning-latent steering

### 11.3 Core ablations

#### A1. Initial noise vs conditioning latent

Compare:

\[
z_0 \text{ steering}
\quad
\text{vs}
\quad
z_c \text{ steering}.
\]

#### A2. Low-dimensional vs high-dimensional conditioning latent

Compare:

\[
d_c
\in
\{4,8,16,32,64,128\}.
\]

Hypothesis:

- too small: reachability bottleneck,
- moderate: stable mode selection,
- too large: renewed exploitation risk.

#### A3. NFE sweep

\[
N
\in
\{1,4,10\}.
\]

Hypothesis:

- noise-space LPS degrades as NFE grows,
- conditioning-latent steering degrades less.

#### A4. Fixed vs random detail noise

Compare:

\[
\epsilon_{\mathrm{fixed}}
\quad
\text{vs}
\quad
\epsilon\sim\mathcal{N}(0,I).
\]

#### A5. Conditioning sensitivity

Measure:

\[
\left\|
\frac{\partial a}{\partial z_c}
\right\|,
\]

\[
\mathbb{E}_{z_c,z_c'}
\left[
\|g_s(z_c)-g_s(z_c')\|
\right].
\]

#### A6. Posterior collapse diagnostics

Track:

- KL usage,
- active latent dimensions,
- \(z_c\)-action mutual-information proxy,
- decoder sensitivity to \(z_c\).

### 11.4 Stability metrics

Track:

- success rate,
- return,
- actor gradient norm,
- action norm,
- clipped-action fraction,
- distance to base action samples,
- critic ensemble disagreement,
- Q-value inflation,
- off-manifold proxy,
- update wall-clock,
- peak memory.

---

## 12. Implementation Sketch

### 12.1 Base model modification

Inject \(z_c\) into every action-expert layer.

For an AdaLN / AdaRMS style architecture:

```python
cond = time_emb + state_cond + zc_proj(z_c)
```

Then use `cond` in every flow step.

Zero-initialize `zc_proj` so the modified model starts close to the original base.

### 12.2 Phase-1 encoder

Create a lightweight encoder:

```python
z_c = encoder(action_chunk, observation)
z_c = project_to_sphere(z_c)
```

Train with:

```python
loss = flow_matching_loss + kl_weight * latent_coverage_loss
```

### 12.3 Phase-2 actor

Train:

```python
z_c = latent_actor(observation)
z_c = project_to_sphere(z_c)
action = frozen_multistep_decode(observation, detail_noise, z_c)
actor_loss = -critic(observation, action)
```

### 12.4 Recommended debug order

1. Verify that changing \(z_c\) changes generated action chunks.
2. Plot action chunks for random \(z_c\) samples.
3. Measure \(\|\partial a/\partial z_c\|\).
4. Train latent actor with frozen critic.
5. Train full actor-critic with target critic and EMA target actor.
6. Compare fixed-noise and random-noise variants.

---

## 13. Limitations

1. **Base retraining is required.**  
   CLaPS cannot be attached to an arbitrary frozen noise-only flow policy without adding and training the conditioning interface.

2. **Low dimension is not a formal support guarantee.**  
   It reduces the steering surface and makes coverage easier, but decoder validity over the entire sphere must be empirically validated.

3. **Posterior collapse is a central risk.**  
   If the flow ignores \(z_c\), steering becomes ineffective.

4. **Reachability ceiling remains.**  
   CLaPS cannot improve beyond the frozen conditional decoder's reachable set.

5. **The best latent dimension is task-dependent.**  
   Too small reduces reachability; too large may reintroduce critic exploitation.

6. **Gradient improvement is not unconditional.**  
   Repeated conditioning injection mitigates attenuation but does not guarantee non-vanishing gradients without additional assumptions.

---

## 14. Recommended Paper Framing

> Pretrained flow policies are expressive but difficult to adapt safely with RL. Existing latent steering methods optimize the initial noise of a frozen generator, which becomes ineffective and exploitable in high-dimensional multi-step policies. We propose CLaPS, a steerable flow-policy architecture that exposes a compact conditioning latent injected throughout the sampling trajectory. The conditional flow is trained once and then frozen; downstream RL optimizes only a lightweight latent actor. This design preserves multi-step flow expressivity while providing a compact, repeatedly injected adaptation interface that reduces gradient attenuation and critic exploitation. Rather than allowing controlled drift from the pretrained policy, CLaPS restricts adaptation to a learned reachable set, trading a measurable reachability gap for stable and efficient downstream steering.

---

## 15. Naming Options

### Recommended

\[
\boxed{
\textbf{CLaPS: Conditioning-Latent Policy Steering}
}
\]

Suggested title:

> **CLaPS: Conditioning-Latent Policy Steering for Stable Adaptation of Multi-Step Flow Policies**

### Alternatives

| Name | Expansion | Comment |
|---|---|---|
| **CoLPS** | Conditioning Latent Policy Steering | Most direct LPS extension |
| **LoCoPS** | Low-Dimensional Conditioning Policy Steering | Memorable, highlights compact latent |
| **SLaPS** | Style-Latent Policy Steering | Intuitive but may sound too narrow |
| **CoLaS** | Conditioning-Latent Steering | Clean, but weaker connection to LPS |

---

## 16. Main Takeaway

\[
\boxed{
\text{Do not steer the full generative noise. Steer a compact conditioning interface trained for downstream control.}
}
\]

CLaPS is best viewed not merely as a fine-tuning trick, but as an **RL-steerable flow-policy design**:

\[
\text{high-dimensional detail noise}
+
\text{low-dimensional steering latent}
+
\text{frozen expressive multi-step decoder}.
\]
