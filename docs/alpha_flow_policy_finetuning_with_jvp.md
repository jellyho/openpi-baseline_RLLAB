# Flow Matching Policy를 α-Flow로 Fine-tuning하기

이 문서는 기존에 학습된 **Flow Matching policy**를 **α-Flow / MeanFlow-style few-step policy**로 fine-tuning하기 위한 구현 가이드입니다.

목표는 다음과 같습니다.

1. 기존 Flow Matching policy의 action distribution과 pretrained behavior를 최대한 보존한다.
2. policy가 `timestep t`에서 `target timestep r`까지 한 번에 이동하는 mean velocity를 예측하도록 확장한다.
3. α-Flow curriculum을 사용해 Flow Matching objective에서 MeanFlow-style consistency objective로 부드럽게 전환한다.
4. 최종적으로 action generation의 NFE를 줄인다. 예를 들어 10-step policy를 1-step 또는 2-step policy로 fine-tuning한다.

---

## 1. 기본 설정

기존 Flow Matching policy가 다음과 같은 형태라고 가정합니다.

```python
v_theta(obs, z_t, t)
```

여기서

- `obs`: observation, image, proprioception, language instruction 등
- `a`: clean action 또는 action chunk
- `eps`: Gaussian noise
- `t`: noise timestep, 일반적으로 `[0, 1]`
- `z_t`: noisy action
- `v_theta`: Flow Matching velocity prediction

일반적인 straight-line Flow Matching에서는 다음과 같이 noisy action을 만듭니다.

```math
z_t = (1 - t)a + t\epsilon
```

정답 velocity는 다음과 같습니다.

```math
v_t = \epsilon - a
```

기존 Flow Matching policy는 다음 objective로 학습됩니다.

```math
L_{\mathrm{FM}}
=
\mathbb{E}_{a,\epsilon,t}
\left[
\|v_\theta(o, z_t, t) - v_t\|_2^2
\right]
```

---

## 2. α-Flow fine-tuning의 목표

α-Flow fine-tuning에서는 기존 policy를 다음 형태로 확장합니다.

```python
u_theta(obs, z_t, r, t)
```

여기서 `r <= t`입니다.

기존 `v_theta(obs, z_t, t)`는 현재 timestep `t`에서의 local velocity를 예측합니다.

반면 `u_theta(obs, z_t, r, t)`는 timestep `t`에서 `r`까지 이동하는 trajectory-level mean velocity를 예측합니다.

즉, 기존 모델이 다음을 학습했다면

```text
현재 noisy action z_t에서 infinitesimal 또는 local velocity를 예측한다.
```

α-Flow fine-tuning 후에는 다음을 학습합니다.

```text
현재 noisy action z_t에서 target timestep r까지 큰 구간을 jump할 수 있는 mean velocity를 예측한다.
```

이 구조를 사용하면 sampling step 수를 줄일 수 있습니다.

---

## 3. 왜 α-Flow curriculum이 필요한가?

MeanFlow objective는 크게 두 가지 성격의 objective를 동시에 포함합니다.

1. **Trajectory Flow Matching**
   - 정확한 velocity를 맞추는 supervised objective입니다.
   - 해 공간이 비교적 좁고, pretrained Flow Matching policy와 잘 맞습니다.

2. **Trajectory Consistency**
   - trajectory 상에서 model prediction이 자기 자신과 일관되도록 만드는 objective입니다.
   - few-step generation에 중요하지만, 단독으로는 불안정할 수 있습니다.

α-Flow 논문의 분석에 따르면, 이 두 objective의 gradient는 학습 중 서로 강하게 충돌할 수 있습니다. 따라서 처음부터 MeanFlow-style consistency를 강하게 거는 것보다, 먼저 Flow Matching에 가까운 objective로 안정적인 velocity structure를 만든 뒤 점진적으로 consistency objective로 전환하는 것이 안정적입니다.

α-Flow는 이 전환을 하나의 scalar `alpha`로 제어합니다.

```math
s = \alpha r + (1 - \alpha)t
```

- `alpha = 1`: Trajectory Flow Matching에 해당
- `alpha = 1/2`: Shortcut-style consistency에 가까움
- `alpha -> 0`: MeanFlow-style continuous consistency에 가까움

따라서 fine-tuning schedule은 다음처럼 설계합니다.

```text
early:  alpha = 1
middle: alpha decreases from 1 to small value
late:   alpha ≈ 0
```

---

## 4. 모델 구조 변경

### 4.1 기존 timestep embedding

대부분의 Flow Matching policy는 timestep `t`를 embedding해서 network에 condition으로 넣습니다.

예를 들어 기존 구조가 다음과 같을 수 있습니다.

```python
t_emb = timestep_embedding(t)        # [B, D]
h = obs_action_encoder(obs, z_t)     # [B, L, D]
h = h + t_emb[:, None, :]
v = network(h)
```

또는 Transformer block 내부의 AdaLN / FiLM condition으로 사용할 수도 있습니다.

```python
scale, shift = adaLN(t_emb).chunk(2, dim=-1)
h = norm(h) * (1 + scale[:, None, :]) + shift[:, None, :]
```

α-Flow에서는 model이 `r`과 jump length `t-r`도 알아야 하므로 time condition을 확장해야 합니다.

---

## 5. 추천 time conditioning 방식

가장 추천하는 방식은 다음입니다.

```python
time_cond = emb(t) + zero_init_mlp(concat(emb(r), emb(t - r)))
```

즉, 기존 `emb(t)`는 그대로 유지하고, 새로 필요한 `r`와 `t-r` 정보만 residual adapter처럼 추가합니다.

### 5.1 왜 이렇게 하는가?

#### 이유 1. 기존 pretrained policy를 보존하기 위해

기존 policy는 이미 `emb(t)`가 들어오는 조건에 맞춰 학습되어 있습니다.

갑자기

```python
time_cond = MLP(concat(emb(t), emb(r), emb(t-r)))
```

처럼 완전히 새로운 embedding을 넣으면 condition distribution이 바뀌고, fine-tuning 초반에 기존 policy behavior가 깨질 수 있습니다.

반면 다음 구조를 사용하면

```python
time_cond = emb(t) + extra_cond(r, t-r)
```

`extra_cond`를 zero-init했을 때 fine-tuning 시작 시점에는

```python
time_cond ≈ emb(t)
```

가 됩니다.

따라서 모델은 처음에는 기존 Flow Matching policy와 거의 동일하게 동작합니다. 이후 fine-tuning이 진행되면서 `r`와 `t-r` 정보를 점진적으로 사용하게 됩니다.

#### 이유 2. `t-r`는 jump length를 직접 알려준다

`r`만 넣는 것도 가능하지만, 실제로 model이 알아야 하는 중요한 정보는 구간 길이입니다.

```math
\Delta t = t - r
```

예를 들어 다음 두 경우를 비교해봅니다.

```text
t = 0.8, r = 0.7  -> small jump
t = 0.8, r = 0.0  -> large jump
```

현재 noise level `t`는 같지만, 해야 하는 prediction은 매우 다릅니다. 따라서 `t-r`를 명시적으로 넣는 것이 좋습니다.

#### 이유 3. residual conditioning은 adapter처럼 동작한다

이 방식은 LoRA나 adapter를 pretrained model에 붙이는 것과 비슷합니다.

- pretrained path: `emb(t)`
- new path: `zero_init_mlp(concat(emb(r), emb(t-r)))`

처음에는 new path가 0이므로 pretrained function을 유지합니다. 학습이 진행되면서 new path가 점점 의미 있는 `r`-conditioned correction을 학습합니다.

---

## 6. TimeCondition 모듈 예시

아래 코드는 sinusoidal timestep embedding을 이미 가지고 있다고 가정합니다.

```python
import torch
import torch.nn as nn


class AlphaFlowTimeCondition(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.extra = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

        # 중요: final layer zero-init
        nn.init.zeros_(self.extra[-1].weight)
        nn.init.zeros_(self.extra[-1].bias)

    def forward(self, t, r):
        """
        Args:
            t: [B] tensor, current timestep
            r: [B] tensor, target timestep, r <= t

        Returns:
            time_cond: [B, D]
        """
        t_emb = timestep_embedding(t, self.dim)
        r_emb = timestep_embedding(r, self.dim)
        dt_emb = timestep_embedding(t - r, self.dim)

        extra = self.extra(torch.cat([r_emb, dt_emb], dim=-1))
        time_cond = t_emb + extra
        return time_cond
```

주의할 점은 다음입니다.

```python
time_cond = t_emb + 0.0 * extra
```

처럼 구현하면 안 됩니다. 이렇게 하면 `extra` branch로 gradient가 흐르지 않습니다.

반드시 마지막 layer의 weight와 bias를 zero-init해야 합니다.

---

## 7. α-Flow loss

### 7.1 Sampling

training batch에서 clean action `a`, observation `obs`를 가져옵니다.

```python
eps = torch.randn_like(a)
t, r = sample_t_r(batch_size)
```

항상 `0 <= r <= t <= 1`이 되도록 sample합니다.

예시:

```python
t = torch.rand(B, device=device)
r = torch.rand(B, device=device) * t
```

noisy action과 ground-truth velocity는 다음과 같습니다.

```python
z_t = (1 - t) * a + t * eps
v_t = eps - a
```

실제 코드에서는 shape broadcasting을 위해 `t`를 action dimension에 맞게 reshape해야 합니다.

```python
t_b = t.view(B, *([1] * (a.ndim - 1)))
r_b = r.view(B, *([1] * (a.ndim - 1)))

z_t = (1 - t_b) * a + t_b * eps
v_t = eps - a
```

---

### 7.2 Intermediate timestep

α-Flow는 중간 timestep `s`를 둡니다.

```math
s = \alpha r + (1-\alpha) t
```

코드에서는 다음과 같습니다.

```python
s = alpha * r + (1.0 - alpha) * t
```

여기서 `alpha`는 scalar일 수도 있고 batch-wise tensor일 수도 있습니다.

---

### 7.3 Intermediate state

straight-line Flow Matching에서는 `v_t = eps - a`가 constant velocity이므로, `z_t`에서 `s`로 이동한 state는 다음과 같이 만들 수 있습니다.

```math
z_s = z_t - (t-s)v_t
```

코드:

```python
s_b = s.view(B, *([1] * (a.ndim - 1)))
z_s = z_t - (t_b - s_b) * v_t
```

---

### 7.4 α-Flow target

α-Flow target은 다음과 같습니다.

```math
u_{\mathrm{target}}
=
\alpha v_t
+
(1-\alpha)u_{\theta^-}(o, z_s, r, s)
```

여기서 `theta^-`는 stop-gradient target입니다. EMA model을 사용할 수도 있고, 단순히 현재 model의 prediction을 `detach()`해 사용할 수도 있습니다.

기본 구현은 다음과 같습니다.

```python
u_pred = model(obs, z_t, r, t)

with torch.no_grad():
    u_next = model(obs, z_s, r, s)
    alpha_b = alpha.view(B, *([1] * (a.ndim - 1)))
    u_target = alpha_b * v_t + (1.0 - alpha_b) * u_next

loss = ((u_pred - u_target) ** 2).mean()
```

논문 formulation에서는 `alpha^{-1}` weight가 들어갑니다.

```math
L_\alpha
=
\mathbb{E}
\left[
\alpha^{-1}
\|u_\theta(o,z_t,r,t) - \mathrm{sg}(u_{\mathrm{target}})\|_2^2
\right]
```

따라서 좀 더 faithful한 구현은 다음과 같습니다.

```python
error = u_pred - u_target
loss = (error.pow(2).mean(dim=action_dims) / alpha.clamp_min(alpha_min)).mean()
```

실제로는 `alpha`가 너무 작을 때 numerical instability가 생길 수 있으므로 `alpha_min`으로 clamp하는 것이 좋습니다.

---

## 8. α schedule

추천 schedule은 세 단계입니다.

```text
Stage 1: alpha = 1
Stage 2: alpha smoothly decreases from 1 to alpha_min
Stage 3: alpha = alpha_min or alpha = 0 with MeanFlow loss
```

가장 간단한 구현은 linear schedule입니다.

```python
def alpha_schedule(step, start_step, end_step, alpha_min=5e-3):
    if step < start_step:
        return 1.0
    if step > end_step:
        return alpha_min

    progress = (step - start_step) / (end_step - start_step)
    alpha = 1.0 + progress * (alpha_min - 1.0)
    return alpha
```

논문 스타일에 더 가까운 sigmoid schedule은 다음과 같습니다.

```python
import math

def sigmoid_alpha_schedule(step, start_step, end_step, gamma=25.0, alpha_min=5e-3):
    if step <= start_step:
        return 1.0
    if step >= end_step:
        return alpha_min

    scale = 1.0 / (end_step - start_step)
    offset = - (start_step + end_step) / (2.0 * (end_step - start_step))

    x = (scale * step + offset) * gamma
    sig = 1.0 / (1.0 + math.exp(-x))

    alpha = 1.0 - sig
    alpha = max(alpha_min, min(1.0, alpha))
    return alpha
```

추천 초기 실험값:

```text
total_steps: 100%
alpha = 1: first 30~40%
transition: middle 40~50%
alpha = alpha_min: last 10~20%
alpha_min: 5e-3
```

예시:

```text
total fine-tuning steps = 100k
start_step = 30k
end_step = 80k
alpha_min = 5e-3
```

---

## 9. 전체 training step pseudocode

```python
def train_step(batch, model, optimizer, step):
    obs = batch["obs"]
    a = batch["action"]  # clean action or action chunk
    B = a.shape[0]

    device = a.device
    eps = torch.randn_like(a)

    # 1. sample t, r
    t = torch.rand(B, device=device)
    r = torch.rand(B, device=device) * t

    # 2. sample alpha
    alpha_value = sigmoid_alpha_schedule(
        step,
        start_step=30000,
        end_step=80000,
        gamma=25.0,
        alpha_min=5e-3,
    )
    alpha = torch.full_like(t, alpha_value)

    # 3. construct z_t and velocity
    view_shape = (B,) + (1,) * (a.ndim - 1)
    t_b = t.view(view_shape)
    r_b = r.view(view_shape)
    alpha_b = alpha.view(view_shape)

    z_t = (1.0 - t_b) * a + t_b * eps
    v_t = eps - a

    # 4. intermediate timestep
    s = alpha * r + (1.0 - alpha) * t
    s_b = s.view(view_shape)

    # 5. intermediate state
    z_s = z_t - (t_b - s_b) * v_t

    # 6. prediction
    u_pred = model(obs, z_t, r, t)

    # 7. target
    with torch.no_grad():
        u_next = model(obs, z_s, r, s)
        u_target = alpha_b * v_t + (1.0 - alpha_b) * u_next

    # 8. loss
    error = u_pred - u_target

    # If action has shape [B, horizon, action_dim], average over all non-batch dims
    reduce_dims = tuple(range(1, error.ndim))
    per_sample_loss = error.pow(2).mean(dim=reduce_dims)

    loss = (per_sample_loss / alpha.clamp_min(5e-3)).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "alpha": alpha_value,
    }
```

---

## 10. 기존 Flow Matching loss를 섞는 선택지

fine-tuning이 불안정하면 기존 Flow Matching loss를 같이 섞는 것이 좋습니다.

```math
L
=
L_\alpha
+
\lambda_{\mathrm{FM}} L_{\mathrm{FM}}
```

코드:

```python
v_pred = model(obs, z_t, t, t)  # r = t로 넣으면 기존 FM과 유사
fm_loss = ((v_pred - v_t) ** 2).mean()

loss = alpha_flow_loss + lambda_fm * fm_loss
```

추천값:

```text
lambda_fm = 0.05 ~ 0.5
```

초기에는 크게, 후반에는 줄이는 schedule도 가능합니다.

```text
early:  lambda_fm = 0.5
middle: lambda_fm = 0.1
late:   lambda_fm = 0.0 ~ 0.05
```

---

## 11. Sampling 방법

### 11.1 1-NFE sampling

noise action에서 시작합니다.

```python
z = torch.randn(action_shape, device=device)
t = torch.ones(B, device=device)
r = torch.zeros(B, device=device)

u = model(obs, z, r, t)
a_hat = z - u
```

왜 `a_hat = z - u`인가?

straight-line path에서 `t=1`은 noise이고 `r=0`은 clean action입니다. `u`가 `[0,1]` 구간의 mean velocity를 예측한다면, `z_0 ≈ z_1 - (1 - 0)u`입니다.

---

### 11.2 2-NFE sampling

중간 timestep `m`을 둡니다.

```python
z = torch.randn(action_shape, device=device)

t = torch.ones(B, device=device)
m = torch.full((B,), 0.5, device=device)
r = torch.zeros(B, device=device)

u1 = model(obs, z, m, t)
z_m = z - (t.view(view_shape) - m.view(view_shape)) * u1

u2 = model(obs, z_m, r, m)
a_hat = z_m - (m.view(view_shape) - r.view(view_shape)) * u2
```

중간 timestep `m`은 0.3, 0.4, 0.5 등으로 ablation하는 것이 좋습니다.

---

## 12. 구현 체크리스트

### Model

- [ ] 기존 `v_theta(obs, z_t, t)`를 `u_theta(obs, z_t, r, t)`로 확장했는가?
- [ ] 기존 `emb(t)` 경로는 유지했는가?
- [ ] `r`와 `t-r`를 추가 condition으로 넣었는가?
- [ ] 추가 condition branch의 마지막 layer를 zero-init했는가?
- [ ] `time_cond = emb(t) + zero_init_mlp(concat(emb(r), emb(t-r)))` 형태로 시작하는가?

### Loss

- [ ] `0 <= r <= t <= 1`이 보장되는가?
- [ ] `s = alpha * r + (1-alpha) * t`를 올바르게 계산했는가?
- [ ] `z_s = z_t - (t-s) * v_t`를 올바르게 계산했는가?
- [ ] target prediction `u_next`에는 gradient를 흘리지 않는가?
- [ ] `alpha^{-1}` weighting을 사용할 경우 `alpha_min`으로 clamp했는가?

### Schedule

- [ ] fine-tuning 초반에는 `alpha = 1`인가?
- [ ] 중반에는 `alpha`를 부드럽게 줄이는가?
- [ ] 후반에는 `alpha ≈ alpha_min`으로 유지하는가?
- [ ] 필요하면 기존 FM loss를 섞었는가?

### Sampling

- [ ] 1-NFE에서 `r=0, t=1`로 sampling하는가?
- [ ] 2-NFE에서 intermediate timestep을 ablation하는가?
- [ ] 기존 multi-step FM sampling과 1-NFE / 2-NFE sampling을 비교하는가?

---

## 13. 추천 ablation

처음부터 너무 많은 실험을 하지 말고 다음 ablation부터 보는 것이 좋습니다.

### A. Time conditioning

1. `emb(t) + zero_init_mlp(emb(r))`
2. `emb(t) + zero_init_mlp(concat(emb(r), emb(t-r)))`
3. `MLP(concat(emb(t), emb(r), emb(t-r)))`

예상으로는 2번이 가장 안정적일 가능성이 높습니다.

### B. α schedule

1. no curriculum, fixed `alpha = alpha_min`
2. short transition
3. long transition

예상으로는 long transition이 더 안정적일 가능성이 높습니다.

### C. FM regularization

1. `lambda_fm = 0`
2. `lambda_fm = 0.1`
3. `lambda_fm = 0.5`

fine-tuning이 불안정하면 `lambda_fm`을 키우는 것이 좋습니다.

### D. Sampling NFE

1. original FM sampler, e.g. 10 NFE
2. α-Flow 1 NFE
3. α-Flow 2 NFE
4. α-Flow 4 NFE

목표는 original FM sampler의 성능을 최대한 유지하면서 NFE를 줄이는 것입니다.

---

## 14. 주의사항

### 14.1 α-Flow는 policy improvement method가 아니다

α-Flow fine-tuning만으로 policy가 task reward 관점에서 더 좋아진다는 보장은 없습니다.

기본적으로 α-Flow는 다음 목적에 적합합니다.

```text
pretrained Flow Matching policy의 distribution을 유지하면서 few-step sampler로 압축한다.
```

더 좋은 action을 찾고 싶다면 critic 또는 Q-guided objective를 추가해야 합니다.

---

### 14.2 Q-guided fine-tuning을 붙일 경우

Offline RL setting에서 critic을 사용할 경우 다음과 같은 objective를 추가할 수 있습니다.

```math
L_Q
=
-
\mathbb{E}_{o,z_1}
[
Q(o, \hat{a}_\theta(o))
]
```

전체 loss는 다음과 같습니다.

```math
L
=
L_\alpha
+
\lambda_Q L_Q
+
\lambda_{\mathrm{FM}}L_{\mathrm{FM}}
```

단, Q objective는 OOD action을 만들 위험이 있으므로 초반부터 강하게 넣지 않는 것이 좋습니다.

추천 schedule:

```text
early:  α-Flow only
middle: α-Flow + small Q loss
late:   α-Flow + Q loss + weak FM regularization
```

---

## 15. 최소 구현 요약

가장 먼저 구현할 버전은 다음이면 충분합니다.

```text
1. pretrained FM policy를 load한다.
2. model input을 u_theta(obs, z_t, r, t)로 확장한다.
3. time condition을 emb(t) + zero_init_mlp(concat(emb(r), emb(t-r)))로 바꾼다.
4. alpha schedule을 1 -> 5e-3으로 anneal한다.
5. L_alpha로 fine-tuning한다.
6. 필요하면 small FM loss를 추가한다.
7. 1-NFE, 2-NFE, original-NFE 성능을 비교한다.
```

핵심 코드는 다음입니다.

```python
time_cond = emb(t) + zero_init_mlp(torch.cat([emb(r), emb(t - r)], dim=-1))

s = alpha * r + (1.0 - alpha) * t
z_s = z_t - (t - s) * v_t

u_pred = model(obs, z_t, r, t)

with torch.no_grad():
    u_next = model(obs, z_s, r, s)
    u_target = alpha * v_t + (1.0 - alpha) * u_next

loss = ((u_pred - u_target) ** 2 / alpha.clamp_min(alpha_min)).mean()
```

이 구현은 pretrained Flow Matching policy를 보존하면서 α-Flow style few-step policy로 fine-tuning하기 위한 가장 간단하고 안정적인 출발점입니다.


---

## 16. 논문에 충실한 최종 단계: `alpha = 0`에서 JVP MeanFlow로 전환

앞 절에서는 구현을 단순화하기 위해 `alpha_min = 5e-3`에서 멈추는 discrete-only 버전을 설명했습니다.

그러나 **논문의 faithful implementation은 마지막에 `alpha = 0`으로 clamp한 뒤, JVP 기반 exact MeanFlow objective로 전환합니다.**

즉, α-Flow 학습은 다음 세 구간으로 나뉩니다.

```text
alpha = 1:
Trajectory Flow Matching

0 < alpha < 1:
Discrete α-Flow transition

alpha = 0:
JVP 기반 exact MeanFlow fine-tuning
```

### 16.1 논문의 branch 구조

```python
if alpha == 0:
    u, dudt = jvp(
        fn,
        (z_t, r, t),
        (v_t, 0, 1),
    )

    u_target = v_t - (t - r) * dudt

else:
    s = alpha * r + (1 - alpha) * t

    u = fn(z_t, r, t)

    z_s = z_t - (t - s) * v_t

    u_target = (
        alpha * v_t
        + (1 - alpha) * fn(z_s, r, s)
    )
```

`alpha > 0`에서는 두 번의 forward pass만 사용합니다.

`alpha = 0`에서는 discrete approximation 대신 JVP를 사용해 continuous MeanFlow objective를 직접 계산합니다.

### 16.2 왜 `alpha = 5e-3`에서 멈추지 않는가?

중간 timestep은 다음과 같습니다.

```math
s = \alpha r + (1-\alpha)t
```

따라서

```math
t-s = \alpha(t-r)
```

입니다.

`alpha -> 0`이면

```math
s -> t
```

가 됩니다.

즉, 두 timestep `t`와 `s`가 거의 같아집니다.

이때 discrete α-Flow target은 MeanFlow target에 가까워지지만, 두 시점 간 차이가 너무 작아져 numerical instability가 커질 수 있습니다.

따라서 논문은 `alpha`가 충분히 작아지면 discrete objective를 계속 사용하지 않고, infinitesimal limit를 JVP로 직접 계산합니다.

### 16.3 논문의 clamp threshold

논문은 다음 threshold를 사용합니다.

```math
\eta = 5 \times 10^{-3}
```

scheduler는 다음처럼 동작합니다.

```python
if alpha > 1 - eta:
    alpha = 1

elif alpha < eta:
    alpha = 0
```

즉 다음 세 구간이 생깁니다.

```text
alpha > 0.995:
alpha = 1
Trajectory Flow Matching branch

0.005 <= alpha <= 0.995:
Discrete α-Flow branch

alpha < 0.005:
alpha = 0
Exact MeanFlow JVP branch
```

### 16.4 JVP에서 계산하는 것

JVP branch에서는 다음 함수를 생각합니다.

```math
u_\theta(z_t, r, t)
```

trajectory를 따라 timestep이 변하면

```math
\frac{d z_t}{dt} = v_t
```

입니다.

`r`은 고정되어 있으므로 tangent는 다음과 같습니다.

```text
(z tangent, r tangent, t tangent)
=
(v_t, 0, 1)
```

따라서 JVP는 다음 total derivative를 계산합니다.

```math
\frac{d}{dt}u_\theta(z_t,r,t)
=
\frac{\partial u_\theta}{\partial z_t}v_t
+
\frac{\partial u_\theta}{\partial t}
```

코드에서는 다음처럼 표현할 수 있습니다.

```python
u, dudt = jvp(
    fn,
    primals=(z_t, r, t),
    tangents=(
        v_t,
        torch.zeros_like(r),
        torch.ones_like(t),
    ),
)
```

그리고 MeanFlow target은 다음과 같습니다.

```math
u_{\mathrm{target}}
=
v_t
-
(t-r)
\frac{d u_{\theta^-}(z_t,r,t)}{dt}
```

코드:

```python
u_target = v_t - (t_b - r_b) * dudt.detach()
```

---

## 17. 30K fine-tuning budget에서 추천하는 faithful schedule

현재 setup은 다음과 같습니다.

```text
pretrained Flow Matching policy
→ new fine-tuning dataset adaptation
→ α-Flow few-step conversion
```

논문 자체는 scratch ImageNet training을 다루므로, 아래 step 비율은 논문에서 직접 나온 값이 아닙니다.

다만 논문의 핵심 recipe를 유지하면서 30K fine-tuning budget에 맞게 조정한 실용적인 baseline입니다.

```text
Total: 30K gradient steps

0K ~ 6K:
alpha = 1
r-conditioned Trajectory Flow Matching
new dataset adaptation + r-condition branch warmup

6K ~ 24K:
sigmoid alpha transition
alpha: 1 -> 0

실제로는
alpha > 0.995            -> TFM branch
0.005 <= alpha <= 0.995 -> discrete α-Flow branch
alpha < 0.005           -> JVP MeanFlow branch

24K ~ 30K:
alpha = 0
JVP 기반 exact MeanFlow fine-tuning
```

비율로 보면 다음과 같습니다.

```text
20%:
Trajectory Flow Matching warmup

60%:
Discrete α-Flow transition

20%:
Exact MeanFlow JVP refinement
```

---

## 18. discrete-only baseline과 faithful baseline을 둘 다 돌리는 이유

처음부터 JVP까지 구현하면 debugging이 어려울 수 있습니다.

따라서 아래 두 버전을 분리해 비교하는 것이 좋습니다.

### 18.1 Discrete-only baseline

```text
0K ~ 6K:
alpha = 1

6K ~ 30K:
alpha: 1 -> 5e-3

final:
alpha = 5e-3 유지
JVP 없음
```

장점:

- 구현이 단순함
- forward pass 두 번으로 끝남
- pipeline 검증이 쉬움

단점:

- exact MeanFlow refinement가 없음
- 논문의 최종 recipe와 다름

### 18.2 Faithful α-Flow baseline

```text
0K ~ 6K:
alpha = 1

6K ~ 24K:
alpha: 1 -> 0

24K ~ 30K:
alpha = 0
JVP MeanFlow refinement
```

장점:

- 논문의 최종 formulation에 가까움
- 1-NFE 성능을 더 끌어올릴 가능성이 높음

단점:

- JVP 구현 필요
- 메모리와 runtime 부담 증가
- FSDP, gradient checkpointing, compile과의 호환성 확인 필요

---

## 19. 추천 실험 순서

### Experiment A. Standard FM baseline

```text
30K standard Flow Matching fine-tuning
original multi-step FM sampling
```

목적:

```text
새 fine-tuning dataset에 대한 adaptation 성능 확인
```

### Experiment B. Discrete-only α-Flow

```text
6K TFM warmup
+
24K discrete α-Flow transition
```

목적:

```text
JVP 없이도 NFE 감소가 가능한지 확인
```

### Experiment C. Faithful α-Flow

```text
6K TFM warmup
+
18K discrete α-Flow transition
+
6K JVP MeanFlow refinement
```

목적:

```text
JVP refinement가 robot policy에서도 실제로 필요한지 확인
```

---

## 20. PyTorch JVP 구현 예시

```python
import torch
from torch.func import jvp


def compute_meanflow_jvp(
    model,
    obs,
    z_t,
    r,
    t,
    v_t,
):
    """
    Compute:
        u_theta(z_t, r, t)
        d/dt u_theta(z_t, r, t)

    along the trajectory direction:
        dz_t / dt = v_t
        dr / dt = 0
        dt / dt = 1
    """

    def fn(z, r_value, t_value):
        return model(obs, z, r_value, t_value)

    primals = (
        z_t,
        r,
        t,
    )

    tangents = (
        v_t,
        torch.zeros_like(r),
        torch.ones_like(t),
    )

    u_pred, dudt = jvp(
        fn,
        primals,
        tangents,
    )

    return u_pred, dudt
```

MeanFlow branch:

```python
u_pred, dudt = compute_meanflow_jvp(
    model=model,
    obs=obs,
    z_t=z_t,
    r=r,
    t=t,
    v_t=v_t,
)

u_target = v_t - (t_b - r_b) * dudt.detach()

error = u_pred - u_target
```

---

## 21. JVP branch의 adaptive loss

`alpha = 0`에서는 α-aware weight 대신 MeanFlow weight를 사용합니다.

```python
reduce_dims = tuple(range(1, error.ndim))
sq_error = error.pow(2).mean(dim=reduce_dims)

weight = 1.0 / (sq_error.detach() + 1e-3)
loss = (weight.detach() * sq_error).mean()
```

정리하면:

```python
if alpha > 0:
    weight = alpha / (sq_error.detach() + 1e-3)

else:
    weight = 1.0 / (sq_error.detach() + 1e-3)
```

---

## 22. faithful scheduler 예시

```python
import math


def get_alpha_faithful(
    step: int,
    warmup_end: int = 6_000,
    transition_end: int = 24_000,
    eta: float = 5e-3,
    gamma: float = 25.0,
) -> float:
    """
    0 ~ warmup_end:
        alpha = 1

    warmup_end ~ transition_end:
        sigmoid transition

    transition_end ~:
        alpha = 0
        exact MeanFlow JVP branch
    """

    if step <= warmup_end:
        return 1.0

    if step >= transition_end:
        return 0.0

    midpoint = 0.5 * (warmup_end + transition_end)
    width = transition_end - warmup_end

    x = gamma * (step - midpoint) / width
    sigmoid = 1.0 / (1.0 + math.exp(-x))

    alpha = 1.0 - sigmoid

    if alpha > 1.0 - eta:
        return 1.0

    if alpha < eta:
        return 0.0

    return alpha
```

---

## 23. 30K faithful training pseudocode

```python
def train_step(
    model,
    obs,
    x,
    step,
    fm_ratio=0.25,
    c=1e-3,
):
    """
    x:
        clean action or action chunk
        shape [B, horizon, action_dim]
        or [B, action_dim]
    """
    B = x.shape[0]
    device = x.device

    eps = torch.randn_like(x)

    # ---------------------------------------------------------
    # 1. Sample t and r
    # ---------------------------------------------------------
    t = torch.rand(B, device=device)
    r = torch.rand(B, device=device) * t

    use_border_fm = torch.rand(B, device=device) < fm_ratio
    r = torch.where(use_border_fm, t, r)

    # ---------------------------------------------------------
    # 2. Alpha schedule
    # ---------------------------------------------------------
    alpha = get_alpha_faithful(step)

    # ---------------------------------------------------------
    # 3. Construct flow trajectory
    # ---------------------------------------------------------
    view_shape = (B,) + (1,) * (x.ndim - 1)

    t_b = t.view(view_shape)
    r_b = r.view(view_shape)

    z_t = (1.0 - t_b) * x + t_b * eps
    v_t = eps - x

    # ---------------------------------------------------------
    # 4. TFM branch
    # ---------------------------------------------------------
    if alpha == 1.0:
        u_pred = model(obs, z_t, r, t)
        u_target = v_t

    # ---------------------------------------------------------
    # 5. Discrete alpha-flow branch
    # ---------------------------------------------------------
    elif alpha > 0.0:
        alpha_tensor = torch.full_like(t, alpha)
        alpha_b = alpha_tensor.view(view_shape)

        s = alpha_tensor * r + (1.0 - alpha_tensor) * t
        s_b = s.view(view_shape)

        z_s = z_t - (t_b - s_b) * v_t

        u_pred = model(obs, z_t, r, t)

        with torch.no_grad():
            u_next = model(obs, z_s, r, s)

        u_target = alpha_b * v_t + (1.0 - alpha_b) * u_next

    # ---------------------------------------------------------
    # 6. Exact MeanFlow JVP branch
    # ---------------------------------------------------------
    else:
        u_pred, dudt = compute_meanflow_jvp(
            model=model,
            obs=obs,
            z_t=z_t,
            r=r,
            t=t,
            v_t=v_t,
        )

        u_target = v_t - (t_b - r_b) * dudt.detach()

    # ---------------------------------------------------------
    # 7. Adaptive weighting
    # ---------------------------------------------------------
    error = u_pred - u_target

    reduce_dims = tuple(range(1, error.ndim))
    sq_error = error.pow(2).mean(dim=reduce_dims)

    if alpha > 0.0:
        weight = alpha / (sq_error.detach() + c)
    else:
        weight = 1.0 / (sq_error.detach() + c)

    loss = (weight.detach() * sq_error).mean()

    return loss
```

---

## 24. JVP branch 구현 시 체크리스트

### Correctness

- [ ] tangent `z_t`는 `v_t`인가?
- [ ] tangent `r`은 0인가?
- [ ] tangent `t`는 1인가?
- [ ] `u_target = v_t - (t-r) * dudt`인가?
- [ ] `dudt` target에는 gradient를 흘리지 않는가?

### Stability

- [ ] JVP 시작 시점에서 gradient norm spike가 발생하는가?
- [ ] JVP branch에서 loss variance가 급증하는가?
- [ ] adaptive MeanFlow weight를 사용하는가?
- [ ] gradient clipping을 적용하는가?
- [ ] JVP phase learning rate를 낮출 필요가 있는가?

### Systems

- [ ] `torch.func.jvp`가 현재 모델 구조에서 동작하는가?
- [ ] FSDP와 함께 동작하는가?
- [ ] gradient checkpointing과 함께 동작하는가?
- [ ] compile을 쓰고 있다면 JVP branch와 호환되는가?
- [ ] JVP phase에서 메모리 사용량을 따로 측정했는가?

---

## 25. 최종 권장 recipe

첫 번째 실험은 아래 세 가지를 나란히 비교하는 것이 좋습니다.

```text
A. Standard FM
30K standard Flow Matching fine-tuning

B. Discrete-only α-Flow
6K TFM
+
24K discrete transition
+
JVP 없음

C. Faithful α-Flow
6K TFM
+
18K discrete transition
+
6K exact MeanFlow JVP refinement
```

평가는 다음 기준으로 진행합니다.

```text
original FM NFE
4 NFE
2 NFE
1 NFE
task success rate
action smoothness
latency
GPU memory
training throughput
```

핵심 결론은 다음입니다.

```text
논문에서는 alpha가 작아질 때
5e-3에 멈추지 않는다.

alpha < 5e-3가 되면
alpha = 0으로 clamp하고
JVP 기반 exact MeanFlow objective로 넘어간다.
```
