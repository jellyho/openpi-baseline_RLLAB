# Alpha-Flow 1-NFE Troubleshooting

> 목적: pi05를 alpha-flow로 fine-tune해서 **1-NFE 정책**을 만드는데 1-NFE 품질이 나쁨.
> 이 문서는 의심 가설 / 확인된 사실 / 시도한 수정 / 다음 액션을 한 곳에 모아 계속 갱신.

## 증상 (Ground truth)
- **30k 풀 학습, 마지막 체크포인트(=JVP 구간 끝)인데도 1-NFE 샘플이 실제로 나쁨.**
  → 단순 "더 학습하면 됨"(증류 부족)은 배제. objective/구현 어딘가가 잘못.
- warmup(alpha=1) 구간에서도 loss가 pi05 FM과 꽤 다름.

## 확인된 사실 (Facts)
- **F1. 옛날 "잘 됐던" 실험(2d9207d)은 alpha-flow가 아니었음.** 그 시점 `Pi0AlphaFlow`에 `compute_loss`가 없어서 `scripts/train.py`가 부모 `Pi0.compute_loss`(= 표준 flow matching)를 호출. 즉 plain pi05 FM이 돌았고, 그래서 초반 loss가 pi05와 동일했던 것. **alpha-flow는 한 번도 검증된 적 없음.**
- **F2. adaptive_l2는 loss 값이 안 떨어지는 게 정상.** `loss = sq_err/(sq_err+c)*w ≈ w(≈1)`로 포화. 실측: adaptive=0.9996 vs raw MSE=3.03 (완전히 다른 metric). wandb `loss/alphaflow`가 pi05와 안 맞아 보인 건 metric이 달라서.
- **F3. sphere vs Gaussian latent는 거의 동일.** 고차원(50×32=1600) concentration of measure로 Gaussian norm도 39~40.5 ≈ sphere(40). → sphere는 주범 아닐 가능성 높음 (사용자도 동의).
- **F4. 원본 alpha-flow는 (t, r)을 더함.** `dit.py`: `c = emb(t) + emb(r) + emb(label)`. t와 r 각각 **동등한 full TimestepEmbedder(표준 init)**.
- **F5. t convention은 pi0/원본/우리 모두 동일 (notation 버그 아님).** 셋 다 `x_t = t·noise + (1-t)·actions`, `v = noise - actions`, **t=1=noise / t=0=clean**, 적분 t:1→0. 1-NFE: `z_0 = noise - u(noise, t=1, r=0)`. 원본 `x_denoised = x_cur - t_cur·v`와 일치. → notation 불일치 가능성 배제.

## 가설 (Hypotheses) — 유력순
- **H1 (유력). r conditioning이 너무 약했음.**
  - 이전: r을 **단일 Linear, zero-init**(`r_proj`)으로 adaRMS에 잔차 add. t는 2-layer MLP(`time_mlp`)인데 r은 단일 linear → 용량 비대칭 + 0에서 시작.
  - mean velocity는 구간 [r,t]에 의존 → r conditioning이 약하면 instantaneous velocity로 collapse → 1-NFE 실패.
  - **시도함 → T1 참고.**
- **H1b (H1의 잔여 의심). r_mlp_out zero-init 자체가 문제일 수 있음.**
  - warmup(alpha=1)에선 target=v_t라 r이 무의미 → r_mlp이 gradient를 거의 못 받음. warmup 0.3(9k step) 동안 r_mlp이 거의 안 배우다가 transition에서 0에서 급히 끌어올려야 함.
  - 원본은 r 임베더도 표준 init이라 처음부터 r에 반응.
  - **후보 수정: r_mlp_out도 표준 init(0 아님)으로. trade-off: 시작 시 pretrained 살짝 흔들림(회복 가능).**
- **H2. time sampler 차이.**
  - pi05: `beta(1.5,1)*0.999+0.001` (t를 노이즈 t=1 쪽으로 치우침).
  - 우리: min-max sigmoid(`sigmoid(normal-0.4)`의 max/min) → t가 0.5~0.7 중심.
  - t 분포가 다르면 같은 step에서 z_t 난이도/loss 다름. warmup이 pi05와 다르게 보이는 실제 원인 후보.
  - **후보 수정: warmup/FM 구간은 pi05의 beta sampler 쓰도록.**
- **H3. adaptive_l2의 gradient 댐핑.**
  - gradient ~ `w·2·err/(sq_err+c)` → error 클수록 gradient 작아짐(MSE와 반대). fine-tune 초반 큰 error에서 학습이 느림. c=1e-3가 우리 action scale(sq_err~3)엔 너무 작을 수 있음.
  - **후보 수정: warmup은 plain MSE 사용 / 또는 c 키우기.**
- **H4. MeanFlow target 버그 (discrete/JVP branch).** 가능성 낮음(원본과 대조해 검증했었음)이지만 H1~H3로 안 풀리면 재점검.

## 시도한 수정 (Tried)
- **T1. r conditioning 강화 (H1).** `r_proj`(단일 Linear, zero-init) → **`r_mlp_in → swish → r_mlp_out → swish`** (t의 time_mlp 미러). `r_mlp_out`만 zero-init → `swish(0)=0` → 시작 시 r 기여 0(pretrained 보존)이되 `r_mlp_in`은 full 용량. 3개 모델(alphaflow/critic/lps_rft) 전부.
  - 검증: r=0 vs r=1 adarms 차이 0(시작 보존), r_mlp_in |sum|=26912(full), compute_loss/1nfe 정상.
  - **결과: (실험 대기) `./train.sh pi05_alphaflow_tabletop_rl_orig 1 32 30000`**
- **T2. `loss/l2_raw` 로깅 추가.** plain MSE(= `mean((u_pred-u_tgt)²)`, pi05 FM과 같은 scale)를 aux에 추가. adaptive 착시 없이 pi05와 직접 비교용. alphaflow/critic 둘 다.
- **T3. `sphere_latent` 토글 옵션 (isolate용).** `Pi0AlphaFlowConfig(sphere_latent=False)` → Gaussian. **기본 True (현재 모든 config가 sphere 사용 중).**
- **T5. inference clip 제거.** `sample_actions_1nfe`/`sample_actions_nfe`/LPS `_decode`의 최종 `clip(·, -1, 1)` 제거 → raw `z_0` 반환 (pi0와 일관). 1-NFE 출력이 [-1,1] 밖으로 폭발하는지 보이게 함 (clip이 학습 실패를 가리는지 진단). training의 `clip(u_tgt, ±10)`은 원본 충실하게 **유지**.
- **T4. `time_sampler` 토글 옵션 (H2 검증용).** `Pi0AlphaFlowConfig(time_sampler="beta")` → pi05의 `beta(1.5,1)*0.999+0.001`로 두 time 샘플 후 max=t/min=r. 기본 `"minmax"`(sigmoid(normal-0.4)). minmax는 t≈0.5~0.7 중심, beta는 t를 노이즈(t=1) 쪽으로 치우침(pi05와 동일).

## 진단 도구 (Diagnostics)
- **1-NFE vs 10-NFE 비교** (`sample_actions_nfe(num_steps=10)` vs `_1nfe`):
  - 10-NFE 좋음 / 1-NFE 나쁨 → 증류(MeanFlow) 문제.
  - 둘 다 나쁨 → 바탕 flow 자체 깨짐.
- **`loss/l2_raw`** (T2): warmup에서 pi05 FM처럼 줄면 base 학습 정상.
- **`visualize_critic.sh`** 패턴으로 학습 없이 추론만 평가 가능.

## O5. JVP 발산 — 코드 버그 아님, 원본도 JVP엔 clip 없음
- JVP 구간(alpha=0) 진입 직후 l2_raw가 11로 치솟고 계속 오름 = **JVP MeanFlow 발산**.
- 원본 대조: discrete(`_compute_mean_velocity_d`)엔 `clip(±clamp_utgt)` 있지만 **JVP(`_compute_mean_velocity_c`)엔 clip 없음.** 우리도 동일 → **JVP clip 누락은 버그/일탈 아님.**
- discrete/JVP 두 branch 수식·tangent·grad 흐름을 원본과 정밀 대조 → **코드 버그 없음** (discrete target=alpha·v+(1-alpha)·u_next ✓, JVP tangent (v_t,dt=1,dr=0) & target v_t-(t-r)·dudt ✓).
- 결론: 원본 JVP는 그들 regime(이미지 DiT, scratch)에선 안정적이나 **우리 2B VLA fine-tune regime에서 불안정**. 코드가 아니라 세팅 문제.
- **불연속 메커니즘 3겹**: (1) alpha 스케줄이 eta 아래서 0으로 snap → discrete→JVP 급전환, (2) discrete loss가 alpha 스케일에 눌려 "수렴한 듯" 보였으나 실제 mean velocity 미학습(alpha≲0.03에서 adaptive eps c=1e-3가 gradient 억제), (3) JVP target clip 없음 → dudt 폭발 자기강화.
- **수정 (H5)**: JVP를 안 쓰고 discrete-only. `use_jvp=False` → alpha가 alpha_min에서 floor(0으로 snap 안 함) → 항상 discrete branch.
- **원본도 동일 옵션 보유**: `loss.py:421-425`의 `discrete_training` 플래그. 켜지면 `ratio<clamp_value`일 때 0이 아니라 **clamp_value로 floor** → 정확히 우리 use_jvp=False. 즉 우리 구현이 논문 구현과 일치.
- alpha_min(=floor)은 논문 기본값(5e-3) 그대로. (이전에 0.05로 올렸던 건 철회 — 논문 구현 따름.)

## 실험 config (한 번에 하나씩 isolate)
| config | H | 변경 | 상태 |
|---|---|---|---|
| `pi05_alphaflow_tabletop_rl_orig` | H1 (r_mlp) | 기본 (minmax/sphere/jvp) | warmup OK, **JVP서 발산** |
| `pi05_alphaflow_tabletop_rl_orig_nojvp` | **H5** | **use_jvp=False** (discrete-only, alpha_min 기본 5e-3) | **대기 (유력 후보)** |
| `pi05_alphaflow_tabletop_rl_orig_beta` | H2 | beta time | 중단 (O2: 효과 없음) |
| `pi05_alphaflow_tabletop_rl_orig_gaussian` | H3 | gaussian | 중단 (O2: sphere와 동일) |

실행: `./train.sh <config> 1 32 30000`. 비교 지표: `loss/l2_raw`(pi05 FM scale), 학습 후 1-NFE vs 10-NFE.

## 진단 보강 (Diagnostics 추가)
- **`l2_raw`는 alpha=1(warmup)에서만 FM L2를 의미.** `raw_l2 = mean((u_pred - u_tgt)²)`인데 `u_tgt = alpha·v_t + (1-alpha)·u_next`. alpha<1(transition/JVP)에선 target이 mean-velocity로 바뀌므로 **transition 이후 l2_raw를 pi05 FM과 직접 비교하면 안 됨** (target 자체가 다름).
- **eval NFE 선택**: `serve_policy.py --num-steps N --nfe-mode {mean|fm}` (top-level 옵션이라 `policy:checkpoint` **앞에** 와야 함 — tyro subcommand 순서). `nfe_mode="mean"`=r=t_next(MeanFlow), `"fm"`=r=t(instantaneous, pi05식 Euler). **10k처럼 MeanFlow 학습 전 체크포인트는 `--num-steps 10 --nfe-mode fm`** 으로 base flow 평가.
- **O4. fm vs mean 모드는 학습 안 된 모델에선 동일(차이 0).** r_mlp_out zero-init → swish(0)=0 → r 무시 → r=t든 r=t_next든 같은 velocity. **시사점: warmup(alpha=1)은 target=v_t라 r_mlp gradient≈0 → r_mlp이 거의 안 배움. 따라서 10k 체크포인트도 r_mlp≈0일 가능성 높고, fm/mean이 비슷하게 나올 것.** → H1b(r_mlp zero-init이 r 학습을 transition까지 지연)와 직결. 1-NFE에 필요한 mean velocity를 transition+JVP(짧음) 동안만 배워야 하는 구조적 부담.

## Gotchas (코드 구조 함정)
- **G1. `Pi0WithCritic` / `Pi0LPSRFT`는 부모 `Pi0AlphaFlow.__init__`을 우회한다** (4-expert Gemma를 새로 구성하려고 `_model.BaseModel.__init__`만 호출). 따라서 **부모 `__init__`에 인스턴스 속성을 추가하면 두 자식에도 똑같이 복사해야 함.** 안 하면 `AttributeError` (예: `sphere_latent`, `time_sampler`를 부모에만 추가했다가 critic 학습 시 `'Pi0WithCritic' object has no attribute 'sphere_latent'`). 새 config 필드/속성 추가 시 3곳(alphaflow/critic/lps_rft __init__) 전부 확인.
- **참고**: LPS는 offline RL이라 alpha-flow loss를 안 써서 aux에 `loss/l2_raw` 없음(정상). critic은 있음.

## 진행 관찰 (Observations)
- **O1. H1 @1k step: `loss/l2_raw`가 pi05 FM과 비슷하게 감소 중.** → warmup(alpha=1)에서 base velocity field가 pi05처럼 정상 학습됨. adaptive_l2는 로깅 착시였음(F2 확인). r_mlp zero-init이 warmup 보존 잘 함. **바탕 flow는 정상 → 문제는 transition/JVP(mean-velocity 학습) 또는 1-NFE 증류 쪽으로 좁혀짐.**
- **O2. H2(beta) loss가 minmax보다 오히려 큼; H3(gaussian)는 sphere와 차이 없음.** → 둘 다 1-NFE 실패의 주범 아님. H2/H3 실험 중단. (F3 재확인: sphere≈gaussian)
- **O3. critic 버전(critic H1, `pi05_alphaflow_critic_tabletop_orig`) 초반 `loss/l2_raw`가 plain alphaflow와 큰 차이 없음.** → critic from-scratch의 gradient가 action 학습을 망치지 않음. critic stop-grad(kv_cache)가 backbone 보호 잘 함. 우려했던 global-grad-norm 경합(critic이 clip 잡아먹는 문제)은 초반엔 안 보임 → **per-group clipping(B안) 일단 불필요.**

## 다음 액션 (Next)
1. **H1 (rl_orig) 결과 대기** → warmup은 OK 확인됨. transition/JVP 구간 지난 뒤 1-NFE가 개선되는지가 관건. (transition≈9k~21k, JVP≈21k~30k)
2. 안 되면: H2(beta) / H3(gaussian) config로 isolate.
3. 그래도 안 되면: H1b(r_mlp_out 표준 init) → H3-adaptive(warmup plain MSE) → H4(target 재점검).
