# Alpha-Flow 1-NFE Troubleshooting

> 목적: pi05를 alpha-flow로 fine-tune해서 **1-NFE 정책**을 만드는데 1-NFE 품질이 나쁨.
> 이 문서는 의심 가설 / 확인된 사실 / 시도한 수정 / 다음 액션을 한 곳에 모아 계속 갱신.

## 증상 (Ground truth)
- **30k 풀 학습, 마지막 체크포인트(=JVP 구간 끝)인데도 1-NFE 샘플이 실제로 나쁨.**
  → 단순 "더 학습하면 됨"(증류 부족)은 배제. objective/구현 어딘가가 잘못.
- warmup(alpha=1) 구간에서도 loss가 pi05 FM과 꽤 다름.

## 📌 현재 상태 종합 (한눈에)

### 원인 규명 타임라인
1. **bad 1-NFE 발견** → "더 학습하면 됨" 배제 (30k 풀 학습도 나쁨).
2. warmup/notation/sphere/beta 차례로 격리 → 전부 주범 아님 (F1~F5, O1~O3).
3. **JVP 구간서 l2_raw 발산** (11→상승). 코드 버그 아님 확인 (O5).
4. **O6 ⭐ — JVP가 bf16에서 돌고 있었음.** 원본은 fp32. forward-mode AD(`dudt`)는 정밀도에 민감 → bf16서 tangent 노이즈 → 발산. discrete(forward eval)는 bf16 OK.
5. **O7 ⭐⭐ — AlphaFlowTSE 논문(2603.10701) 정독.** ① JVP-free가 정답(그들도 bf16). ② **우리 MF 가중 `α/(m+c)`가 small-α 샘플(=배우려는 mean-velocity)을 죽이고 있었음** → bounded `κ/(m+ακ+ε)`로 교체. ③ α_min=0.1, large-span 15%, 50/50 split.

### 두 줄 결론
> **JVP를 쓰지 마라**(bf16서 발산; 쓸 거면 fp32). 그리고 **MF loss를 α로 곱하지 마라**(small-α 억제) — bounded reweighting을 써라. 나머지(α_min=0.1, large-span, split)는 1-NFE 품질 보강.

### 제어 축 (config 플래그)
> **⭐ 2026-06-02 정리:** AlphaFlowTSE 레시피를 `Pi0AlphaFlowConfig` **기본값**으로 승격. 탐색용 ablation config(nojvp/beta/gaussian/mf_from_start/paper*)는 전부 삭제. 변경점은 전부 flag로 남아 configurable. JVP 코드+`jvp_fp32` 옵션은 코드에 보존.
>
> **⭐ 2026-06-02 (2차):** config을 **4-카테고리 RFT 파이프라인**으로 재편 (README 참고) — (1) FM baseline `pi05_tabletop{,_bc}`, (2) `pi05_alphaflow_critic_{rl,bc}` (joint, from base, warmup 25/50/25), (3) `pi05_rft_phase1_{rl,bc}` (rectify + critic warmup, FM ckpt에서 init, **VLM freeze** `get_rectify_freeze_filter`, warmup=0), (4) `pi05_rft_phase2_rl` (LPS). 튜닝 레시피 = `flow_ratio=0.25` + `lambda_fm=lambda_mf=0.5` + `large_span_warmup_gate=True`(warmup엔 large-span off). 나머지 tabletop ablation(fm25/fmext/rl_orig/rl/jvp 등)은 삭제 → 아래 "등록된 config" 표는 **stale**.
>
> **⭐ Critic = multi-horizon value (warmup) / single-Q (LPS).** causal mask라 token `h-1` = `Q(s,a₀:h)`. **warmup**(cat-2/cat-3): `critic_horizons`(기본 `(5,10,25,50)`) 토큰을 읽어 `[b,K,n_bins]` C51 head K개, **전부 chunk MC return `G_t`로 supervise**. (MC에선 telescoping이라 horizon별 타겟이 전부 `G_t` 동일 → warmup은 head를 MC 스케일에 예열만; horizon 차별화는 LPS chunked-TD에서.) **LPS phase2**: 지금은 full-chunk head 1개만 single-Q로 사용 — multi-horizon을 RL TD에서 어떻게 쓸지는 TBD. `critic_horizons` 플래그. 옛 per-token(ah개) broadcast는 폐기. checkpoint 호환(`critic_out_proj` 그대로).

### TSE 기본값 (Pi0AlphaFlowConfig defaults)
| 플래그 | 기본값 | 의미 / 대안 |
|---|---|---|
| `warmup_ratio`/`transition_ratio` | `0.05` / `0.667` | FM 5% / transition 61.7% / floor 33.3% |
| `alpha_gamma` | `15.0` | sigmoid steepness (paper k) |
| `alpha_min` | `0.1` | discrete floor (JVP 극한과 거리) |
| `alpha_eta` | `5e-3` | boundary-snap threshold (floor와 분리) |
| `use_jvp` / `jvp_fp32` | `False` / `True` | discrete-only; JVP 쓸 땐 fp32 (O6) |
| `mf_reweight` / `reweight_kappa` | `bounded` / `1.0` | κ/(m+ακ+ε), small-α 억제 해소 (O7 핵심) |
| `lambda_fm` / `lambda_mf` | `0.6` / `0.4` | branch 가중 |
| `flow_ratio` | `0.5` | FM/MF 50:50 |
| `large_span_ratio` | `0.15` | full-span(t≥0.85,r≤0.15) 오버샘플 |
| `delta_conditioning` | `True` | emb(t)+emb(Δ) |
| `mf_loss_weight` | `None` | adaptive 모드 전용 (α 곱 우회) |
| `time_sampler`/`sphere_latent` | `minmax`/`True` | (O2: 둘 다 영향 없음) |

### 등록된 config (9개, 전부 TSE 기본)
| config | 데이터 / 차이 |
|---|---|
| `pi05_alphaflow_tabletop_bc_orig` | bc_orig (성공 데이터) |
| `pi05_alphaflow_tabletop_rl_orig` | rl_orig_mc |
| `pi05_alphaflow_tabletop_rl` | rl_mc |
| `pi05_alphaflow_tabletop_jvp` | rl_orig_mc, **use_jvp=True**(transition 0.8, 마지막 20% JVP) — JVP 레퍼런스 |
| `pi05_alphaflow_critic_tabletop` | rl_mc + C51 critic |
| `pi05_alphaflow_critic_tabletop_orig` | rl_orig_mc + critic |
| `pi05_lps_rft_tabletop` | LPS-RFT (critic 체크포인트 로드) |
| `pi05_alphaflow_insert-mouse-battery` / `…seal-water-bottle-cap` | 실제 태스크 |

### 권장 실험 순서
1. **`pi05_alphaflow_tabletop_rl_orig`** (또는 bc_orig) — TSE 기본 레시피. 1순위.
2. `pi05_alphaflow_tabletop_jvp` — fp32 JVP가 발산 잡는지(O6 확정), discrete와 비교.
- 실행 `./train.sh <config> 1 32 30000`. 비교: `loss/l2_raw`(pi05 FM scale) + 학습 후 **1-NFE vs 10-NFE** 품질.
- ⚠️ batch 32에선 large-span 샘플이 스텝당 ~2개 → 필요시 batch↑ 또는 large_span_ratio↑.

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

## O6. ⭐ 가장 유력한 구현 레벨 원인 — JVP가 bf16에서 돌고 있었음
- pi0 backbone+suffix는 `dtype="bfloat16"` 기본 ([pi0_config.py:20](../src/openpi/models/pi0_config.py)). gemma는 `dtype=x.dtype`로 weight까지 activation dtype에 맞춰 캐스팅 → **suffix forward 전체가 bf16**.
- 원본 alpha-flow는 autocast/amp **전혀 없음** → fp32. 즉 **원본 JVP는 fp32, 우리 JVP는 bf16** — 명백한 구현 차이.
- 왜 JVP만 죽나: **discrete=forward eval만** (bf16 OK, pi0 학습 정밀도) / **JVP=forward-mode AD(`dudt`)** — RMSNorm/attention/residual 18+층을 tangent가 통과하며 누적. tangent는 정규화 안 됨(RMSNorm JVP는 오히려 증폭). bf16(rel.err ~0.4%/op)에서 `dudt` 상대오차 수십% → `u_tgt=v-(t-r)·dudt` 노이즈 → 부트스트랩이 노이즈 추종 → 발산. **discrete 안정/JVP 발산 관찰을 정확히 설명.**
- 부차: discrete `clamp_utgt` 우리 10.0 vs 원본 4.0 ([alphaflow.yaml:58](../third_party/alphaflow/configs/loss/alphaflow.yaml)). JVP 발산 직접원인 아님(discrete만).
- **수정 (B 구현)**: `jvp_fp32: bool = True`. JVP branch에서 suffix 토큰+adaRMS cond를 fp32로 캐스팅 → gemma가 weight 자동 upcast → suffix forward 전체 fp32. cached prefix KV(bf16)는 attention concat에서 fp32로 promote(검증함). backbone은 여전히 bf16(JVP는 alpha=0 구간만 → 비용 제한적). `jvp_fp32=False`로 bf16 발산 재현 가능.
- lax.cond 두 branch 출력 dtype 일치 필요 → 양쪽 `.astype(fp32)` 고정 (`v_t`가 어차피 fp32로 promote하지만 안전장치).
- **검증 우선순위 1**: `rl_orig`(이제 fp32 JVP 기본) 재실행 → 기존 bf16 발산 run과 비교. 안정화되면 O6 확정 = "regime 아니라 bf16".

## O7. ⭐⭐ AlphaFlowTSE 논문 (2603.10701) — 우리 문제 직격 + 핵심 단서들
> JVP-free AlphaFlow로 one-step 생성하는 논문 (TSE 도메인이지만 objective는 동일). `docs/2603.10701v1 (1).pdf`.

1. **JVP-free가 검증된 정답.** 논문 전체가 JVP를 의도적으로 회피. 인용: JVP는 *"overhead를 늘리고 supervision term들이 상호작용할 때 optimization을 불안정하게 만든다."* → 우리 `use_jvp=False` 방향 강력 지지.
2. **그들도 bf16인데 잘 됨 (O6 재확인).** §4.2: *"AdamW with **bfloat16 mixed precision**, grad clip 0.5"*. **JVP-free라서** bf16 OK. → bf16은 보편적 문제 아니라 **JVP 전용 문제**. discrete-only면 bf16 그대로 써도 됨. jvp_fp32는 JVP 고집할 때만 의미.
3. **⭐ reweighting이 우리랑 다름 (가장 중요한 코드 단서).**
   - 우리: `weight = α/(m+c)` (weight_scale=alpha). **α→0이면 weight→0 → MF loss 소멸 (gradient 억제).**
   - 논문 (Eq.18): `ℓbnd = sg[κ/(m+ακ+ε)]·m`. **α→0이면 weight→κ/(m+ε) = FM과 동급 full strength.** α는 분모 saturation만 조절. *"informative sample을 amplify하되 α 작을 때 과도한 weight는 방지(bounded α⁻¹)."*
   - 즉 **우리 구현이 small-α MF를 죽이고 있던 것**(O5 메커니즘2 확정). `mf_loss_weight=1.0`은 거친 근사 — 논문의 `κ/(m+ακ+ε)`가 정석.
4. **αmin=0.1 (0도 5e-3도 아님).** sigmoid로 1→0.1 anneal (epoch 5-100, k=15). **JVP 극한 근처(α→0)에 절대 안 감.** 우리 floor 5e-3은 너무 작음 → 0.1 권장.
5. **large-span 오버샘플링.** MF 시간쌍은 logit-normal (µ,σ)=(-0.4,1.0) [= 우리 minmax와 동일] + **15%를 t≤0.15, r≥0.85 큰 구간에서 추가 샘플.** 추론이 full-span (t=1,r=0)이므로 큰 구간을 일부러 더 봄. **우리엔 없음 → 1-NFE 품질에 직접 영향 가능.**
6. **branch 50/50 + λFM=0.6, λMF=0.4.** 우리 flow_ratio=0.25 (FM 25%)와 다름.
7. **conditioning은 emb(t)+emb(∆), ∆=r-t** (interval length). 우리는 (t,r) 따로. ∆ 파라미터화가 더 자연스러울 수 있음.

→ **종합 권장 config**: use_jvp=False + 논문식 bounded reweighting + αmin=0.1 + large-span 15% + 50/50 split. (bf16 그대로 OK.)

### O7 구현 완료 — `pi05_alphaflow_tabletop_paper`
- 새 config 추가. 플래그: `mf_reweight="bounded"`, `reweight_kappa=1.0`, `alpha_min=0.1`, `large_span_ratio=0.15`, `flow_ratio=0.5`, `lambda_fm=0.6`, `lambda_mf=0.4`, `use_jvp=False`, `warmup=0.05/transition=0.67`.
- 코드: `_bounded_l2_loss` 추가([pi0_alphaflow.py]), `_discrete_branch`가 mf_reweight 분기 + λ 가중, `_sample_flow_inputs`에 large-span overwrite, `embed_suffix_with_r`에 `delta_conditioning` 토글(기본 False).
- 검증: bounded는 α=0.005에서 loss 0.94 (adaptive는 0.005로 붕괴) — small-α 억제 해소 확인. large-span 슬라이싱·FM영역 무손상 확인.
- `delta_conditioning`은 **기본 True** (전 config 적용). r_mlp_out zero-init이라 init 시점엔 r-cond와 동일 → 안전. emb(t)+emb(Δ=t-r).
- 기존 config 전부 기본값 유지(mf_reweight="adaptive", flow_ratio=0.25, large_span=0) → 영향 없음.
- 실행: `./train.sh pi05_alphaflow_tabletop_paper 1 32 30000`.

### 스케줄: scaled-sigmoid로 phase 비율 정확화
- 기존 `alpha_schedule`은 `1-σ`를 floor에 **hard-clamp** → α가 floor에 **일찍 도달**해 평탄화 (transition window의 절반이 이미 floor).
- 수정: **scaled sigmoid** `α = 1 + (end_val-1)·σ(γ·progress)` → floor를 **transition_end에서 정확히** 도달. 이제 phase가 step 비율과 1:1.
  - FM (α=1)        : `[0, warmup_ratio)`
  - transition (1→α_min): `[warmup_ratio, transition_ratio)`
  - floor (α=α_min) : `[transition_ratio, 1.0]`
- 논문 매핑: epochs 5/100 of 150 → FM 3.3%(우린 5%), transition ~62%, floor 33%. γ=k=15.
- adaptive loss는 논문 그대로: **FM=Eq.14(γ=0) `1/(m+ε)·m`** (=원본 AlphaFlow p=1), **MF=Eq.18 `κ/(m+ακ+ε)·m`** (bounded). 둘 다 일치.

## 실험 config (한 번에 하나씩 isolate)
| config | H | 변경 | 상태 |
|---|---|---|---|
| `pi05_alphaflow_tabletop_rl_orig` | H1 (r_mlp) | 기본 (minmax/sphere/jvp) | warmup OK, **JVP서 발산** |
| `pi05_alphaflow_tabletop_rl_orig_nojvp` | **H5** | **use_jvp=False** (discrete-only, alpha_min 기본 5e-3) | **대기 (유력 후보)** |
| `pi05_alphaflow_tabletop_rl_orig_beta` | H2 | beta time | 중단 (O2: 효과 없음) |
| `pi05_alphaflow_tabletop_rl_orig_gaussian` | H3 | gaussian | 중단 (O2: sphere와 동일) |
| `pi05_alphaflow_tabletop_mf_from_start` | **A1** | curriculum 제거 (warmup=0,transition=0), discrete-only | 대기 (ablation) |
| `pi05_alphaflow_tabletop_mf_from_start_jvp` | **A1** | curriculum 제거, JVP | 대기 (ablation, 발산 예상) |

실행: `./train.sh <config> 1 32 30000`. 비교 지표: `loss/l2_raw`(pi05 FM scale), 학습 후 1-NFE vs 10-NFE.

## A1. Ablation — 처음부터 MeanFlow (curriculum 제거)
- `warmup_ratio=0, transition_ratio=0` → `alpha_schedule`의 degenerate-curriculum 경로 (`transition_end <= warmup_end`) → **alpha가 step 0부터 end_val로 상수.** warmup(TFM)→transition 커리큘럼을 건너뛰고 mean-velocity를 pretrained init에 직접 학습.
- **discrete-only** 변종(`mf_from_start`): alpha == alpha_min(5e-3) 상수 → 안정적 부트스트랩이 처음부터.
- **JVP** 변종(`mf_from_start_jvp`): alpha == 0 상수 → exact-derivative MeanFlow from scratch. O5 메커니즘상 가장 불안정할 것으로 예상.
- 목적: **커리큘럼이 정말 필요한가?** 를 격리. discrete-only가 curriculum 유무와 무관하게 잘 되면 → 커리큘럼 불필요(구현 단순화). JVP가 from-start에서 더 빨리 터지면 → O5의 "pretrained dudt 미정규화" 가설 강화.
- 25% FM-border 샘플(r=t, weight 1.0)은 alpha와 무관하게 항상 존재 → from-start에서도 순수 FM 앵커링은 유지됨.
- **alpha의 두 역할 분리 (`mf_loss_weight`)**: alpha는 ① 이산화 지점 `s=alpha·r+(1-alpha)·t` ② MF loss 가중 `weight_scale=alpha`, 두 곳에 쓰임. from-start(discrete)에서 alpha=alpha_min 상수면 ②가 MF gradient를 영구히 0.005배로 눌러버림(FM-border는 1.0인데). curriculum이 없으니 ②는 무의미 → `mf_loss_weight=1.0`로 **loss 가중만 1.0 고정**, 이산화 alpha_min은 유지. `mf_from_start`는 이걸 적용. (`None`이면 기존 alpha 가중 = 원본 커리큘럼.)
- **참고**: 커리큘럼 nojvp(`rl_orig_nojvp`)도 transition 이후엔 같은 0.005 가중 억제가 생김. 필요하면 거기도 `mf_loss_weight=1.0` A/B 가능.

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
> 초기 H1~H4 가설은 O5~O7로 대체됨. 현재 계획은 상단 "📌 현재 상태 종합 > 권장 실험 순서" 참고.
1. **`pi05_alphaflow_tabletop_paper` 학습** (O7 종합 레시피) — 1-NFE 개선 확인.
2. `…rl_orig_nojvp` 대조군 — paper의 어느 요소가 핵심인지 분리(bounded reweighting 단독 효과).
3. (선택) `…rl_orig`(fp32 JVP) — O6(bf16가 발산 원인) 확정.
- 과거 후보(H1b r_mlp_out 표준init / H4 target 재점검)는 필요 시 재소환.
