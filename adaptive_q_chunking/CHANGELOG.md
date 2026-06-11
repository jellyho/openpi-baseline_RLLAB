# VLA critic learning — change log

코드 변경 이력 기록. 과학적 세팅(타깃, 보상, discount 등)을 건드리는 변경과
순수 엔지니어링(속도/인프라) 변경을 구분해서 적는다.
OGBench 쪽 default는 별도 관리: `~/projects/rss_ptf/adaptive_q_chunking/qc/scripts/aqc/sweep/ogbench_defaults.json`
(OGBench discount=0.99 / VLA discount=0.999 — 서로 다른 config가 맞음).

---

## 2026-06-11 — multi-GPU data parallelism + multiprocess loader (속도만, method 불변)

**왜**: 학습이 GPU가 아니라 host 쪽 데이터 조립에 묶여 있었음 (B=256 기준 loader
~217ms/batch vs GPU step ~135ms). batch를 키우거나 GPU를 늘려도 loader가 1개면
samples/sec이 그대로라, openpi와 같은 구조(멀티프로세스 로더 + mesh sharding)를 도입.

**과학적 변경 없음**: 타깃 계산, loss, 보상 relabel, discount, prefix grid, candidate 수
전부 그대로. critic은 여전히 순수 JAX/flax (torch는 데이터 로딩 워커에만 사용).

### 바뀐 파일

| 파일 | 변경 |
|---|---|
| `vla_loader.py` | **신규.** torch `IterableDataset` + `DataLoader` 래퍼. 워커 프로세스 N개가 row-group work-list를 분할(`shard=(wid, nw)`) 받아 각자 기존 generator를 돌리고 **완성된 global batch**를 yield. jax import 없음 (워커는 jax-free). |
| `vla_data.py` | `iter_batches` / `iter_bootstrap_batches`에 `shard=(i, n)` 인자 추가 — 셔플된 row-group 리스트를 i::n 으로 분할. `(0, 1)` 기본값이면 기존과 동일 동작. |
| `vla_train.py` | (1) data-parallel mesh: params/opt_state는 전 GPU에 replicate, batch는 `jax.make_array_from_process_local_data`로 leading axis 분할 → grad all-reduce는 jit이 자동 삽입. GPU 1장이면 identity (기존과 동일). (2) `batch_iter`가 `loader_processes>0`이면 torch 멀티프로세스 로더 사용, 0이면 기존 thread prefetch. (3) eval 시 `jax.device_get(params)`로 host 복사 후 사용 (mesh-replicated params와 single-device eval 입력의 device 충돌 회피). |
| `vla_config.py` | `loader_processes: int = 4` 추가 (0 = 기존 thread 로더). `num_workers`는 이제 "로더 프로세스당 parquet read thread 수" 의미. run_name에는 영향 없음 (순수 throughput knob). |

### 동작상 주의점 (결과에 영향 없는 수준이지만 기록)

- **배치 구성 순서가 기존 run과 다름**: 각 batch가 한 워커의 row-group shard에서
  나옴 (기존: 전체에서 8개 group pool). 셔플 통계는 동일 수준이지만 같은 seed로
  기존 run과 step-by-step 재현은 안 됨.
- 메모리: 워커당 row-group pool + prefetch로 **프로세스당 ~2-3GB RAM**. SLURM
  `--mem`은 `64G+`, `--cpus-per-task >= loader_processes x num_workers/2` 권장
  (예: 4 proc x 8 thread → 16-32 cores).
- multi-GPU 노드에서 batch_size는 GPU 수로 나누어떨어져야 함 (assert 있음).
- 기존 단일 GPU + 기존 로더로 완전히 되돌리려면: `--loader_processes 0`.

### 사용법

```bash
# A100 4장 노드, global batch 1024 (GPU당 256 = 기존과 동일), 로더 4프로세스
python vla_train.py --config vla_aqc_td_macro --batch_size 1024 --loader_processes 4
# lr은 batch에 맞춰 별도 결정 (--lr). 기존 b256 세팅 그대로면 batch_size 생략.
# 속도 probe: --timing_steps 100
```

---

## (기준점) 2026-06-11 이전 상태

- `vla_config.py`의 named preset registry (vla_aqc_td_macro 등), run_name 규약,
  checkpoint/resume, W&B(RSS-PFT_RLLAB/rlt_critic_learning), eval value-curves.
- 데이터: in-loader relabel은 비활성 (dataset이 디스크에서 재annotate됨: living -4e-4,
  fail -0.5, mc_return gamma=0.999).
