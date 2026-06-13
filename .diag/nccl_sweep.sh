#!/bin/bash
# Run the 2-GPU collective reproducer under several NCCL transport settings.
# Each test is timeout-bounded: EXIT=124 => hang. Run on a 2-GPU allocation.
echo "=== node $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== nvidia-smi topo (P2P/NVLink matrix) ==="
nvidia-smi topo -m 2>&1 | head -12

run() {
  name="$1"; shift
  echo ""; echo "########## TEST: $name  (flags: $*) ##########"
  timeout 90 env "$@" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
      NCCL_DEBUG=WARN \
      uv run python .diag/collective_test.py > /tmp/ct.$$ 2>&1
  rc=$?
  tail -8 /tmp/ct.$$
  if grep -q COLLECTIVE_OK /tmp/ct.$$; then echo ">>> [$name] PASS"; else echo ">>> [$name] FAIL rc=$rc"; fi
}

run baseline
run p2p_off       NCCL_P2P_DISABLE=1
run p2p_shm_off   NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
run cumem_off     NCCL_CUMEM_ENABLE=0
run p2p_cumem_off NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0
echo ""; echo "=== SWEEP DONE ==="
