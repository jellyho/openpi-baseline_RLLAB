"""Minimal 2-GPU JAX collective reproducer: builds the SAME dp-mesh all-reduce the
critic train step uses, and times the cross-device clique init. Hangs => NCCL transport
problem; prints COLLECTIVE_OK on success."""
import time, numpy as np, jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devs = jax.devices()
n = len(devs)
print(f"devices: {n} x {devs[0].device_kind}", flush=True)
mesh = Mesh(np.array(devs), ("dp",))
sh = NamedSharding(mesh, P("dp"))
x = jax.device_put(np.ones((n * 8, 16), np.float32), sh)   # sharded over dp

@jax.jit
def f(x):
    return jnp.sum(x)                                       # reduction over dp-sharded axis -> all-reduce

t = time.time()
r = float(f(x))                                            # triggers NCCL clique init
print(f"result={r}  (expected {n*8*16})  in {time.time()-t:.1f}s", flush=True)
assert abs(r - n * 8 * 16) < 1e-3
print("COLLECTIVE_OK", flush=True)
