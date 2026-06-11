"""RLT/AQC critic learning, absorbed into openpi (see rlt_critic/README).

Lightweight ACSAC prefix-critic trained on precomputed RLT latents
(rl_token + base_action) from an annotated LeRobot dataset. Framework-light
(flax.linen + numpy/pyarrow loader); kept import-side-effect-free so torch
worker processes can import .data/.loader without pulling JAX."""
