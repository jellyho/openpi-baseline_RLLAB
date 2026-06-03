"""Build the frame cache for a repo, verify preloaded samples match real video
decode, and compare per-sample latency. Run inside the v3.0 .venv on a node with
torchcodec + enough /dev/shm."""
import random
import sys
import time

import openpi.training.preload_dataset as P
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo = sys.argv[1] if len(sys.argv) > 1 else "jellyho/aloha_handover_box_joint_pos_rl_224"

t = time.time()
P.build_frame_cache(repo)
print("[build] cache built in %.1fs" % (time.time() - t), flush=True)

meta = LeRobotDataset(repo).meta
fps = meta.fps
img_keys = [k for k in meta.features if "image" in k]
offs = [0, 5, 10, 25, 50]
dts = {k: [o / fps for o in offs] for k in img_keys + ["observation.state"]}
dts["action"] = [i / fps for i in range(50)]

real = LeRobotDataset(repo, delta_timestamps=dts)
pre = P.PreloadedLeRobotDataset(repo, delta_timestamps=dts)

random.seed(1)
N = len(real)
for i in [random.randrange(N) for _ in range(4)]:
    r, p = real[i], pre[i]
    for k in img_keys:
        assert r[k].shape == p[k].shape, (k, r[k].shape, p[k].shape)
    diff = (r[img_keys[0]] - p[img_keys[0]]).abs().mean().item()
    cam0 = img_keys[0].split(".")[-1]
    print("  idx %d: shapes ok, mean|diff| %s = %.4f" % (i, cam0, diff), flush=True)

idxs = [random.randrange(N) for _ in range(40)]
real[idxs[0]]; pre[idxs[0]]
t = time.time()
for i in idxs:
    real[i]
tr = (time.time() - t) / len(idxs)
t = time.time()
for i in idxs:
    pre[i]
tp = (time.time() - t) / len(idxs)
print("[speed] real-decode %.1f ms/sample | preloaded %.1f ms/sample | speedup %.1fx"
      % (tr * 1000, tp * 1000, tr / tp), flush=True)
