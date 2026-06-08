#!/usr/bin/env python3
"""Submit an AQC hyperparameter sweep as a single SLURM array job.

Pattern mirrors the project's cfg.json + submit.py + sbatch.sh convention, generalized
to a multi-dimensional sweep: this script expands (env x critic_type x num_atoms x H x seed)
into a flat job manifest (one `main.py` arg-line per row), then submits one array job
(`--array=0-N%max_gpus`). `sbatch_aqc_sweep.sh` reads its row from the manifest by
`SLURM_ARRAY_TASK_ID` and runs it.

Usage:
    python submit_aqc_sweep.py --dry-run          # print the grid, do not submit
    python submit_aqc_sweep.py --max-gpus 16      # submit, capping concurrent tasks
"""
import argparse
import json
import os
import subprocess
from datetime import datetime


def build_jobs(cfg, ts):
    fixed, sweep, envs = cfg["fixed"], cfg["sweep"], cfg["envs"]
    tag = fixed.get("tag", "")          # optional label (e.g. 'sparse'/'dense') for run_group
    prefix = f"aqc_sweep_{tag + '_' if tag else ''}{ts}"
    default_support = fixed.get("support_type", "data")  # 'data' (DEAS-d) / 'universal' / 'custom'
    jobs = []
    for env, env_cfg in envs.items():
        for critic_type in sweep["critic_type"]:
            # num_atoms & support_type only matter for the distributional critic.
            if critic_type == "distributional":
                atoms_list = sweep["num_atoms"]
                support_list = sweep.get("support_type", [default_support])
            else:
                atoms_list = [None]
                support_list = [default_support]   # passed through, unused by regression
            for num_atoms in atoms_list:
                for support in support_list:
                    for H in sweep["horizon_length"]:
                        for seed in sweep["seed"]:
                            rg = f"{prefix}_{critic_type}"
                            if critic_type == "distributional":
                                rg += f"_a{num_atoms}_sup-{support}"
                            rg += f"_H{H}"
                            a = [
                                f"--agent={fixed['agent']}",
                                f"--env_name={env}",
                                f"--seed={seed}",
                                f"--run_group={rg}",
                                f"--horizon_length={H}",
                                f"--sparse={str(env_cfg['sparse'])}",
                                f"--agent.critic_type={critic_type}",
                                f"--agent.use_target_critic={str(fixed['use_target_critic'])}",
                                f"--agent.num_action_samples={fixed['num_action_samples']}",
                                # set both the dataset discount and the agent's bootstrap discount.
                                f"--discount={fixed['discount']}",
                                f"--agent.discount={fixed['discount']}",
                                f"--offline_steps={fixed['offline_steps']}",
                                f"--online_steps={fixed['online_steps']}",
                                f"--eval_interval={fixed['eval_interval']}",
                                f"--eval_episodes={fixed['eval_episodes']}",
                                f"--video_episodes={fixed['video_episodes']}",
                                f"--save_interval={fixed['save_interval']}",
                                f"--wandb_entity={fixed['wandb_entity']}",
                                f"--wandb_project={fixed['wandb_project']}",
                                f"--support_type={support}",
                            ]
                            if critic_type == "distributional":
                                a.append(f"--agent.num_atoms={num_atoms}")
                                if support == "custom":
                                    a += [f"--agent.v_min={env_cfg['v_min']}",
                                          f"--agent.v_max={env_cfg['v_max']}"]
                            jobs.append(" ".join(a))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-gpus", type=int, default=16, help="max concurrent array tasks")
    ap.add_argument("--cfg", default="cfg_aqc_sweep.json")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, args.cfg)))
    sbatch_script = os.path.join(here, "sbatch_aqc_sweep.sh")

    ts = datetime.now().strftime("%m%d_%H%M%S")
    cfg_stem = os.path.splitext(os.path.basename(args.cfg))[0]
    jobs = build_jobs(cfg, ts)
    n = len(jobs)

    # Include the cfg name + seconds so concurrent submissions never share a manifest
    # (the array reads this file at run time -- a shared name = silent job overwrite).
    manifest = os.path.join(here, f"jobs_{cfg_stem}_{ts}.txt")
    with open(manifest, "w") as f:
        f.write("\n".join(jobs) + "\n")

    os.makedirs(os.path.join(here, "logs"), exist_ok=True)
    cmd = [
        "sbatch", "--parsable",
        f"--array=0-{n - 1}%{args.max_gpus}",
        f"--job-name=aqc_sweep_{ts}",
        f"--output={here}/logs/%A_%a.out",
        f"--error={here}/logs/%A_%a.err",
        sbatch_script, manifest,
    ]

    print(f"Generated {n} jobs -> {manifest}")
    print("sbatch:", " ".join(cmd))
    if args.dry_run:
        print(f"\n[DRY RUN] sample jobs (showing 4 of {n}):")
        for j in jobs[:4]:
            print("  python main.py", j)
        return
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("submitted array job:", r.stdout.strip())
    if r.stderr.strip():
        print("stderr:", r.stderr.strip())


if __name__ == "__main__":
    main()
