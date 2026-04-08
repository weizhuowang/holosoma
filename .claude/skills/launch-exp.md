---
name: launch-exp
description: Coordinate launching RL training experiments via remote_exp_manager
---

# Launch Experiments Coordinator

Follow this SOP to launch one or more RL training experiments.

## Key Facts

- All experiments use **Isaac Sim** (`simulator:isaacsim`). MuJoCo is not used for training.
- Env setup script: `source scripts/source_isaacsim_setup.sh` (prepend to training command)

## Constants

| Key | Value |
|-----|-------|
| REM CLI | `python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py` |
| REM API | `http://127.0.0.1:8989` |
| Project dir | `/home/ubuntu/Documents/gits/holosoma` |
| Conda env | `~/.holosoma_deps/miniconda3/envs/hssim` |
| Env setup | `source scripts/source_isaacsim_setup.sh` |

## Master Port Rule (torchrun only)

`master_port = 29501 + min(gpu_ids)` for each multi-GPU experiment.

Examples:
- GPU 0,1 → port 29501 (min=0, 29501+0)
- GPU 1,2 → port 29502 (min=1, 29501+1)
- GPU 2,3 → port 29503 (min=2, 29501+2)

This guarantees no port conflicts as long as GPU groups don't overlap.
Single-GPU experiments don't use torchrun, so no master port.

## Step 1: Parse User Request

Understand what the user wants:
- Experiment config(s): e.g. `exp:g1-29dof-thermal`, `exp:g1-29dof`
- Number of runs / seeds
- Single-GPU or multi-GPU (torchrun)?
- Any CLI overrides (iterations, num-envs, etc.)
- Any WandB project override

## Step 2: Health Check

Verify rem server is running:
```bash
curl -sf http://127.0.0.1:8989/api/gpus > /dev/null && echo "REM server OK" || echo "ERROR: rem server not running at :8989"
```

If not running, tell the user:
> rem server is not running. Start it with:
> `cd ~/Documents/gits/remote_exp_manager && python3 server.py &`

## Step 3: Check Resources

GPU availability:
```bash
curl -s http://127.0.0.1:8989/api/gpus | python3 -c "
import json, sys
data = json.load(sys.stdin)
for g in data['gpus']:
    p = g.get('processes', [])
    s = 'BUSY' if p else 'FREE'
    print(f'GPU {g[\"index\"]}: {s} | Mem {g[\"memory_used_mb\"]}/{g[\"memory_total_mb\"]}MB | Util {g[\"utilization_gpu_pct\"]}%')
    for proc in p:
        print(f'  PID {proc[\"pid\"]}: {proc[\"used_memory_mb\"]}MB')
"
```

Running jobs:
```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py list
```

Master ports are deterministic (29501 + min GPU id), so no need to check for port conflicts as long as GPU groups don't overlap.

## Step 4: Plan Allocation

Based on user request + available resources:
- Assign free GPUs to each experiment
- For torchrun multi-GPU: assign N consecutive free GPUs per experiment
- Compute master_port per the rule above: `29501 + min(gpu_ids)`
- Single-GPU: no torchrun, no master_port
- If not enough free GPUs: tell user and ask how to proceed (wait? queue? fewer experiments?)

**Present the allocation table and confirm with the user before launching.**

Example (single-GPU):
| # | Name | Config | GPU | Seed |
|---|------|--------|-----|------|
| 1 | thermal_s1 | exp:g1-29dof-thermal | 0 | 1 |
| 2 | thermal_s2 | exp:g1-29dof-thermal | 1 | 2 |

Example (multi-GPU, 2 GPUs each):
| # | Name | Config | GPUs | Port | Seed |
|---|------|--------|------|------|------|
| 1 | thermal_s1 | exp:g1-29dof-thermal | 0,1 | 29501 | 1 |
| 2 | thermal_s2 | exp:g1-29dof-thermal | 2,3 | 29503 | 2 |

## Step 5: Launch

For each experiment, spawn a **background** Agent (general-purpose) with the instructions from `.claude/agents/run-exp.md`.

Include in each agent's prompt:
1. The experiment name
2. Whether it's single-GPU or multi-GPU
3. The exact GPU id(s), master_port, seed
4. The full experiment config string
5. Any extra CLI overrides
6. Instruction: "Submit via rem, monitor until completion, return summary when done."

**Spawn all agents in a single message** (parallel) with `run_in_background: true`.

## Step 6: Report

After spawning, tell the user:
- Number of experiments launched
- Their names, GPU assignments, seeds
- "Each experiment runs in a background agent. You will be notified as each one completes."

When a background agent returns (experiment finished), relay its summary to the user immediately.
