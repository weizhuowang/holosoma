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

## GPU Constraint

Only **1-GPU** or **2-GPU** experiments are supported. Do NOT allocate 3+ GPUs to a single experiment.
This machine has 8 GPUs — use them for parallelism (more experiments), not wider single experiments.

## Default num-envs

- **2 GPU (torchrun)**: `--training.num-envs 16384`
- **1 GPU**: `--training.num-envs 10240`

More envs does NOT improve speed. Use these defaults unless the user explicitly overrides.

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

## Step 2: Check Resources (health check + GPU + jobs in one call)

```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py status --json
```

If this fails, rem server is not running. Tell the user:
> rem server is not running. Start it with:
> `cd ~/Documents/gits/remote_exp_manager && python3 server.py &`

From the JSON response, extract:
- `available_gpu_indices`: list of free GPU indices (no compute processes) — use these for allocation
- `summary.running` / `summary.queued`: how busy the system is
- `running_jobs[].gpu_indices`: which GPUs are occupied and by what
- `gpus[].utilization_gpu_pct` / `memory_used_mb`: per-GPU health

Master ports are deterministic (29501 + min GPU id), so no port conflicts as long as GPU groups don't overlap.

## Step 4: Plan Allocation

Based on user request + available resources:
- Assign free GPUs to each experiment
- For torchrun multi-GPU: assign N consecutive free GPUs per experiment
- Compute master_port per the rule above: `29501 + min(gpu_ids)`
- Single-GPU: no torchrun, no master_port
- If not enough free GPUs: **use `--wait-for-job` to queue behind running jobs on the same GPU(s)**.
  Ask user whether to queue or skip. Don't just error out.

**Present the allocation table and confirm with the user before launching.**

Example (single-GPU, all GPUs free):
| # | Name | Config | GPU | Seed | Wait |
|---|------|--------|-----|------|------|
| 1 | thermal_s1 | exp:g1-29dof-thermal | 0 | 1 | - |
| 2 | thermal_s2 | exp:g1-29dof-thermal | 1 | 2 | - |

Example (multi-GPU, 2 GPUs each):
| # | Name | Config | GPUs | Port | Seed | Wait |
|---|------|--------|------|------|------|------|
| 1 | thermal_s1 | exp:g1-29dof-thermal | 0,1 | 29501 | 1 | - |
| 2 | thermal_s2 | exp:g1-29dof-thermal | 2,3 | 29503 | 2 | - |

Example (queuing — GPU 0 is busy with job 260407_101530_0):
| # | Name | Config | GPU | Seed | Wait |
|---|------|--------|-----|------|------|
| 1 | thermal_s1 | exp:g1-29dof-thermal | 0 | 1 | wait-for-job 260407_101530_0 |
| 2 | thermal_s2 | exp:g1-29dof-thermal | 1 | 2 | - |

## Step 5: Launch

For each experiment, spawn a **background** general-purpose Agent with `run_in_background: true`.
**Spawn all agents in a single message** (parallel).

Use this prompt template for each agent (fill in the `{placeholders}`):

~~~
You are monitoring an RL training experiment submitted via remote_exp_manager (rem).

## Your Task
1. Submit the experiment
2. Monitor until completion
3. Return a summary when done

## Submit Command
```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py submit \
  --name "{name}" \
  --cwd /home/ubuntu/Documents/gits/holosoma \
  --conda-env ~/.holosoma_deps/miniconda3/envs/hssim \
  --env CUDA_VISIBLE_DEVICES={gpu_ids} \
  {--wait-for-job JOB_ID, if queuing behind another job, otherwise omit this line} \
  --tmux --snapshot \
  --command "source scripts/source_isaacsim_setup.sh && {training_command}"
```

Parse the job_id from output: `submitted: id=XXXXXX_XXXXXX_X ...`
If submit fails, retry ONCE. If it fails again, return the error — do NOT keep retrying.

## Monitor
Wait 30s, then poll using `remexp_cli.py status --json`. Run with timeout 600000 (10 min max).
Script polls every 30s internally — failure detected fast, agent only does 1 tool call per cycle.
```bash
python3 -c "
import json, time, subprocess, os
JOB_ID = 'REPLACE_WITH_ACTUAL_ID'
CLI = os.path.expanduser('~/Documents/gits/remote_exp_manager/remexp_cli.py')
consecutive_errors = 0
while True:
    try:
        out = subprocess.check_output(['python3', CLI, 'status', '--json'], text=True)
        data = json.loads(out)
        consecutive_errors = 0
        for j in data.get('recent_jobs', []):
            if j['id'] == JOB_ID:
                print(f'[done] status={j[\"status\"]} rc={j.get(\"return_code\")} elapsed={j.get(\"elapsed\")}')
                print(json.dumps(j, indent=2)); exit(0)
        found = False
        for j in data.get('running_jobs', []):
            if j['id'] == JOB_ID: found = True; break
        if not found:
            for j in data.get('queued_jobs', []):
                if j['id'] == JOB_ID: found = True; break
        if not found:
            print(f'[poll] job not found, may have finished earlier'); exit(0)
    except Exception as e:
        consecutive_errors += 1
        print(f'[poll] error ({consecutive_errors}/3): {e}', flush=True)
        if consecutive_errors >= 3:
            print('FATAL: 3 consecutive poll errors, giving up'); exit(1)
    time.sleep(30)
"
```
If this times out, re-run it. After 2 consecutive timeouts where job is not found, stop.

## When Done
Fetch the last 100 lines of log:
```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py log {job_id} --tail 100
```

Return raw status only: job_id, name, status, return code, elapsed, log tail. Do NOT analyze results.
~~~

### Training command templates

**Single-GPU** (default 10240 envs):
```
python src/holosoma/holosoma/train_agent.py {exp_config} simulator:isaacsim logger:wandb --training.seed {seed} --training.num-envs 10240 --logger.wandb.notes "{name}" {extra_overrides}
```

**Multi-GPU / 2-GPU** (default 16384 envs):
```
torchrun --nproc_per_node=2 --master_port={port} src/holosoma/holosoma/train_agent.py {exp_config} simulator:isaacsim logger:wandb --training.num-envs 16384 --logger.wandb.notes "{name}" {extra_overrides}
```

`{name}` is the same name used in `--name` for rem submit, so rem job and wandb run can be cross-referenced.

## Step 6: Log Launch

**IMPORTANT: You MUST do this.** Append a launch entry for each experiment to the experiment log:

File: `/home/ubuntu/Documents/gits/holosoma/.claude/experiments_log.md`

For each experiment, append (using the Edit tool):
```
### [LAUNCHED] {name} — {YYYY-MM-DD}
- **Job ID**: {job_id}
- **Config**: {exp_config}
- **GPU**: {gpu_ids} ({n_gpus} GPU)
- **Seed**: {seed}
- **num-envs**: {num_envs}
- **Overrides**: {extra_overrides or "none"}
- **Hypothesis/Goal**: {why this experiment is being run — ask user if unclear}
```

## Step 7: Report

After spawning and logging, tell the user:
- Number of experiments launched
- Their names, GPU assignments, seeds
- "Each experiment runs in a background agent. You will be notified as each one completes."

When a background agent returns (experiment finished), relay its summary to the user immediately.

## Step 8: Log Completion

**IMPORTANT: You MUST do this.** When a background agent returns with results, append a
completion entry directly below the corresponding launch entry in the experiment log:

```
**[DONE]** {YYYY-MM-DD} | status={status} | rc={return_code} | elapsed={elapsed}
- **Final reward**: {value}
- **Final ep length**: {value}
- **Max winding temp**: {value} (if thermal)
- **Takeaway**: {one-line summary of what we learned — ask user if unsure}
```

## Step 9: Post-Experiment Report (optional)

When an experiment completes, if the user wants deeper analysis, spawn a `report-exp` agent
(general-purpose, background) with this prompt:
~~~
You are generating an experiment report. Read the instructions in
/home/ubuntu/Documents/gits/holosoma/.claude/agents/report-exp.md and follow them.
The rem job_id is: {job_id}
~~~
The report agent reads local TensorBoard events from the snapshot, generates plots, and
returns a compact summary with image paths. Read the images to understand the training curves.
