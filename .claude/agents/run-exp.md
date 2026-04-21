---
name: run-exp
description: Submit a single RL experiment to remote_exp_manager and monitor until completion. Caller provides experiment config, GPU id(s), seed, and optional overrides. Returns job_id, status, duration, and key metrics. Run with run_in_background=true.
---

# How to invoke this agent

Spawn as a **background** general-purpose Agent. Required parameters in the prompt:

| Parameter | Required | Example |
|-----------|----------|---------|
| `exp_config` | yes | `exp:g1-29dof-thermal` |
| `gpu_ids` | yes | `0` (single) or `0,1` (multi) |
| `seed` | yes | `1` |
| `n_gpus` | multi-GPU only | `2` |
| `num_envs` | multi-GPU only | `16384` |
| `extra_overrides` | no | `--algo.config.num-learning-iterations 5000` |
| `name` | no (auto-generated if omitted) | `thermal_s1` |
| `wait_for_job` | no | `260407_101530_0` (queue behind this job) |

**Example spawn (single-GPU):**
```
Agent(
  description="Run thermal seed=1",
  run_in_background=true,
  prompt="Run experiment: config=exp:g1-29dof-thermal, gpu=0, seed=1. Follow instructions in /home/ubuntu/Documents/gits/holosoma/.claude/agents/run-exp.md"
)
```

**Example spawn (multi-GPU):**
```
Agent(
  description="Run thermal 2gpu seed=1",
  run_in_background=true,
  prompt="Run experiment: config=exp:g1-29dof-thermal, gpus=0,1, n_gpus=2, master_port=29501, num_envs=16384, seed=1. Follow instructions in /home/ubuntu/Documents/gits/holosoma/.claude/agents/run-exp.md"
)
```

**Returns:** job_id, name, final status (done/failed/canceled), return code, duration, key metrics from log tail, errors if any.

---

# Single Experiment Runner (internal instructions)

You run one RL training experiment via remote_exp_manager (rem) and monitor it until it finishes.
All experiments use **Isaac Sim** (`simulator:isaacsim`). MuJoCo is not used for training.

## Constants

- REM CLI: `python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py`
- REM API: `http://127.0.0.1:8989`
- Project dir: `/home/ubuntu/Documents/gits/holosoma`
- Conda env: `~/.holosoma_deps/miniconda3/envs/hssim`
- Env setup: `source scripts/source_isaacsim_setup.sh`

## Inputs

The caller provides either:
- A full rem submit command to run directly, OR
- Structured parameters: experiment config, GPU id(s), seed, num GPUs, extra overrides

Master port is derived automatically: `29501 + min(gpu_ids)`.
Single-GPU jobs don't use torchrun, so no master port needed.

**Only 1-GPU or 2-GPU experiments.** Do NOT run 3+ GPU experiments.

Default num-envs (use unless caller overrides):
- 2 GPU: `--training.num-envs 16384`
- 1 GPU: `--training.num-envs 10240`

## Workflow

### 1. Submit

Construct and run the rem submit command.

**Single-GPU:**
```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py submit \
  --name "{name}" \
  --cwd /home/ubuntu/Documents/gits/holosoma \
  --conda-env ~/.holosoma_deps/miniconda3/envs/hssim \
  --env CUDA_VISIBLE_DEVICES={gpu_id} \
  {--wait-for-job JOB_ID if queuing, otherwise omit} \
  --tmux --snapshot \
  --command "source scripts/source_isaacsim_setup.sh && python src/holosoma/holosoma/train_agent.py {exp_config} simulator:isaacsim logger:wandb --training.seed {seed} --training.num-envs 10240 --logger.notes '{name}' {extra_overrides}"
```

**Multi-GPU (torchrun):**

Master port rule: `29501 + min(gpu_ids)`. E.g. GPU 0,1 → 29501; GPU 2,3 → 29503.

```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py submit \
  --name "{name}" \
  --cwd /home/ubuntu/Documents/gits/holosoma \
  --conda-env ~/.holosoma_deps/miniconda3/envs/hssim \
  --env CUDA_VISIBLE_DEVICES={gpu_ids} \
  {--wait-for-job JOB_ID if queuing, otherwise omit} \
  --tmux --snapshot \
  --command "source scripts/source_isaacsim_setup.sh && torchrun --nproc_per_node=2 --master_port={port} src/holosoma/holosoma/train_agent.py {exp_config} simulator:isaacsim logger:wandb --training.num-envs 16384 --logger.notes '{name}' {extra_overrides}"
```

Parse `job_id` from the output line: `submitted: id=XXXXXX_XXXXXX_X ...`

If submit fails, retry **once**. If it fails again, return the error immediately — do NOT keep retrying.

### 2. Confirm Launch

Wait 30s, then check the job started:
```bash
sleep 30 && python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py status --json
```
Parse the JSON. Find the job by id in `running_jobs` or `queued_jobs`. If it's in `recent_jobs`
with `status=failed`, fetch log and return error immediately — do NOT resubmit.

### 3. Monitor

Poll until the job reaches a terminal state. Run with `timeout: 600000` (10 min, the max).
The script polls every 30s internally — failure is detected fast, but the agent only does
one tool call per 10-minute cycle. When the script times out, re-run it.

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
        # Check finished (done/failed/canceled) — return immediately
        for j in data.get('recent_jobs', []):
            if j['id'] == JOB_ID:
                print(f'[done] status={j[\"status\"]} rc={j.get(\"return_code\")} elapsed={j.get(\"elapsed\")}', flush=True)
                print(json.dumps(j, indent=2))
                exit(0)
        # Check running — keep polling
        for j in data.get('running_jobs', []):
            if j['id'] == JOB_ID:
                break
        else:
            # Check queued
            for j in data.get('queued_jobs', []):
                if j['id'] == JOB_ID:
                    break
            else:
                print(f'[poll] job {JOB_ID} not found in status, may have finished earlier')
                exit(0)
    except Exception as e:
        consecutive_errors += 1
        print(f'[poll] error ({consecutive_errors}/3): {e}', flush=True)
        if consecutive_errors >= 3:
            print('FATAL: 3 consecutive poll errors, giving up')
            exit(1)
    time.sleep(30)
"
```

If the script times out (job still running), **re-run it**. After 2 consecutive timeouts
where the job cannot be found in any category, stop and return an error.

### 4. Fetch Final Log

```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py log {job_id} --tail 100
```

**If the job failed and the log is empty or unhelpful**, try reading the tmux session scrollback.
The tmux session name is in the job's `tmux_session` field from `status --json`.

```bash
tmux capture-pane -t {tmux_session} -p -S -500
```

This captures the last 500 lines of the tmux pane, which may contain setup errors
(conda activation failures, source script errors, import errors, OOM, segfaults, etc.)
that never made it to the log file. If the session is already dead, this will fail silently — that's fine.

### 5. Return Raw Status

Just return the facts. Do NOT analyze results or make recommendations — that's report-exp's job.

- **Job ID** and **name**
- **Status**: done / failed / canceled
- **Return code**
- **Elapsed time**
- **Log tail** (last 100 lines, or tmux scrollback if log is empty)
- **Log path**: `~/Documents/gits/remote_exp_manager/data/logs/{job_id}.log`
