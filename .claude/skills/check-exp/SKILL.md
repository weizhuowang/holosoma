---
name: check-exp
description: Check status of running experiments — GPU utilization, rem jobs, and WandB metrics
---

# Experiment Status Check

Check the current state of all running and recent experiments.

## 1. REM Status (GPU + Jobs in one call)

```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py status
```

This single command shows:
- Summary: running / queued / recent_finished / total
- Available GPUs (no compute processes)
- Per-GPU: util / mem / process count
- Running jobs: id, GPU, elapsed, name
- Queued jobs: id, prepare_state, wait status, name
- Recently finished jobs: id, status, return code, elapsed, name

For machine-readable output (useful if you need to parse):
```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py status --json
```

To show more history:
```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py status --recent-minutes 240 --recent-limit 10
```

## 2. Running Job Logs

For any **running** jobs shown in step 1, fetch the last 20 lines of log:
```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py log {job_id} --tail 20
```

## 3. WandB Metrics (for running experiments)

Query the latest metrics from WandB for any active runs:

```bash
~/.holosoma_deps/miniconda3/envs/hssim/bin/python -c "
import wandb, os
os.environ['WANDB_BASE_URL'] = 'https://far.wandb.io'
api = wandb.Api()
for project in ['g1-manager-thermal', 'g1-manager-thermal-baseline']:
    runs = api.runs(project, filters={'state': 'running'}, per_page=10)
    for r in runs:
        h = r.history(keys=['Train/mean_reward','Train/mean_episode_length','Env/max_winding_temp','Perf/total_fps','_step'], samples=1)
        if h.empty:
            continue
        row = h.iloc[-1]
        step = int(row.get('_step', 0))
        reward = row.get('Train/mean_reward', '?')
        ep_len = row.get('Train/mean_episode_length', '?')
        temp = row.get('Env/max_winding_temp', '-')
        fps = row.get('Perf/total_fps', '?')
        print(f'[{project}] {r.name} (step {step}): reward={reward:.2f}  ep_len={ep_len:.1f}  max_temp={temp}  fps={fps:.0f}')
"
```

## 4. Present Summary

Combine REM status + WandB metrics into a concise overview. Flag any issues:
- Jobs in `failed` state
- GPU at >95% memory utilization
- Reward plateauing or dropping
- Abnormally high temperatures (GPU or winding)

## Tip

For continuous monitoring, combine with the loop skill:
```
/loop 10m /check-exp
```
