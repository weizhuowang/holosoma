---
name: check-exp
description: Check status of running experiments — GPU utilization, rem jobs, and WandB metrics
---

# Experiment Status Check

Check the current state of all running and recent experiments. Run all three checks in parallel.

## 1. GPU Status

```bash
curl -s http://127.0.0.1:8989/api/gpus | python3 -c "
import json, sys
data = json.load(sys.stdin)
for g in data['gpus']:
    p = g.get('processes', [])
    s = 'BUSY' if p else 'FREE'
    print(f'GPU {g[\"index\"]}: {s} | Util {g[\"utilization_gpu_pct\"]}% | Mem {g[\"memory_used_mb\"]}/{g[\"memory_total_mb\"]}MB | Temp {g[\"temperature_c\"]}C | Power {g[\"power_w\"]}W')
    for proc in p:
        print(f'  PID {proc[\"pid\"]}: {proc[\"used_memory_mb\"]}MB')
"
```

## 2. REM Job Status

```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py list
```

For any **running** jobs, also fetch the last 20 lines of their log:
```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py log {job_id} --tail 20
```

## 3. WandB Metrics (for running experiments)

Query the latest metrics from WandB for any active runs. Check both thermal projects:

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

Combine all three into a concise status table:

| GPU | Util | Mem | Job | Status | Step | Reward | Ep Len |
|-----|------|-----|-----|--------|------|--------|--------|
| ... | ...  | ... | ... | ...    | ...  | ...    | ...    |

Flag any issues:
- GPU at >95% memory utilization
- Jobs in `failed` state
- Reward plateauing or dropping (if enough history visible)
- Abnormally high temperatures (GPU or winding)

## Tip

For continuous monitoring, combine with the loop skill:
```
/loop 10m /check-exp
```
This will auto-check every 10 minutes.
