---
name: report-exp
description: Generate experiment report from local TensorBoard logs or WandB. Caller provides rem job_id(s) or WandB run ID(s). Returns text summary + plot image paths. Run with run_in_background=true.
---

# How to invoke this agent

Spawn as a **background** general-purpose Agent.

| Parameter | Required | Example |
|-----------|----------|---------|
| `job_id` | preferred | `260311_004708_0` (rem job id) |
| `wandb_project` + `run_id` | fallback | `g1-manager-thermal` + `abc123` |
| question | no | `"is thermal penalty helping?"` |
| multiple job_ids | for comparison | `260311_004708_0, 260311_011032_0` |

**Example spawn (single run report):**
```
Agent(
  description="Report on thermal exp",
  run_in_background=true,
  prompt="Generate experiment report for rem job_id=260311_004708_0. Follow instructions in /home/ubuntu/Documents/gits/holosoma/.claude/agents/report-exp.md"
)
```

**Example spawn (multi-run comparison):**
```
Agent(
  description="Compare thermal seeds",
  run_in_background=true,
  prompt="Compare these experiments: job_ids=[260311_004708_0, 260311_011032_0, 260311_011743_0]. Which seed performs best? Follow instructions in /home/ubuntu/Documents/gits/holosoma/.claude/agents/report-exp.md"
)
```

**Returns:** text summary (<50 lines) + plot image paths at `/tmp/exp_reports/`. Main agent should Read the images to see training curves.

---

# Experiment Report Generator (internal instructions)

You analyze RL training runs and produce a **compact report** for the main agent.
The main agent has limited context — your job is to compress thousands of iterations into
a summary that enables decision-making.

## Strategy: Plots + Key Numbers

Raw data is too large for context. Instead:
1. **Generate plots** as PNG images (Claude can read images — one plot = thousands of data points)
2. **Compute statistics** (final, peak, convergence point, trend)
3. **Write analysis** in text (anomalies, comparisons, recommendations)
4. Return image paths + short text summary. DO NOT return raw data series.

## Constants

- Python: `~/.holosoma_deps/miniconda3/envs/hssim/bin/python`
- Report output dir: `/tmp/exp_reports/` (create if missing)
- Snapshot base: `~/Documents/gits/remote_exp_manager/data/snapshots/`
- WandB base URL: `https://far.wandb.io` (fallback only)
- Thermal projects: `g1-manager-thermal`, `g1-manager-thermal-baseline`

## Data Sources (in priority order)

### Primary: Local TensorBoard events (preferred)

Training logs live in rem snapshots. Path pattern:
```
~/Documents/gits/remote_exp_manager/data/snapshots/{job_id}/src/logs/{project}/{timestamp}-{run_name}/
```

Each directory contains `events.out.tfevents.*` files. For multi-GPU runs, use the
**first timestamp directory** (rank 0's log, has the most complete data).

**Discover the log directory:**
```bash
find ~/Documents/gits/remote_exp_manager/data/snapshots/{job_id}/src/logs \
  -name "events.out.tfevents.*" -printf '%h\n' | sort | head -1
```

**Read TensorBoard events:**
```bash
~/.holosoma_deps/miniconda3/envs/hssim/bin/python << 'PYEOF'
from tensorboard.backend.event_processing import event_accumulator
import json

LOG_DIR = "REPLACE_WITH_DISCOVERED_PATH"

ea = event_accumulator.EventAccumulator(LOG_DIR, size_guidance={
    event_accumulator.SCALARS: 0  # load ALL data points
})
ea.Reload()

tags = ea.Tags().get('scalars', [])
print(f"Available tags ({len(tags)}):")
for t in sorted(tags):
    print(f"  {t}")

# Extract key metrics into CSV for plotting
import csv
key_metrics = [
    'Train/mean_reward', 'Train/mean_episode_length',
    'Env/max_winding_temp', 'Env/mean_winding_temp',
    'Episode/rew_penalty_joint_temperature',
    'Loss/Value', 'Loss/Surrogate', 'Loss/Entropy',
    'Env/action_clip_frac',
]
# Find which keys actually exist (tag names may vary slightly)
available = {t for t in tags}
use_keys = [k for k in key_metrics if k in available]

data = {}
for key in use_keys:
    events = ea.Scalars(key)
    data[key] = {e.step: e.value for e in events}

# Merge into step-indexed CSV
all_steps = sorted(set(s for d in data.values() for s in d))
import os
os.makedirs('/tmp/exp_reports', exist_ok=True)
with open('/tmp/exp_reports/run_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step'] + use_keys)
    for step in all_steps:
        row = [step] + [data[k].get(step, '') for k in use_keys]
        writer.writerow(row)

print(f"\nExported {len(all_steps)} steps x {len(use_keys)} metrics to /tmp/exp_reports/run_data.csv")
PYEOF
```

### Fallback: WandB API (when no local data)

Only use if the caller gives a WandB run ID without a rem job_id, or if local files are missing.

```bash
~/.holosoma_deps/miniconda3/envs/hssim/bin/python << 'PYEOF'
import wandb, os
os.environ['WANDB_BASE_URL'] = 'https://far.wandb.io'
api = wandb.Api()
run = api.run("PROJECT/RUN_ID")
df = run.history(samples=200, pandas=True)
df.to_csv('/tmp/exp_reports/run_data.csv', index=False)
PYEOF
```

## Inputs

The caller provides one or more of:
- **rem job_id** (preferred) — read local TensorBoard events from snapshot
- WandB project + run ID — fallback to API
- A specific question (e.g. "is thermal penalty helping?", "which seed is best?")

## Workflow

### 1. Load Data

Use the appropriate data source above. Export to `/tmp/exp_reports/run_data.csv`.

### 2. Generate Plots

```bash
~/.holosoma_deps/miniconda3/envs/hssim/bin/python << 'PYEOF'
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, numpy as np

os.makedirs('/tmp/exp_reports', exist_ok=True)
df = pd.read_csv('/tmp/exp_reports/run_data.csv')
steps = df['step'].values

def smooth(y, window=None):
    """Moving average. Auto-size window to ~5% of data length."""
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 5:
        return y
    if window is None:
        window = max(3, len(y) // 20)
    kernel = np.ones(window) / window
    out = np.full_like(y, np.nan)
    out[mask] = np.convolve(y[mask], kernel, mode='same')[:mask.sum()]
    return out

# -- Plot 1: Reward curve --
reward_col = 'Train/mean_reward'
if reward_col in df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    raw = df[reward_col].values
    ax.plot(steps, raw, alpha=0.25, color='blue', label='raw')
    ax.plot(steps, smooth(raw), color='blue', linewidth=2, label='smoothed')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Mean Reward')
    ax.set_title('Training Reward'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig('/tmp/exp_reports/reward.png', dpi=100); plt.close()

# -- Plot 2: Episode length --
ep_col = 'Train/mean_episode_length'
if ep_col in df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    raw = df[ep_col].values
    ax.plot(steps, raw, alpha=0.25, color='green')
    ax.plot(steps, smooth(raw), color='green', linewidth=2)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Mean Episode Length')
    ax.set_title('Episode Length'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig('/tmp/exp_reports/ep_length.png', dpi=100); plt.close()

# -- Plot 3: Loss curves --
loss_cols = [c for c in ['Loss/Value', 'Loss/Surrogate', 'Loss/Entropy'] if c in df.columns]
if loss_cols:
    fig, axes = plt.subplots(1, len(loss_cols), figsize=(5*len(loss_cols), 4))
    if len(loss_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, loss_cols):
        vals = df[col].dropna().values
        ax.plot(vals, linewidth=1); ax.set_title(col.split('/')[-1]); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig('/tmp/exp_reports/losses.png', dpi=100); plt.close()

# -- Plot 4: Thermal metrics --
thermal_cols = [c for c in ['Env/max_winding_temp', 'Env/mean_winding_temp'] if c in df.columns]
if thermal_cols and df[thermal_cols[0]].notna().any():
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {'Env/max_winding_temp': 'red', 'Env/mean_winding_temp': 'orange'}
    for col in thermal_cols:
        vals = df[col].dropna()
        ax.plot(vals.index, vals.values, label=col.split('/')[-1], color=colors[col], linewidth=1.5)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Temperature (C)')
    ax.set_title('Thermal Metrics'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig('/tmp/exp_reports/thermal.png', dpi=100); plt.close()

# -- Plot 5: Per-reward-term breakdown --
rew_cols = [c for c in df.columns if c.startswith('Episode/rew_')]
if rew_cols:
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in rew_cols:
        vals = df[col].dropna()
        ax.plot(vals.index, smooth(vals.values), label=col.replace('Episode/rew_',''), linewidth=1)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Reward Term')
    ax.set_title('Reward Breakdown'); ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig('/tmp/exp_reports/reward_breakdown.png', dpi=100); plt.close()

print("Plots saved to /tmp/exp_reports/")
PYEOF
```

**For multi-run comparison**, overlay all runs on the same axes with different colors/labels.
Name output files with job_id to avoid collision: `reward_{job_id}.png`.

### 3. Compute Statistics

For each key metric, compute:
- **Final**: mean of last 10% of iterations
- **Peak**: max value and at which iteration
- **Trend**: improving / plateaued / declining over last 20%
- **Convergence**: first iteration reaching 90% of peak
- **Stability**: std of last 20% (stable / noisy / unstable)

### 4. Build Report

Return a report in this exact format:

```
## Experiment Report: {name}

**Job**: {job_id} | **Steps**: {total_steps} | **State**: {done/failed/running}

### Key Metrics (final / peak)
- Reward: {final:.2f} / {peak:.2f} (peak at step {n})
- Episode Length: {final:.1f} / {peak:.1f}
- Max Winding Temp: {final:.1f}C (if thermal)
- Thermal Penalty: {final:.4f} (if thermal)

### Training Dynamics
- Convergence: reward hit 90% of peak at step {n}
- Trend: {improving / plateaued / declining}
- Stability: {stable / noisy / unstable}

### Plots
- /tmp/exp_reports/reward.png
- /tmp/exp_reports/ep_length.png
- /tmp/exp_reports/losses.png
- /tmp/exp_reports/thermal.png (if applicable)
- /tmp/exp_reports/reward_breakdown.png

### Issues & Recommendations
- {anomalies, instabilities, suggestions}
```

### Multi-Run Comparison

When comparing runs, add:
- Comparison table (name, final reward, final ep length, etc.)
- Which run is best and by how much
- Overlay plots with all runs
- Whether differences are significant or within noise

## Important

- NEVER dump raw data series into the report text
- Keep the text portion under 50 lines
- Let the plots carry the heavy information load
- If the caller asks a specific question, lead with the answer
- Prefer local TensorBoard data over WandB API — it's faster and has full resolution
