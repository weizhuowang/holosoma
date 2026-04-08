---
name: report-exp
description: Generate a concise experiment report from WandB data — plots + statistical summary
---

# Experiment Report Generator

You analyze RL training runs from WandB and produce a **compact report** for the main agent.
The main agent has limited context — your job is to compress thousands of iterations into
a summary that enables decision-making.

## Strategy: Plots + Key Numbers

Raw data is too large for context. Instead:
1. **Generate plots** as PNG images (Claude can read images — one plot = thousands of data points)
2. **Compute statistics** (final, peak, convergence point, trend)
3. **Write analysis** in text (anomalies, comparisons, recommendations)
4. Return image paths + short text summary. DO NOT return raw data series.

## Constants

- WandB base URL: `https://far.wandb.io`
- Python: `~/.holosoma_deps/miniconda3/envs/hssim/bin/python`
- Report output dir: `/tmp/exp_reports/` (create if missing)
- Thermal projects: `g1-manager-thermal`, `g1-manager-thermal-baseline`

## Inputs

The caller provides one or more of:
- WandB project name + run ID(s)
- WandB project name + filters (e.g. "all running", "name contains thermal_s1")
- A specific question to answer (e.g. "is thermal penalty helping?", "which seed is best?")

## Workflow

### 1. Fetch Data

Use WandB API with **sampling** to keep data manageable:

```bash
~/.holosoma_deps/miniconda3/envs/hssim/bin/python << 'PYEOF'
import wandb, os, json
os.environ['WANDB_BASE_URL'] = 'https://far.wandb.io'
api = wandb.Api()

run = api.run("PROJECT/RUN_ID")

# Sampled history — 200 points is enough for plots and stats
# WandB evenly samples across the full run
keys = [
    'Train/mean_reward', 'Train/mean_episode_length',
    'Env/max_winding_temp', 'Env/mean_winding_temp',
    'Episode/rew_penalty_joint_temperature',
    'Loss/value_loss', 'Loss/surrogate_loss', 'Loss/entropy',
    'Perf/total_fps', '_step', '_runtime',
]
df = run.history(keys=keys, samples=200, pandas=True)
df.to_csv('/tmp/exp_reports/run_data.csv', index=False)

# Run metadata
print(json.dumps({
    'id': run.id,
    'name': run.name,
    'state': run.state,
    'config': dict(run.config),
    'summary': dict(run.summary),
    'created_at': run.created_at,
}, indent=2, default=str))
PYEOF
```

For multi-run comparison, fetch each run separately and save as `run_{id}.csv`.

### 2. Generate Plots

Create plots with matplotlib. Save as PNG to `/tmp/exp_reports/`.

**Essential plots (always generate):**

```bash
~/.holosoma_deps/miniconda3/envs/hssim/bin/python << 'PYEOF'
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, numpy as np

os.makedirs('/tmp/exp_reports', exist_ok=True)
df = pd.read_csv('/tmp/exp_reports/run_data.csv')
steps = df['_step'].values

def smooth(y, window=10):
    """Simple moving average for noisy RL curves."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')

# Plot 1: Reward curve
fig, ax = plt.subplots(figsize=(10, 4))
raw = df['Train/mean_reward'].values
ax.plot(steps, raw, alpha=0.3, color='blue', label='raw')
sm = smooth(raw)
ax.plot(steps[:len(sm)], sm, color='blue', linewidth=2, label='smoothed')
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Reward')
ax.set_title('Training Reward')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('/tmp/exp_reports/reward.png', dpi=100)
plt.close()

# Plot 2: Episode length
fig, ax = plt.subplots(figsize=(10, 4))
raw = df['Train/mean_episode_length'].values
ax.plot(steps, raw, alpha=0.3, color='green')
sm = smooth(raw)
ax.plot(steps[:len(sm)], sm, color='green', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Episode Length')
ax.set_title('Episode Length')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('/tmp/exp_reports/ep_length.png', dpi=100)
plt.close()

# Plot 3: Loss curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, label in zip(axes,
    ['Loss/value_loss', 'Loss/surrogate_loss', 'Loss/entropy'],
    ['Value Loss', 'Surrogate Loss', 'Entropy']):
    vals = df[col].dropna().values
    ax.plot(vals, linewidth=1)
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('/tmp/exp_reports/losses.png', dpi=100)
plt.close()

# Plot 4: Thermal metrics (if available)
if 'Env/max_winding_temp' in df.columns and df['Env/max_winding_temp'].notna().any():
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, label, color in [
        ('Env/max_winding_temp', 'Max Winding Temp', 'red'),
        ('Env/mean_winding_temp', 'Mean Winding Temp', 'orange'),
    ]:
        if col in df.columns:
            vals = df[col].dropna()
            ax.plot(vals.values, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Temperature (C)')
    ax.set_title('Thermal Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('/tmp/exp_reports/thermal.png', dpi=100)
    plt.close()

print("Plots saved to /tmp/exp_reports/")
PYEOF
```

**For multi-run comparison, overlay all runs on the same axes** with different colors/labels.

### 3. Compute Statistics

For each key metric, compute:
- **Final**: mean of last 10% of iterations
- **Peak**: max value and at which iteration
- **Trend**: is it still improving, plateaued, or declining?
- **Convergence**: iteration at which the metric first reaches 90% of its peak (approximate)

### 4. Build Report

Return a report in this exact format:

```
## Experiment Report: {run_name}

**Run**: {id} | **Project**: {project} | **State**: {state}
**Duration**: {runtime} | **Steps**: {total_steps} | **FPS**: {avg_fps}

### Key Metrics (final / peak)
- Reward: {final_reward:.2f} / {peak_reward:.2f} (peak at step {peak_step})
- Episode Length: {final_ep_len:.1f} / {peak_ep_len:.1f}
- Max Winding Temp: {final_temp:.1f}C (if thermal)
- Thermal Penalty: {final_penalty:.4f} (if thermal)

### Training Dynamics
- Convergence: reward reached 90% of peak at step {convergence_step}
- Trend: {improving / plateaued / declining} over last 20% of training
- Stability: {stable / noisy / unstable} (based on reward variance in last 20%)

### Plots
(list the image paths so the main agent can read them)
- Reward curve: /tmp/exp_reports/reward.png
- Episode length: /tmp/exp_reports/ep_length.png
- Losses: /tmp/exp_reports/losses.png
- Thermal: /tmp/exp_reports/thermal.png (if applicable)

### Issues & Recommendations
- {any anomalies, instabilities, or suggestions}
```

### Multi-Run Comparison

When comparing multiple runs, add:
- A comparison table (run name, final reward, final ep length, etc.)
- Which run is best and by how much
- Overlay plots with all runs
- Whether differences are significant or within noise

## Important

- NEVER dump raw data series into the report text
- Keep the text portion under 50 lines
- Let the plots carry the heavy information load
- If the caller asks a specific question, lead with the answer
