---
name: run-exp
description: Submit a single RL experiment to remote_exp_manager and monitor until completion
---

# Single Experiment Runner

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
  --tmux --snapshot \
  -- bash -c "source scripts/source_isaacsim_setup.sh && python src/holosoma/holosoma/train_agent.py {exp_config} simulator:isaacsim logger:wandb --training.seed {seed} {extra_overrides}"
```

**Multi-GPU (torchrun):**

Master port rule: `29501 + min(gpu_ids)`. E.g. GPU 0,1 → 29501; GPU 2,3 → 29503.

```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py submit \
  --name "{name}" \
  --cwd /home/ubuntu/Documents/gits/holosoma \
  --conda-env ~/.holosoma_deps/miniconda3/envs/hssim \
  --env CUDA_VISIBLE_DEVICES={gpu_ids} \
  --tmux --snapshot \
  -- bash -c "source scripts/source_isaacsim_setup.sh && torchrun --nproc_per_node={n_gpus} --master_port={port} src/holosoma/holosoma/train_agent.py {exp_config} simulator:isaacsim logger:wandb --training.num-envs {num_envs} {extra_overrides}"
```

Parse `job_id` from the output line: `submitted: id=XXXXXX_XXXXXX_X ...`

### 2. Confirm Launch

Wait 30s, then check the job started successfully:
```bash
sleep 30 && python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py list
```
If status is already `failed`, fetch log and return the error immediately.

### 3. Monitor

Poll until the job reaches a terminal state. Use this polling script with `timeout: 540000` (9 minutes):

```bash
python3 -c "
import json, time, urllib.request
JOB_ID = 'REPLACE_WITH_ACTUAL_ID'
while True:
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:8989/api/jobs')
        jobs = json.loads(resp.read())['jobs']
        job = next((j for j in jobs if j['id'] == JOB_ID), None)
        if job is None:
            print('ERROR: job not found'); break
        status = job['status']
        print(f'[poll] status={status} rc={job.get(\"return_code\",\"?\")}', flush=True)
        if status in ('done', 'failed', 'canceled'):
            print(json.dumps(job, indent=2)); break
    except Exception as e:
        print(f'[poll] error: {e}', flush=True)
    time.sleep(60)
"
```

If the command times out (experiment still running), **re-run the same script**. Repeat until terminal state.

### 4. Fetch Final Log

```bash
python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py log {job_id} --tail 100
```

### 5. Return Summary

Report back with:
- **Job ID** and **name**
- **Status**: done / failed / canceled
- **Return code**
- **Duration**: started_at to ended_at
- **Key metrics** from log tail (if visible): final `Train/mean_reward`, `Train/mean_episode_length`, `Env/max_winding_temp`
- **Errors**: any error messages
- **Log path**: `~/Documents/gits/remote_exp_manager/data/logs/{job_id}.log`
