# Experiment Log

Records of all RL training experiments launched via `/launch-exp` or `/run-exp`.

## Format

Each experiment gets two entries: one at launch, one at completion.

### Launch entry
```
### [LAUNCHED] {name} — {date}
- **Job ID**: {job_id}
- **Config**: {exp_config}
- **GPU**: {gpu_ids} ({n_gpus} GPU)
- **Seed**: {seed}
- **num-envs**: {num_envs}
- **Overrides**: {extra_overrides or "none"}
- **Hypothesis/Goal**: {why this experiment is being run}
```

### Completion entry (append below the launch entry)
```
**[DONE]** {date} | status={status} | rc={return_code} | elapsed={elapsed}
- **Final reward**: {value}
- **Final ep length**: {value}
- **Max winding temp**: {value} (if thermal)
- **Takeaway**: {one-line summary of what we learned}
```

---

<!-- Experiments below -->

### [LAUNCHED] thermal_smoke_verify — 2026-04-08
- **Job ID**: 260408_065050_0
- **Config**: exp:g1-29dof-thermal
- **GPU**: 1 (1 GPU)
- **Seed**: 42
- **num-envs**: 10240
- **Overrides**: --algo.config.num-learning-iterations 200
- **Hypothesis/Goal**: Smoke test to verify recent code changes (command range curriculum, per-term termination stats logging, TorqueMonitor callback) don't crash during training

**[DONE]** 2026-04-08 | status=done | rc=0 | elapsed=24min
- **Final reward**: 25.85
- **Final ep length**: 470.02
- **Max winding temp**: 86.07
- **Takeaway**: All 3 new features working — `Env/term_contact`/`term_timeout`/`term_temperature` logged correctly, `Env/command_curriculum_alpha`=0.0 (expected at 200 iter, curriculum starts at 2000). No crashes.

### [LAUNCHED] baseline_g1_29dof_s1 — 2026-04-08
- **Job ID**: 260408_091546_0
- **Config**: exp:g1-29dof
- **GPU**: 1 (1 GPU)
- **Seed**: 1
- **num-envs**: 10240
- **Overrides**: none
- **Hypothesis/Goal**: Baseline non-thermal experiment for comparison against thermal runs. Previous attempts (260408_075316_0, 260408_075547_0, 260408_085431_0) failed due to --logger.wandb.notes causing tyro parsing crash. Resubmitted without that flag.

### [LAUNCHED] thermal_baseline_s1 — 2026-04-08
- **Job ID**: 260408_093532_0
- **Config**: exp:g1-29dof-thermal
- **GPU**: 1 (1 GPU)
- **Seed**: 1
- **num-envs**: 10240
- **Overrides**: none
- **Hypothesis/Goal**: Full thermal baseline (2000 iter) with command curriculum + termination stats. For comparison against non-thermal baseline.

**[DONE]** 2026-04-08 | status=done | rc=0 | elapsed=3h35m
- **Final reward**: ~25 (from WandB sparkline)
- **Final ep length**: 998.95 (hitting timeout)
- **Max winding temp**: 84.73
- **Takeaway**: Thermal penalty worked — temp_penalty_max peaked mid-training then decreased as policy learned to manage heat. penalty_scale ramped from 0.1→1.0. command_curriculum_alpha=0 (2000 iter < start at iter 2000). term_contact dropped to 0 by end. WandB: https://far.wandb.io/far-wandb/g1-manager-thermal/runs/tycbhk8y

### [LAUNCHED] nonthermal_baseline_s1 — 2026-04-08
- **Job ID**: 260408_093534_0
- **Config**: exp:g1-29dof
- **GPU**: 2 (1 GPU)
- **Seed**: 1
- **num-envs**: 10240
- **Overrides**: none
- **Hypothesis/Goal**: Non-thermal baseline (2000 iter) for A/B comparison. Same env class but no thermal obs/reward/termination.

**[DONE]** 2026-04-08 | status=done | rc=0 | elapsed=03:32
- **Final reward**: 25.00
- **Final ep length**: 1001.00
- **Takeaway**: Non-thermal baseline completed successfully. Mean episode length saturated at ~1000 (timeout). WandB run: https://far.wandb.io/far-wandb/g1-manager-thermal/runs/zccl4sic
