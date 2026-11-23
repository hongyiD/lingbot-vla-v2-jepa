# JEPA teachers and action-conditioned prediction for LingBot-VLA 2.0 post-training

A research fork of [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) (upstream `be969b8`; the original README is in [`README_upstream.md`](README_upstream.md)).

During training, LingBot-VLA 2.0 distills two kinds of targets into the VLM's image tokens through small heads: depth maps from LingBot-Depth, and video features of the current frame and of the frame 49 steps later from a DINO-Video teacher. This fork asks whether that video target can be made more useful for control, in two ways:

1. **Replace the teacher with V-JEPA 2**, a self-supervised video model trained to predict in feature space.
2. **Make the prediction action-conditioned**: a head that sees the action chunk and predicts how the scene changes.

All experiments use the same setting as the baseline: RoboTwin 2.0 `aloha-agilex` clean demonstrations (50 tasks × 50 episodes), full-parameter post-training of `lingbot-vla-v2-6b` on 4× NVIDIA B300, global batch 256, 10,735 steps (5 epochs), cosine LR 1e-4 → 5e-5. Every change is compared with the unchanged recipe (A0) at the same seed and data order.

## Results

All numbers are training-time measurements from single-seed runs.

### Stage 1: offline probes of teacher features

Frozen features from each teacher, on (current, future) frame pairs sampled every 10 frames from the 2,500 training episodes, built exactly as in LingBot training, with a held-out split by episode. Small attention probes predict the 50-step action chunk from the features. The main metric is P1s, an inverse-dynamics probe that also gets the proprioceptive state, because the action expert already has it.

| Teacher (frames encoded) | P1s test nMSE | vs DINO-Video | Without state (P1) | Gripper-event subset (P1s) |
|---|---|---|---|---|
| DINO-Video, LingBot's own (`[f_t, f_t, f_t+49]`) | 0.0244 | — | 0.0441 | 0.0107 |
| V-JEPA 2 ViT-L, 2 frames | 0.0272 | +11.9%, n.s. | 0.0463 | 0.0142 |
| V-JEPA 2 ViT-L, 8 frames `t … t+49` | **0.0217** | −9.6% [−18.0%, +0.8%], n.s. | **0.0362** (−14.0%, sig.) | **0.0072** (−33%) |
| V-JEPA 2.1 ViT-L, 384 px | 0.0247 | −3.0%, n.s. | 0.0371 (−17.5%, sig.) | 0.0117 |

- The pre-registered gate required a JEPA variant to beat DINO-Video significantly on P1s. **None did**, so replacing the teacher was not supported by this test.
- The 8-frame V-JEPA 2 clip was best on all three inverse-dynamics probes, and clearly better around gripper open/close events (gripper dimensions 0.0043 vs 0.0099).
- V-JEPA features did not lose pose information. On a probe that reads the robot state from the frame (P2), every V-JEPA variant except the separately encoded one beat DINO-Video by 38–65%.
- The full report covers ten inputs, four probes, a teacher-cost benchmark and token PCA: [`research/jepa_probe/reports/stage1_report.md`](research/jepa_probe/reports/stage1_report.md).

### Stage 2: action-conditioned prediction head

A training-only head reads the VLM tokens, the state and the action chunk, and predicts the change in DINO-Video features between now and 49 steps later. The target was chosen by an offline label check ([`label_check.md`](research/jepa_probe/reports/label_check.md)). Arms: B1 feeds the real actions; B0 is the same head fed zeros.

- **Pilots (1 epoch, two context designs, three loss weights): B1 = B0.** At matched weight, the real actions did not lower the head's loss beyond the zero-action control: over the last 1,000 steps the difference was within ±0.0003 of the zero-action loss. The label check explains why: in these scripted demonstrations, the current frame already predicts about 85% of the change, and the action adds about 6%.
- **Full run, B1 against A0:** VLA (action) loss unchanged (epoch 5 ratio of means +0.04% [−0.72%, +0.79%]). Current-frame depth loss worse by 3.1%, future depth better by 0.45%, about 7% slower per step.

### Stage 3: V-JEPA 2 replaces DINO-Video (A2)

A2 is a single-seed reference run of the best Stage 1 variant: an 8-frame V-JEPA 2 clip `[f_t, f_t, f_t+8, f_t+16, f_t+24, f_t+33, f_t+41, f_t+49]`. The current target is tubelet 0, the future target is the last tubelet, and the MSE weight is rescaled by the measured target variance (0.0026). The existing video head is reused.

Paired against A0 at the same steps (ratio of means, 95% CI from a 50-step block bootstrap):

| Epoch | VLA loss, A2 vs A0 | VLA loss, B1 vs A0 | Current depth, A2 vs A0 | Current depth, B1 vs A0 |
|---|---|---|---|---|
| 1 | +0.61% [+0.25, +0.98] | +0.55% [+0.23, +0.85] | −0.51% | +0.34% |
| 2 | +1.33% [+0.81, +1.83] | +0.16% | −0.52% | +1.33% |
| 3 | +1.39% [+0.72, +2.01] | +0.11% | −0.40% | +2.53% |
| 4 | +2.82% [+2.07, +3.57] | −0.05% | −0.38% | +3.02% |
| 5 | +3.33% [+2.71, +3.97] | +0.04% [−0.69, +0.76] | −0.38% | +3.10% |

- **The two changes trade off in opposite directions.** A2 keeps current depth slightly better but raises the action loss by a small, growing amount. B1 leaves the action loss alone and pays in current depth. Both improve future depth by about 0.5%.
- **The V-JEPA target is much harder to fit.** Its cosine loss plateaus at about 0.25–0.28, five to seven times DINO-Video's, and the gradient norm stays about 9% higher. A likely explanation for the growing action-loss gap is that this gradient keeps competing with the action loss; this has not been tested yet.
- **Cost:** in the first epoch, steps were about 1% slower than A0, even though each sample decodes five extra frames.

## What is in this repository

**Model and training changes** (all off by default; with the default configs, training is the upstream recipe):

| Path | Change |
|---|---|
| `lingbotvla/models/vla/vision_models/jepa/teacher.py` | V-JEPA 2 / 2.1 teacher wrapper (HF and official checkpoints), shared by the probes and training |
| `lingbotvla/models/vla/vision_models/module_utils.py` | `video.teacher_type: dino_video \| vjepa2`; builds the multi-frame V-JEPA clip target; config checks |
| `lingbotvla/data/vla_data/{base_dataset,utils}.py` | `data.future_video_mid_frames`: extra top-camera frames inside the action chunk, with the sample's own colour augmentation |
| `lingbotvla/models/vla/vision_models/align_heads/action_cond_head.py`, `modeling_lingbot_vla_v2.py` | Action-conditioned prediction head with real / zero / shuffled-action diagnostics |
| `tasks/vla/train_lingbotvla.py` | Wiring for the above; `train.stop_step` stops and saves without shortening the LR schedule, so long runs can be split across GPU windows and resumed |

**Research code** (`research/`):
- `jepa_probe/`: Stage 1 pipeline, covering the sample index, frame reading identical to training, feature extraction, attention and ridge probes, the label check, the teacher-cost benchmark, and a pipeline check that compares offline targets with the training data path.
- `tools/`: paired statistics, the append-only experiment ledger, and Pod helpers (config generator for every arm, metrics parser that merges resumed logs, launch functions).
- `ledger/`: every decision, run and comparison above, with configs, commits and hashes.
- Tests: 89 CPU tests, using synthetic data and small random models.

**Plans** (`plan/`, in Chinese): the overall research plan with pre-registered gates, and one design-and-progress document per stage.

## Running

```bash
# tests (CPU only)
python -m pytest -q research/tools/tests research/model_tests research/jepa_probe/tests

# training configs for an arm, derived from configs/vla/robotwin/robotwin.yaml
python research/tools/pod/make_train_configs.py formal --arm A2 --video_mse_weight 0.0026 \
    --micro_batch 64 --accum 1 --max_steps 10735 --save_steps 2147
```

Data preparation and training otherwise follow the upstream instructions in [`README_upstream.md`](README_upstream.md). The V-JEPA 2 weights come from [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256).

## Notes

- Datasets, weights, checkpoints and training logs are not included.
- Plans, ledger notes and some comments are in Chinese.
- Licensed under Apache 2.0, as upstream. This fork modifies the files listed above and adds `research/` and `plan/`.
