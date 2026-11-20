# A2：用 V-JEPA 2（8 帧 clip）替换 DINO-Video 教师

- 版本：v0.1（2026-11-20）
- 上游文档：[`00-overall-plan.md`](00-overall-plan.md)（总计划，A2 的原始设计见第 0 节和阶段二）、[`01-stage1-offline-probe.md`](01-stage1-offline-probe.md)（阶段一探针，变体 V2）
- 相关报告：`research/jepa_probe/reports/stage1_report.md`、`research/jepa_probe/reports/label_check.md`
- **状态（2026-11-20）：** 代码和本地 CPU 测试已完成，下一步是 Pod 自检（第 4 节步骤 1）。进度见第 6 节。

---

## 0. 为什么重启 A2，为什么选 V2

B 线收尾以后（台账 #10），两条路各选一个最好的方案做完整训练。B1 的完整训练已经跑完（台账 #11、#12），这里是 JEPA 这条。

**选 V2 的理由（阶段一探针，台账 #5）：**
- 主指标（已知本体状态时反推动作块，P1s）上，V2 比 DINO-Video 低 9.6%，置信区间 [−18.0%, +0.8%]，是所有 JEPA 变体里离显著最近的。
- 夹爪开合的样本上，V2 的误差是 0.0072，DINO 是 0.0107。V-JEPA 2.1 的两个变体在这里比 DINO 差。
- 替换（A2）而不是加（A3）：输出形状和 DINO-Video 完全一样，可以沿用预训练 head 的权重，变量最少。

**已知风险：**
- G1 没有通过，这是一次"参考 run"，不是主消融。结论只能说"在这个配方下换教师有没有可见的作用"。
- 标签检验里，V2 未来特征里只有知道动作才能预测的部分比 DINO 少（ΔR² 0.019 对 0.055）。
- 预训练的 6B checkpoint 已经被 DINO-Video 蒸馏过，后训练阶段换教师抹不掉这段历史（总计划第 1 节）。

---

## 1. 教师输入：8 帧 clip

和阶段一 V2 完全一致（`research/jepa_probe/configs/variants.yaml`）：

```
[f_t, f_t, f_{t+8}, f_{t+16}, f_{t+24}, f_{t+33}, f_{t+41}, f_{t+49}]
```

- V-JEPA 2 的 tubelet 是 2 帧，8 帧编码成 4 个 tubelet，每个 tubelet 256 个 token × 1024 维。
- **当前目标** = 第一个 tubelet（`f_t` 重复两次，不混入未来帧）。
- **未来目标** = 最后一个 tubelet（`f_{t+41}, f_{t+49}`）。
- 中间帧的位置是 `round(49·k/6)`，k = 1…5，即 chunk 内的偏移 8、16、24、33、41。
- 只有顶部相机进教师，这和 DINO-Video 一样（`get_video_target` 只取 `pil_images[:, :1]`）。

**和 DINO-Video 的区别：** DINO 的 clip 是 `[f_t, f_t, f_{t+49}]`，当前目标取第 2 帧（`use_warmup_frame`），并且带 `effective_fps`。V-JEPA 按 tubelet 序号编码时间，不用 fps。

---

## 2. 代码改动

原则：默认值下 A0 的路径一个字节都不变；新逻辑只在显式打开时生效。

**数据（只在训练时，部署不受影响）：**
- 新增 `data.future_video_mid_frames`（chunk 内偏移的列表，默认空）。需要 `data.use_future_image: true`。
- `VLADataset.get_video_delta_timestamps`：只给顶部相机（`data.cameras` 的第一个）的原始列加中间帧的时间戳，其余相机仍然只取 `[t, t+49]`。
- `FeatureTransform.pad_and_concat`：当前帧仍取 `[0]`，未来帧仍取 `[-1]`，中间帧 `[1:-1]` 单独放进新键。**不能**把中间帧塞进 `future_pil_images`：它的第 1 维是相机，而且未来深度也用它。
- `FeatureTransform.apply`：中间帧用和当前帧、未来帧同一组颜色增强参数，输出新键 `future_video_mid_pil_images`，形状 `[M, 3, H, W]`（batch 后 `[B, M, 3, H, W]`），uint8。
- 部署时 `FeatureTransform` 不开 `use_future_image`，中间帧逻辑不会触发。

**教师：**
- 新增 `video.teacher_type`：`dino_video`（默认）或 `vjepa2`。
- `vjepa2` 读 `video.jepa_model_path`（HF 格式的 `vjepa2-vitl-fpc64-256` 目录），用 `VJepaTeacher.from_hf` 加载 fp32 权重、在 bf16 autocast 下前向，和阶段一提特征的方式相同。
- `get_video_target` 新增参数 `mid_pil_images`。`vjepa2` 时拼成 `[cur, cur, *mid, fut]`，帧数为奇数时末尾补一帧未来帧（所以不给中间帧时就是阶段一的 V1 `[cur, cur, fut, fut]`）；`current_index` 为 0。
- `vjepa2` 要求 `use_warmup_frame: false`、`use_cls_loss: false`；`dino_video` 不接受中间帧。配置不一致时启动就报错。

**训练脚本：** 从 batch 里取出 `future_video_mid_pil_images` 传给 `get_video_target`，不传进模型前向。启动时检查数据和教师配置是否一致。

**不改的部分：** prefix 布局、query 数量、head 结构和输出维度、loss 形式（MSE + 0.2·(1−cos)，当前 + 未来）、深度教师（未来深度仍然用 `t+49`）。head 从预训练的 DINO head 权重继续训练。

**工具：**
- `make_train_configs.py --arm A2`：写入 `teacher_type: vjepa2`、模型路径（`$W/vjepa/vjepa2-vitl-fpc64-256`，阶段一用的同一份）、`use_warmup_frame: false` 和中间帧偏移；`--video_mse_weight` 改 `mse_loss_weight`（第 3 节）。
- `check_pipeline.py`：`data` 模式检查中间帧的像素和阶段一读出的帧逐位相等；`video --variant V2` 模式比较训练路径算出的目标和阶段一 V2 的目标（每个 token 余弦 ≥ 0.999，相对 L2 ≤ 1%），并报告两种教师目标的数值尺度，用于校准权重。

---

## 3. loss 权重怎么定

V-JEPA 的特征尺度和 DINO-Video 不同，直接用 0.004 会改变视频 loss 在总 loss 里的占比。

视频 loss 是 `future_video_loss_weight × (mse_loss_weight × MSE + 0.2 × (1 − cos))`，当前和未来各一份。MSE 直接算在原始特征上（`modeling_lingbot_vla.py::_video_emb_loss`），随尺度变；余弦项与尺度无关。所以只校准 `mse_loss_weight`，总权重 0.004 和余弦权重 0.2 保持 A0 的值。

**校准方法：** 在同一批样本上分别算 DINO 和 V-JEPA 的目标，取"每个 token 位置、每一维都用样本均值去预测"的 MSE（即目标在样本间的方差，记作 `S`），当前和未来分别算。取

\[ \text{mse\_loss\_weight}_{A2} = S_{\text{DINO}} / S_{\text{V-JEPA}} \]

这让 MSE 项相对目标尺度的强度和 A0 一样（A0 的 `mse_loss_weight` 是 1.0）。当前和未来的比值相差不大时取两者的平均，取两位有效数字，写进台账；`make_train_configs.py --video_mse_weight` 写入配置。

**校准结果（2026-11-20，64 个训练样本，台账 #14）：**

| | 当前 | 未来 |
|---|---|---|
| `S_DINO` | 0.01250 | 0.01592 |
| `S_V-JEPA` | 5.529 | 5.278 |
| 比值 | 0.00226 | 0.00302 |

V-JEPA 目标的方差是 DINO 的 300 多倍。取两个比值的平均，**`mse_loss_weight` = 0.0026**。

**冒烟时第 1 步的视频 loss 不能和 A0 比：** head 是在 DINO 目标上预训练的，输出方向和 V-JEPA 特征对不上，余弦项一开始接近满值。日志里的 `FutureVideo_Loss` 不含 0.004 的总权重，所以当前、未来两份加起来最多约 0.2 × 2 = 0.4，A0 只有零点零几（B1 完整训练末尾是 0.026）。乘上 0.004 以后约 0.0016，比 VLA loss 小一个数量级，不会主导梯度。冒烟要看的是它在 30 步里明显下降、没有非有限值。

---

## 4. Pod 上的步骤

1. **自检（CPU + 单卡，约 20 分钟）：**
   - `check_pipeline data`：A2 配置下，中间帧的像素和阶段一逐位相等。
   - `check_pipeline video --variant V2`：训练路径的 V-JEPA 目标和阶段一一致；同时得到校准用的尺度比。

   ```bash
   source $LB/research/tools/pod/pod.env; cd $LB; P=$D/jepa/probe
   # 先用默认 mse 权重生成配置；自检只读它的 data 段和 video 段
   python research/tools/pod/make_train_configs.py formal --arm A2 --micro_batch 64 --accum 1 \
       --max_steps 10735 --save_steps 2147 --stop_step 2147
   python -m research.jepa_probe.check_pipeline data --variant V2 --train_config $D/configs/jepa_A2_s42.yaml \
       --index $P/index --samples 50 --out $R/jepa/a2_check_data.json
   python -m research.jepa_probe.check_pipeline video --variant V2 --train_config $D/configs/jepa_A2_s42.yaml \
       --index $P/index --batches 4 --batch 16 --out $R/jepa/a2_check_video.json
   # 用 video 报告里的 suggested_mse_loss_weight 重新生成配置
   python research/tools/pod/make_train_configs.py formal --arm A2 --video_mse_weight <建议值> --micro_batch 64 \
       --accum 1 --max_steps 10735 --save_steps 2147 --stop_step 2147
   ```

   判据：`data` 的 5 项全部 PASS（pixels 项含中间帧，config 项比较 V2 的帧位置、输入尺寸、层和权重目录）；`video` 的 `v2_target` 余弦均值 ≥ 0.999、相对 L2 ≤ 1%。训练路径的目标是 bf16，阶段一的是 fp32，所以两者不会逐位相同，相对 L2 应在 0.4% 左右。
2. **冒烟（4 卡 × micro batch 64，30 步，S3b 的设置）：**
   - **A0 回归：** 新代码跑 A0，第 1 步各项 loss 与旧的 S3b 日志（`$D/smoke/s3b.log`）一致。
   - **A2：**
     - 不出现非有限值；
     - 第 1 步的 `Depth_Loss`、`Future_Depth_Loss` 与 A0 一致（深度教师和输入没变）；
     - `FutureVideo_Loss` 在 30 步里明显下降（第 1 步不和 A0 比，见第 3 节）；
     - StepTime 不超过 A0 的 1.1 倍；
     - 用 `step_wall` 看数据等待有没有变长（每个样本多解码 5 帧，阶段一实测约占数据加载的 24%）。
3. **试点（1 个 epoch）：** `--max_steps 10735 --stop_step 2147`，和 `train_c50_v1` 同步数比较 VLA loss（标准：不劣于 1.05 倍）。
4. **完整训练：** 不通过试点就停；通过就从 2147 步续训到 10735 步，和 A0（`train_c50_v1`）、B1（`jepa_B1_s42`）比较。
5. 成功率评测仍然卡在评测链路上（OIDN 不支持 sm_103），和 B1 一起等。

---

## 5. 判定

- **试点（G3）：** 不发散；VLA loss 不劣于 A0 同步数的 1.05 倍。
- **完整训练：** 单 seed 的参考 run，只报告 loss 曲线和成功率，不做预注册的统计判定。成功率比 A0 高出评测噪声（以阶段零测得的为准）才考虑补 seed。

---

## 6. 进度记录

| 日期 | 事项 | 结果 |
|---|---|---|
| 2026-11-20 | 设计定稿（本文档），决定重启 A2 | 台账 #13 |
| 2026-11-20 | 代码：数据中间帧、`teacher_type: vjepa2`、8 帧 clip、训练脚本接线、`--arm A2`、`check_pipeline --variant V2` | 完成，见第 2 节 |
| 2026-11-20 | 本地 CPU 测试 | 84 个全部通过（新增 15 个，见下） |
| 2026-11-20 | Pod 自检（包 `fa8730e`） | 通过。data 5 项 PASS（中间帧逐位相等）；`v2_target` 余弦均值 0.999999，相对 L2 0.17%（台账 #14） |
| 2026-11-20 | 尺度校准 | `mse_loss_weight` = 0.0026（第 3 节） |
| — | 冒烟（30 步）、试点（1 个 epoch） | 待做 |

**新增的本地测试：**
- `research/model_tests/test_jepa_video_target.py`（8 个）：
  - clip 的帧序是 `[当前, 当前, 中间 5 帧, 未来]`，`current_index` 为 0；
  - 不给中间帧时补成 `[当前, 当前, 未来, 未来]`；
  - 同一个 toy 教师，训练路径和阶段一探针 runner 的输出逐位相同（含需要缩放的输入尺寸）；
  - DINO 路径不变，并且拒收中间帧；配置不一致时报错；
  - `build_video_model` 按 fp32 加载 V-JEPA 且冻结参数；
  - 中间帧和当前帧用同一组颜色增强参数。
- `research/jepa_probe/tests/test_check_pipeline.py`（新增 5 个，用 lerobot 写出的真实格式小数据集，走 LingBot 自己的 `VLADataset`）：
  - 只有顶部相机多取中间帧，其余相机仍是 `[t, t+49]`；
  - 未来深度用的 `future_pil_images` 不变；
  - `data --variant V2` 通过；中间帧位置写错时 config 项失败；
  - 不开 `use_future_image`（部署时就是这样）时中间帧逻辑关闭；
  - `video --variant V2` 通过，并给出尺度比。
- `research/tools/tests/test_pod_configs.py`（新增 2 个）：A2 配置的各项字段；A2 缺 V-JEPA 权重时报错；`--video_mse_weight` 和 `--ac_*` 不能用在不对应的臂上。

**本地没法验证、留给 Pod 的：** 真实 V-JEPA 2 权重的前向（HF `VJEPA2Model`，阶段一已在 Pod 镜像里跑过）、bf16 autocast 下的数值、吞吐和数据等待。
