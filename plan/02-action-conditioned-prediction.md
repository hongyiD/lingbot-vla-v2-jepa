# 阶段二（改道后）：动作条件预测头 B0 / B1

- 版本：v0.1（2026-11-16）
- 上游文档：[`00-overall-plan.md`](00-overall-plan.md)（总计划）、[`01-stage1-offline-probe.md`](01-stage1-offline-probe.md)（阶段一）
- 相关报告：`research/jepa_probe/reports/stage1_report.md`、`research/jepa_probe/reports/label_check.md`

---

## 0. 为什么改道

**阶段一的结论：** 按预注册标准，G1 没有通过。JEPA 特征对机器人位姿很敏感，但没有超出本体状态已经提供的信息；V-JEPA 2 在夹爪事件上明显更好；其余差异不显著。台账记录推迟到补充实验做完。

**新的问题：** 与其只换"拿哪个教师当目标"，不如改"VLM 学什么"。借鉴 V-JEPA 2-AC 的做法：给定当前画面、本体状态和接下来要执行的动作块，预测画面会怎样变化。这个任务只有在理解"动作会造成什么后果"时才做得好，梯度回传到 VLM，可能让 VLM 的表征更贴近动作。

**这一版只做一件事：** 在现有 LingBot 后训练配方上加一个训练时才用的预测头，用 DINO-Video 的"未来 − 当前"特征差作为标签，比较"给真动作"（B1）和"给零动作"（B0）。

---

## 1. 标签为什么选 DINO 的"未来 − 当前"

`label_check.py` 在阶段一的离线特征上比较了 7 种候选标签。

**做法：**
- 输入是"当前画面特征 + 本体状态 + 动作块"，用一个 MLP 预测各候选标签。
- 指标是 ΔR²：加上动作块以后，预测得比只用当前画面和状态好多少，也就是标签里"只有知道动作才能预测的部分"有多大。
- 按测试集 episode 做配对 bootstrap，给出与参考标签之差的置信区间。

| 标签 | ΔR²（MLP） |
|---|---|
| V-JEPA 2.1 的未来 − 当前（`v21_delta`） | 0.0675 |
| **DINO-Video 的未来 − 当前（`dino_delta`）** | **0.0612** |
| V-JEPA 2，7 帧 clip 的未来帧（`v2_7f_future`） | 0.0558 |
| V-JEPA 2，7 帧联合编码的未来 − 当前 | 0.0515 |
| V-JEPA 2，7 帧分别编码的未来 − 当前 | 0.0444 |
| V-JEPA 2，2 帧分别编码的未来 − 当前 | 0.0435 |
| DINO-Video 的未来帧本身（`dino_future`） | 0.0385 |

**结论：**
- "差值"类标签明显好于"未来帧本身"，因为未来帧的大部分内容（背景、桌面）从当前画面就能猜到。
- V-JEPA 2.1 的差值与 DINO 差值之差的置信区间是 [−0.0001, +0.0106]，不显著；让 JEPA 看中间帧也没带来优势。
- DINO 差值的现成目标已经在训练管线里（`future_video_targets` 和 `future_video_current_patch`），不增加教师前向。所以先用 DINO 差值。

**注意：** 这个测法把特征压到了 4×4 网格和 PCA 256 维，可能抹掉夹爪附近的细节。阶段一的注意力探针里 V-JEPA 2 在夹爪上更好。要做 JEPA 标签臂（B2，见第 6 节）之前，先补一个细粒度的标签检验（8×8，注意力预测器）。

---

## 2. 训练方式

### 2.1 新增的预测头

**位置：** `lingbotvla/models/vla/vision_models/align_heads/action_cond_head.py`（`ActionConditionedHead`），由 `FlowMatchingV2.init_action_cond_head` 构建，挂在 `model.action_cond_head`。

**输入（拼成一个上下文序列，约 83 个 token，每个 2560 维）：**
1. VLM 输出的顶部相机 64 个图像 token。默认不 detach，梯度回到 VLM。
2. VLM 输出的 8 个未来 task token（`Q_{t+T}`，与现有 future head 读的是同一段）。
3. 1 个本体状态 token：`Linear(55 → 2560)`，无效关节置零。
4. 10 个动作 token：50 步的干净动作块（不加噪声），每 5 步拼成一个 275 维向量，经 `Linear(275 → 2560)`，再加可学习的时间位置嵌入。无效关节和 episode 末尾补齐的步都置零。

**结构：** 与现有 head 相同的 1 层 Perceiver（`TaskTokenResampler`，4 头 × 32 维，ff_mult 1），256 个可学习 query，输出 256 × 1024。

**与现有 head 的两点不同：**
- 去掉了输出端的 LayerNorm。目标是"变化量"，静止的 patch 应该接近 0；输出 LayerNorm 会把每个 token 拉到同样的模长。
- 输出层零初始化。第一步的预测全是 0，loss 恰好等于"预测不变"的基线 `align/action_cond_zero_loss`，方便看后面学到了多少。

**参数量：** 约 30M，与现有 future head 相当。只在训练时使用；推理路径 `sample_actions` 不调用它。

### 2.2 目标与 loss

- **目标：** 对每个 patch token，`LN(DINO 未来) − LN(DINO 当前)`。LN 是不带参数的 LayerNorm（在 1024 维上做），先消掉两帧之间的整体亮度和尺度差，再相减。标签不回传梯度。
- **loss：** L1，对 256 × 1024 取平均，乘 `loss_weight`，加进总 loss。
- 现有的深度、DINO 当前和未来四个蒸馏项都保留不变。

**明确不做的事：** 不把教师的当前帧特征喂给这个 head。否则 head 只要学"教师当前特征 + 动作 → 变化"，完全绕开了 VLM，这个实验就失去意义。

### 2.3 打乱动作诊断

每一步额外做一次不回传梯度的前向：把 batch 内的动作块整体错位一个样本（第 i 个样本用第 i−1 个样本的动作），重新算 loss。

- `align/action_cond_shuffle_gap` = 打乱后的 loss − 真动作的 loss。大于 0 说明 head 确实在用动作。
- B0 喂的是零动作，这个值恒为 0。
- 原计划每 200 步算一次。但整个模型在 `torch.compile` 下运行，按步数计数器分支会让编译图随 Python 整数变化，所以改为每步都算。额外开销只是这个小 head 的一次无梯度前向。

### 2.4 实验臂

| 臂 | 视频教师 | 动作条件 head | 含义 |
|---|---|---|---|
| A0 | DINO-Video | 无 | 官方配方。`train_c50_v1` 作为 seed 42 |
| A1 | 无（模块保留，不算 loss） | 无 | 对照：后训练里视频蒸馏本身有没有用。**代码还没做**，见第 5 节 |
| B0 | DINO-Video | 有，喂零动作 | 控制"多了一个 head、多了一项 loss、多了梯度"这些因素 |
| B1 | DINO-Video | 有，喂真动作 | 主假设 |

B1 与 B0 只差"head 看不看得到动作"，其余完全一样（同样的参数量、同样的目标、同样的 loss 权重）。所以 B1 对 B0 是主比较。

### 2.5 判定规则（阶段四，预注册）

- **主终点：** B1 对 B0 的成功率（clean 与 randomized 两个设置，3 个 seed 合并），用 `ab_test.py` 的配对检验，置信区间不含 0 才算 B1 胜。
- **辅助：**
  - B1 对 A0：新 head 相对官方配方有没有净收益；
  - B1 的 `AC_Shuffle_Gap` 在训练后期稳定大于 0；
  - B1 的 VLA loss 不高于 A0 同步数的 1.05 倍。
- **提前止损：** 1 个 epoch 的试点里，如果 B1 的 `AC_Shuffle_Gap` 基本是 0（head 没在用动作），就不进主消融，先查原因（权重太小、动作 token 设计、标签信号太弱）。

---

## 3. 代码改动清单

| 文件 | 改动 |
|---|---|
| `lingbotvla/models/vla/vision_models/align_heads/action_cond_head.py` | 新增 `ActionConditionedHead` 与 `teacher_delta_target` |
| `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py` | `init_action_cond_head`、`action_cond_emb_forward`；`FlowMatchingV2.forward` 新增 `joint_mask`、`state_joint_mask`、`action_is_pad` 参数；`LingbotVlaV2Policy.post_training_new_modules` |
| `lingbotvla/models/module_utils.py` | 后训练加载时，对模型声明的新子模块（整个都不在 checkpoint 里）调用它自己的 `reset_parameters()`，不再触发"缺参数"断言。只缺一部分时报错 |
| `tasks/vla/train_lingbotvla.py` | 启用时在每步日志里加 `AC_Loss`（未加权）和 `AC_Shuffle_Gap`，跨卡平均 |
| `configs/vla/robotwin/robotwin.yaml` | 新增 `train.align_params.action_cond`，默认 `enabled: false` |
| `research/tools/pod/make_train_configs.py` | `--arm A0/B0/B1`、`--seed`、`--ac_weight`；formal 模式的默认名为 `jepa_<臂>_s<seed>`；smoke 模式带 `--arm` 时文件名和输出目录加 `_b0` / `_b1` 后缀 |
| `research/tools/pod/train_metrics.py` | 解析 `AC_Loss`、`AC_Shuffle_Gap`；`--ac_calibrate N` 给出 loss 权重建议 |
| `research/model_tests/`、`research/tools/tests/test_pod_configs.py` | CPU 单元测试，共 17 个（见第 4 节） |

**守住的约束（来自总计划 2 节的"已知的坑"）：**
- 输出元组长度不变（仍是 11）。新 loss 通过一个私有键 `_action_cond_loss` 交给 `LingbotVlaV2Policy.forward`，在那里取出并加进总 loss。`FutureVideo_Loss` 的含义不变。
- 新参数都在子模块 `action_cond_head` 里，没有在 `FlowMatchingV2` 上挂裸 `nn.Parameter`。
- 关闭时（A0）不构建这个 head，模型结构和随机数消耗都与改动前一致。
- 推理时 `deploy/lingbot_vla_v2_policy.py` 从训练配置构建模型并严格加载权重。B 臂的 checkpoint 里有这个 head，配置里也开着它，所以两边一致。

**配置项（`train.align_params.action_cond`）：**

| 键 | 默认 | 说明 |
|---|---|---|
| `enabled` | false | 打开 head |
| `use_actions` | true | false 即 B0，喂零动作 |
| `loss_weight` | 0.004 | 占位值。试点前按 3.1 节校准 |
| `action_group` | 5 | 每几步动作合成一个 token，要能整除 50 |
| `detach_image_feats` | false | true 时梯度不经图像 token 回到 VLM（只经未来 task token） |
| `shuffle_metric` | true | 每步计算打乱动作诊断 |
| `num_layers` / `num_heads` / `dim_head` / `ff_mult` | 1 / 4 / 32 / 1 | 不写时沿用 video head 的设置 |

### 3.1 loss 权重怎么校准

目标：第一步时，新 head 的加权 loss 与现有 DINO 视频项（当前 + 未来）的加权 loss 量级相同。这是一项实际的定义调整：原计划只对齐"未来 DINO 项"，但训练日志只打印两者之和 `FutureVideo_Loss`。

1. 用占位权重 0.004 跑 B1 的 30 步冒烟（见第 4 节）。
2. 运行 `train_metrics.py <log> --ac_calibrate 10`。它取前 10 步的平均值，计算 `FutureVideo_Loss × 0.004 / AC_Loss`，给出 0.5×、1×、2× 三档权重。
3. 试点阶段对 B1 跑这三档，选"VLA loss 不变差、`AC_Shuffle_Gap` 最明显"的一档。B0 用同一个权重。

---

## 4. 执行步骤

### 4.1 本地测试（已完成）

在一个干净的 CPU 环境（torch 2.8、transformers 4.57.3）里：
- `research/model_tests/test_action_cond.py`，13 个测试：
  - 目标公式；
  - 输出形状，以及第一步输出全零；
  - 无效关节和补齐步不影响输出，B0 对动作完全不敏感；
  - 梯度能回到 VLM token，`detach_image_feats` 的效果；
  - 图像 token 取的是顶部相机那一段；
  - 一个小型合成任务上训练 150 步：B1 的 shuffle gap 明显大于 0，B0 恒为 0；
  - 元组长度不变，新 loss 被加进总 loss；
  - 加载器的初始化，只缺一部分参数时报错；
  - 官方 yaml 默认关闭。
- `research/tools/tests/test_pod_configs.py`，4 个测试：配置生成与日志解析。

Triton 和 grouped-GEMM 内核在 macOS 上导入不了，测试的 `conftest.py` 用替身模块绕过。这些测试不调用任何 GPU 内核。完整模型的前向和反向只能在 Pod 上验证（4.2）。

### 4.2 Pod 冒烟（2026-11-16 完成，通过）

结果：
- A0 回归：step-1 的 Depth / Future_Depth / FutureVideo / VLA 四项与 S3b 一字不差。
- B1：30 步无非有限值；step-1 三项蒸馏 loss 与 A0 一致；StepTime 3.38 s（S3b 3.36 s，新 A0 3.60 s，差异是机器波动）。
- 第 1 步 `AC_Loss` = `zero_loss` = 0.6133。到第 30 步，`AC_Loss / zero_loss` 降到 0.987。`AC_Shuffle_Gap` 从 0 单调涨到约 2e-6：head 在学，也开始用动作，只是起步慢（输出层零初始化，Muon 每步更新很小）。
- 权重校准（前 10 步）：1 倍档 3.9e-3，和占位值 0.004 基本一致。
- 注意：日志和 TensorBoard 里的 loss 数值是 bf16 精度（约 0.6 附近每格 0.0039）。`AC_Loss / zero_loss` 要用多步平均来看。gap 本身是在转换前相减的，不受影响。
- TensorBoard 的 `EventAccumulator` 在 Pod 上和 numpy 2 不兼容，要用 `tensorboard.compat.proto.event_pb2` 直接解析事件文件。

原始命令：

代码照旧用 `git archive` 打包上传，然后在 Pod 里：

```bash
source $D/code/lingbot-vla-v2-jepa/research/tools/pod/pod.env
cd $LB
python research/tools/pod/make_train_configs.py smoke --arm B1      # 写出 $D/configs/smoke_*_b1.yaml
python research/tools/pod/make_train_configs.py smoke               # A0 冒烟配置（重写同名文件，内容只多了关闭的 action_cond）

# A0 回归：新代码、关闭 head，写到新目录（旧的 $D/smoke/s3b 里有 checkpoint，开着 enable_resume 会续训）
lb_run 0,1,2,3 29611 $D/smoke/s3b_a0new.log tasks/vla/train_lingbotvla.py $D/configs/smoke_s3b.yaml \
    --train.output_dir $D/smoke/s3b_a0new
smoke_stats $D/smoke/s3b_a0new.log $D/smoke/s3b.log

# B1：新 head
lb_run 0,1,2,3 29612 $D/smoke/s3b_b1.log tasks/vla/train_lingbotvla.py $D/configs/smoke_s3b_b1.yaml
smoke_stats $D/smoke/s3b_b1.log $D/smoke/s3b_a0new.log
grep -m1 "action_cond_head is not in the checkpoint" $D/smoke/s3b_b1.log
python research/tools/pod/train_metrics.py $D/smoke/s3b_b1.log --bin 10 --ac_calibrate 10
```

`$D/smoke/s3b.log` 是此前 S3b 冒烟（同样是 4 卡 × micro batch 64）的日志。如果路径不同，换成实际的那份。

**通过标准（G2）：**
- 日志里出现 `model.action_cond_head is not in the checkpoint; initialised ...`；
- 30 步没有非有限值；
- 第 1 步的 `AC_Loss` 与 `align/action_cond_zero_loss` 相等（输出层零初始化）；
- B1 第 1 步的 `Depth_Loss`、`Future_Depth_Loss`、`FutureVideo_Loss` 与 A0 冒烟一致。`VLA_Loss` 只作参考，因为初始化新 head 多消耗了随机数，噪声和时间步的采样会变；
- StepTime 不超过 A0 的 1.1 倍；
- A0 冒烟的 step-1 各项与 `train_c50_v1` 一致（关闭时代码路径不变）。

### 4.3 试点（阶段三）

- B1 × 3 档权重，B0 × 选定权重，各 seed 42、1 个 epoch（2147 步，约 2.2 小时）。一夜能排 3–4 个。
- 为了一夜排完，B0 先用 1 倍档 0.004。顺序：B1@0.004 → B0@0.004 → B1@0.002 → B1@0.008，最关键的对比最先出结果。B1 最后选了别的档，再补一个同权重的 B0。
- `max_steps` 设成 2147，lr 余弦在 1 个 epoch 内走完，和 `train_c50_v1` 的调度不同。所以 VLA loss 主要和同一批试点里的 B0 比，和 `train_c50_v1` 的对比只作参考。
- 看：`AC_Loss` 相对 `zero_loss` 下降多少；`AC_Shuffle_Gap` 是否稳定大于 0；VLA loss 与 `train_c50_v1` 同步数对比；1 个 epoch 的 checkpoint 冒烟评测（50 个任务 × 2 条 × 2 个设置）。
- 按 2.5 节的提前止损规则决定是否进主消融。

### 4.4 主消融（阶段四）

- B0、B1 各 3 个 seed（42、43、44），A0 用 `train_c50_v1` 和补跑的 43、44；A1 做好以后加 3 个。
- 配方与 `train_c50_v1` 完全一致：4 卡 × micro batch 64 × 累积 1，5 个 epoch（10735 步）。

```bash
python research/tools/pod/make_train_configs.py formal --arm B1 --seed 43 --ac_weight <校准值> \
    --micro_batch 64 --accum 1 --max_steps 10735 --save_steps 2147
```

- 评测协议按总计划 3.4 节，只评最终 checkpoint。

---

## 5. 还没做的事

1. **A1 开关**（`teacher_type: none`）。不能只把视频 loss 权重设成 0：训练脚本会用这个权重做除数，教师前向也还在。要按总计划阶段二第 3 项实现。B0/B1 的试点不依赖它，可以并行。
2. **台账**：阶段一结果、G1 判定、标签检验、本阶段的代码同步与冒烟，都等补充实验做完一起记。
3. **可选的 margin loss**：要求"真动作的误差 < 打乱动作的误差"，让 head 更依赖动作。只在 B1 的 gap 很小时考虑。

---

## 6. 后续可能的扩展

- **B2：JEPA 标签。** 把目标换成 V-JEPA 2 或 2.1 的"未来 − 当前"。前提是细粒度标签检验显示 JEPA 差值明显更好，尤其是在夹爪区域。需要接入 JEPA 教师（总计划阶段二第 1、2 项）。
- **多步 rollout**（V-JEPA 2-AC 的 2 步展开）：要中间帧的目标，解码开销会上升，阶段一已测过成本。
- **推理时利用 head**：例如用它给候选动作块打分。这超出了"只改训练"的范围，留到 B1 有正结果以后再说。

---

## 7. 风险

| 风险 | 对策 |
|---|---|
| head 不用动作（gap ≈ 0）：仅凭画面和状态就能猜出大部分变化 | 试点提前止损；检查权重；考虑 margin loss，或者把图像 token 也 detach，迫使 head 依赖动作和 task token |
| 新 loss 抢走 VLM 容量，VLA loss 变差 | 3 档权重校准；VLA loss 超过 A0 的 1.05 倍就降权 |
| 梯度经图像 token 把 VLM 的视觉表征往 DINO 差值的方向拉，对动作预测未必有利 | 备选 `detach_image_feats: true`，作为阶段五的消融 |
| B1 > B0，但只是因为"多看了动作"让 VLM 记住了训练集的动作分布，泛化不涨 | 主终点包括 randomized 设置；阶段五做学生端探针，看 `Q_{t+T}` 里的动作信息是否增加 |
| 编译后每步多一次 head 前向，StepTime 上升 | 冒烟实测；超出预算就把 `shuffle_metric` 关掉，只在试点里开 |
