# LingBot-VLA 2.0 × JEPA 视觉教师替换：总体阶段性实施计划

- 版本：v0.2（2026-11-16）
- 工作目录：`/Users/hongyizhao/Desktop/具身智能/model/lingbot-vla-v2`（LingBot 官方仓库，`main` @ `be969b8`）
- 参考资料：`embodiedx/`，即此前在 TAIP B300 上的训练、数据转换和评测环境记录。只读引用，`embodiedx/refs/` 下的代码不改
- 阶段一的详细执行计划：[`01-stage1-offline-probe.md`](01-stage1-offline-probe.md)
- 阶段二（改道后）的详细计划：[`02-action-conditioned-prediction.md`](02-action-conditioned-prediction.md)

> **v0.2 改道说明（2026-11-16）。**
> - 阶段一按预注册标准没有通过 G1：JEPA 特征没有在逆动力学探针上显著优于 DINO-Video。报告见 `research/jepa_probe/reports/stage1_report.md`。
> - 改为先做"动作条件预测头"：在 VLM 后加一个训练时才用的 head，输入 VLM token、本体状态和动作块，预测 DINO-Video 的"未来 − 当前"特征差。
> - 新增实验臂 B0（喂零动作）和 B1（喂真动作），主比较是 B1 对 B0。标签的选择依据见 `research/jepa_probe/reports/label_check.md`。
> - A2/A3（换教师、加 JEPA 教师）暂缓；以后作为"JEPA 标签臂"（B2）的前提条件再评估。
> - 下文第 0–5 节保留 v0.1 的原始设计。凡与 B 臂相关的内容，以 `02` 文档为准。

---

## 0. 一页结论

**研究问题。** 把 LingBot-VLA 2.0 后训练中的视频教师 DINO-Video 换成 V-JEPA（2 或 2.1），或者把 V-JEPA 作为额外教师加进来，能否提升 RoboTwin 上的操作成功率？重点看低数据和分布外（randomized）设置。

**核心做法（"视频槽位"替换）。**
- V-JEPA ViT-L 在 256 分辨率下，每个 tubelet 输出 256 个 token × 1024 维，和 DINO-Video 目标的形状完全一样。
- 因此不需要改 prefix 的布局、query 的数量和 head 的输出维度，只换计算目标的教师。
- 新教师包装成与 `DinoVideoTeacher.get_future_feature` 相同的接口，用配置项 `video.teacher_type` 切换。

**四个实验臂。** 全部从预训练的 `lingbot-vla-v2-6b` 出发，后训练配方相同：

| 臂 | 深度教师 | 视频教师 | 含义 |
|---|---|---|---|
| A0 | LingBot-Depth | DINO-Video | 官方配方（基线）。`train_c50_v1` 可直接作为 seed 42 |
| A1 | LingBot-Depth | 无（模块保留，不算 loss） | 对照：后训练阶段的视频蒸馏本身有没有用 |
| A2 | LingBot-Depth | V-JEPA（替换 DINO-Video） | 主假设 |
| A3 | LingBot-Depth | DINO-Video + V-JEPA（新增一个 future head） | 互补假设 |
| B0 | LingBot-Depth | DINO-Video + 动作条件 head（喂零动作） | v0.2 新增：控制"多一个 head、多一项 loss" |
| B1 | LingBot-Depth | DINO-Video + 动作条件 head（喂真动作） | v0.2 主假设：预测"动作造成的画面变化"能改善 VLM 表征 |

v0.2 的执行顺序是 A0、B0、B1 先做，A1 随后补上；A2、A3 暂缓。

**阶段总览：**

| 阶段 | 目标 | 闸门（不过就不进下一阶段） | 估计耗时 |
|---|---|---|---|
| 0 | 环境、数据、评测链路、基线评测、协议冻结 | G0：评测链路能复现官方 checkpoint 的成绩；A0 基线有数字和噪声估计 | 1–2 周（评测链路是瓶颈） |
| 1 | 离线探针：不训练 VLA，先比较几种教师特征的信息量 | G1：至少一种 JEPA 变体在逆动力学探针上显著优于 DINO-Video，或与之互补。**结果：未通过（v0.2 改道）** | 约 1 周（3–5 个夜间 GPU 窗口） |
| 2 | 代码改造。v0.2：动作条件 head（B0/B1，代码与 CPU 测试已完成）、A1 开关、回归测试；v0.1 的教师抽象和 A3 暂缓 | G2：A0 路径与 `train_c50_v1` 的 step-1 loss 一致；吞吐开销可接受（B 臂的标准见 `02` 文档 4.2 节） | 约 1 周（主要在白天写代码，夜间验证） |
| 3 | 试点：每个臂短跑 1–2 epoch，校准 loss 权重 | G3：不发散，VLA loss 不劣于 A0 同步数，冒烟评测不崩 | 2–3 夜 |
| 4 | 主消融：4 臂 × 3 seed，低数据档 | 预注册的统计判定（见 3.4） | 约 11 夜训练 + 评测（见第 4 节） |
| 5 | 机制分析与扩展消融 | 按阶段四的结论选做 | 2–4 周 |
| 6 | 扩展数据档（clean_50 + randomized_500） | 只对阶段四胜出的臂做 | 每个 run 3–4 夜起 |
| 7 | （可选）用 JEPA 教师在开源数据上继续预训练 | 超出 4×B300 的常规预算，作为后续方向 | — |

**必须正视的三条现实约束（来自 `embodiedx` 实测）：**
1. **训练吞吐比之前估的低。** 4×B300、global batch 256 实测 3.36 s/step，约 76 样本/秒。我之前估的 150–300 样本/秒偏乐观，以下预算一律按实测值计算。
   - clean_50 跑 5 个 epoch 约 11.4 小时，正好一夜。
   - 官方完整配方（global 1024 × 50k 步 = 5120 万样本）约 187 小时，按夜间窗口算一个 run 要约 16 夜，所以阶段六必须缩减。
2. **GPU 只在夜间可用。** 10-19 起，`lingbot-train` 每晚 18:30 扩容，运维 09:00 前缩回 0，有效训练约 12 小时/夜。这条现在是否还成立要先确认。
3. **评测链路还没打通。**
   - 已完成：B300 能光追出图，前提是设置 `VK_ICD_FILENAMES` 指向 EGL 版 ICD。
   - 卡在 OIDN 降噪：OIDN 2.0.1 不支持 sm_103，替换到 2.3.3 的脚本还没验收。
   - 未完成：RoboTwin 仿真环境、curobo 的 sm_103 编译和启动器补丁。
   - 没有评测就没有阶段三以后的结论，所以评测链路列为阶段零的关键路径。

---

## 1. 研究问题与假设

| 编号 | 假设 | 用什么检验 |
|---|---|---|
| H1（表征） | V-JEPA 的未来帧特征比 DINO-Video 含更多"动作引起的变化"信息，对机器人位姿也同样敏感 | 阶段一的逆动力学探针、状态探针和目标可预测性 |
| H2（蒸馏效果） | 把 V-JEPA 作为 `Q_{t+T}` 的蒸馏目标，能提高成功率，randomized 设置提升更明显 | 阶段四：A2 对 A0、A2 对 A1 |
| H3（互补） | DINO-Video 与 V-JEPA 的目标互补，双教师优于单教师 | 阶段一的拼接探针；阶段四：A3 对 A0 |
| H0（对照） | 后训练阶段的视频蒸馏本身有贡献 | A0 对 A1。若 A0 ≈ A1，说明后训练里视频教师几乎不起作用，替换教师的收益上限也低，这本身就是重要结论 |
| H4（动作条件预测，v0.2） | 让 VLM 的输出配合动作块去预测"画面会怎么变"，能提高成功率 | 阶段四：B1 对 B0（主），B1 对 A0（辅）；训练中的打乱动作差值 `AC_Shuffle_Gap` 大于 0 |

**解释时要注意：** 预训练的 6B checkpoint 已经被 DINO-Video 蒸馏过，后训练时换教师抹不掉这段历史。A2 测的是"后训练阶段换教师"的效果，不等于"从头用 JEPA 预训练"。只有阶段七能回答后者。A1 是区分这两种解释的关键对照。

---

## 2. 现有机制回顾（只列与改造相关的事实）

详细分析见此前的对话。这里只保留后续改代码要用到的事实。

**Prefix 布局：**
- 3 个相机，每个 64 个图像 token，加 vision start/end 标记。
- 然后是语言 token，最多 72 个。
- 然后是 `Q_t`（8 个 token）和 `Q_{t+T}`（8 个 token）。
- 注意力是 `vlm_causal`，action expert 能看到包括 query 在内的整个 prefix。

**Query 的来源：**
- 4 张表 `depth_align_embs`、`current_video_align_embs`、`future_depth_align_embs`、`future_video_align_embs`，每张是 256×2560 的 `nn.Parameter`。
- 每张表按每组 32 行做均值池化，得到 8 个 token。
- 两张表的结果由 `current_shared_task_proj` / `future_shared_task_proj`（Linear 5120→2560）融合，再按 batch 复制。

**Head：** `TaskTokenResampler`。
- 用同一张表的 256 行作为对齐 query。
- K/V 是顶部相机 64 个 LLM 输出 hidden state + 8 个 query hidden state + latents。
- 每个 head 1 层、4 头 × 32 维、ff_mult 1，输出 1024 维，约 30M 参数。
- future head 会对图像 hidden state 做 detach。

**目标：** 只用顶部相机，取当前帧 t 和未来帧 t+49。
- 深度：MoGe-2 ViT-B 估深度，再由 LingBot-Depth 编码成 16×16×1024 的特征。loss 为 smooth L1 × 0.004，当前和未来各一份。
- 视频：DINO-Video 是 DINOv3 ViT-L/16，配块因果 mask 和 3D-RoPE。
  - clip 为 `[I_t, I_t, I_{t+T}]`，fps = 数据集 fps / 49。
  - 第 1 帧的输出是当前目标，最后一帧是未来目标。
  - loss 为 MSE + 0.2·(1−cos)，当前和未来相加后 × 0.004。

**`train_c50_v1` 末期的各项 loss（台账 #24，可作后续参照）：**
- VLA 0.0132
- Depth 0.24
- Future_Depth 0.34
- FutureVideo 0.026

**关键代码位置：**

| 内容 | 位置 |
|---|---|
| 对齐超参 | `configs/vla/robotwin/robotwin.yaml` 的 `train.align_params` |
| query 构建（`embed_prefix`）、`forward`、总 loss | `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py` |
| `init_depth_heads`、`init_video_heads`、`depth_emb_forward`、`video_emb_forward`、`_video_emb_loss` | `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py` |
| Resampler | `lingbotvla/models/vla/vision_models/align_heads/resampler.py` |
| 教师构建与目标计算：`build_video_model`、`get_video_target`、`get_depth_target` | `lingbotvla/models/vla/vision_models/module_utils.py` |
| DINO-Video 教师 | `lingbotvla/models/vla/vision_models/dino_video/teacher.py` |
| 未来帧的时间偏移、`future_video_effective_fps` | `lingbotvla/data/vla_data/base_dataset.py`：`get_video_delta_timestamps` |
| 在线计算教师目标，按输出元组长度拆包 | `tasks/vla/train_lingbotvla.py` 约 761–801 行 |
| 权重加载：缺失或多余 key 的断言、`_init_parameter` | `lingbotvla/models/module_utils.py` |

**已知的坑，阶段二必须处理：**
1. 不能删 DINO-Video 的模块。否则加载时会多出 key 而报错，query 的输入嵌入也会变。
2. 在 `FlowMatchingV2` 上直接挂裸 `nn.Parameter`，会触发整个模型重新初始化。新增参数必须放进子模块。
3. 后训练加载器对缺失参数做断言，排除列表只在 `adanorm_time` 分支下生效。A3 新增的 head 要有显式的允许缺失列表。
4. 训练脚本按输出元组长度（6/8/10/11）拆包。A1、A3 不能改变元组长度，否则要同步修改拆包逻辑。
5. fps 元数据决定 DINO-Video 的时间 RoPE 步长。V-JEPA 的时间处理方式不同，要明确规定。
6. 换成 ViT-g 教师（1408 维）会改变 head 的输出维度，不在主线范围内，只放在阶段五。

---

## 3. 实验设计

### 3.1 实验臂的实现方式

| 臂 | `video.teacher_type` | 视频 loss | 模块与参数 | 额外开销 |
|---|---|---|---|---|
| A0 | `dino_video` | 当前 + 未来，权重 0.004 | 不变 | 无 |
| A1 | `none` | 0（不构建教师） | DINO-Video 的 query 和 head 全部保留，只是不参与 loss | 省掉教师前向 |
| A2 | `vjepa2` / `vjepa2_1`（阶段一选定） | 当前 + 未来，权重由阶段三校准 | head 结构不变，从预训练权重继续训练 | 教师前向（阶段一实测） |
| A3 | `dino_video` + `jepa_aux` | DINO-Video 原样 + JEPA 未来目标 | 新增 1 个 future JEPA head，放在子模块里。读取同一个 `Q_{t+T}`，不新增 prefix token | 两个教师的前向 + 约 30M 参数的新 head |
| B0 / B1 | `dino_video` | DINO-Video 原样 + 动作条件项：L1 拟合 LN(DINO 未来) − LN(DINO 当前) | 新增 `model.action_cond_head`（约 30M），读顶部相机图像 token、`Q_{t+T}`、状态和动作块。配置 `train.align_params.action_cond`，B0 设 `use_actions: false` | 不增加教师前向；新 head 每步两次前向（一次无梯度的打乱动作诊断） |

A2 的 head 沿用 DINO-Video 的预训练权重继续训练，不重新初始化。理由是两种目标形状相同；如果重新初始化，就混入了"head 从零学"这个额外变量。阶段五可以补一个"重新初始化 head"的消融。

### 3.2 数据档

- **低数据档（主线）。**
  - 训练：clean_50，50 个任务 × 50 条 = 2500 条 episode，549,787 帧。
  - 评测：clean（分布内）和 randomized（分布外）。
  - 数据、训练清单和归一化统计都已在 PVC 上，与 `train_c50_v1` 完全一致。
- **扩展档（阶段六）。** clean_50 + randomized_500，共 27,500 条 episode，约 600 万帧。原始数据还没下载，压缩包约 391 GB，要扩容 PVC。

### 3.3 训练配方（阶段四）

与 `train_c50_v1` 完全一致，只改实验臂相关的配置项：
- 从 `lingbot-vla-v2-6b` 全参数后训练（LingBot v2 没有接入 LoRA）。
- 4 卡 × micro batch 64 × 梯度累积 1 = global batch 256。
- 5 个 epoch，即 max_steps 10,735（每个 epoch 2147 步）。
- 学习率余弦衰减，从 1e-4 到 5e-5。
- 不开图像增强（`image_augment` 默认为 False）。
- 3 个 seed：42（复用 `train_c50_v1`）、43、44。

复用 `train_c50_v1` 作为 A0 seed 42 的前提：阶段二改过的代码在 A0 配置下，step-1 的各项 loss 与它一致（G2）。

### 3.4 评测协议与判定规则（阶段零冻结，之后不改）

**评测设置：**
- 方向性档：每个 checkpoint 评测 50 个任务 × 每个任务 10 条 episode × clean 和 randomized 两个设置，共 1000 条 episode。
- 只评每个 run 的最终 checkpoint（10735 步），预先规定，不挑 checkpoint。
- 所有臂用同一份评测代码，`seed=0`（起始种子 100000），按 (setting, task, seed) 配对。

**还要确定的一项：** 指令类型。官方启动器默认 clean 和 randomized 都用 unseen 指令，与官方公布的成绩同口径。建议沿用，在阶段零写进协议。

**主要终点：**
- randomized 设置下 3 个 seed 的平均成功率；
- clean 设置下的同一指标。

**统计方法：**
- 复用 `research/tools/ab_test.py`：配对的 McNemar 检验（主检验），加上按任务成功率的 Wilcoxon 检验和 bootstrap 置信区间。
- 3 个 seed 的结果合并做配对。另外报告各 seed 分别的结论，看方向是否一致。

**预注册的比较：**

| 比较 | 回答的假设 |
|---|---|
| A2 对 A0 | H2 |
| A2 对 A1 | V-JEPA 教师是否本身有用 |
| A0 对 A1 | H0 |
| A3 对 A0 | H3 |
| B1 对 B0（v0.2 主比较） | H4 |
| B1 对 A0 | H4 的净收益 |

判定沿用 `ab_test.py` 的 ADOPT / REJECT / INCONCLUSIVE 规则。如果阶段零测出的评测噪声表明 10 条 episode 不够，就在阶段零按功效分析调高到 20 条，然后再冻结协议。

---

## 4. 资源、约束与预算

### 4.1 资源

| 资源 | 现状 | 来源 |
|---|---|---|
| TAIP `lingbot-train` | 4 张 B300（268 GiB，sm_103），夜间可用；无外网，可走 Pod 代理（8 连接约 28 MB/s），可用 hf-mirror | 台账 #12；训练 runbook 第 1 节 |
| TAIP `lingbot-data` | 仅 CPU，32 核；与训练 Pod 不能同时满配额运行 | 训练 runbook 第 3 节 |
| 4090 | 有外网，白天可用；用于构建镜像、下载和打包 | 训练 runbook 第 0 节 |
| PVC `/mnt/work` | 1.5T；10-24 已用 780G。PVC 个数配额 5/5 已满 | 台账 #23 |
| 训练镜像 | `<平台镜像仓库>/lingbotvla-train:20261017-be969b8-r1`（完整地址见 `embodiedx` 的训练 runbook），含 torch 2.8 cu129、transformers 4.57.3、lerobot 0.4.2、av、torchcodec；sm_103 内核已验证 | 台账 #1、#13 |
| 权重 | `/mnt/work/weights/`：`lingbot-vla-v2-6b`、`Qwen3-VL-4B-Instruct`、`moge-2-vitb-normal`，sha256 已核对 | 台账 #15 前后 |

### 4.2 单次训练的成本（实测）

- **A0：** 10,735 步 × 3.36 s ≈ 10.0 小时，加上 9 次周期保存（每次约 7 分钟）和启动，约 11.4 小时，正好一夜。
- **A2：** 增量是 V-JEPA ViT-L 前向一次（一个 4 帧 clip，2 个 tubelet），替换掉 DINO-Video 的前向，预计与 A0 相当。阶段一实测。
- **A3：** 两个教师加一个新 head，预计比 A0 慢 5–15%，可能要跨到第二夜。阶段一、二实测。
- **A1：** 省掉一次教师前向，略快。
- **存储：** 完整 checkpoint 约 74 GB，其中 `hf_ckpt` 23.8 GiB。主消融每个 run 只保留最终的 `hf_ckpt`，加上一个用于续训的完整 checkpoint（跑完即删）。11 个 run 约 11 × 24 ≈ 260 GB 的 `hf_ckpt`。
- **当前 PVC 余量（约 730G）勉强够。** 要清理 `train_c50_v1` 的中间 `hf_ckpt`，或者申请扩容。阶段零确认。

### 4.3 阶段四的总预算

| 项目 | 数量 | 夜数 |
|---|---|---|
| 训练 | 4 臂 × 3 seed − 1（复用 `train_c50_v1`）= 11 个 run | 约 11–13 夜（A3 可能各需 1.2 夜） |
| 评测 | 12 个最终 checkpoint × 1000 条 episode | 取决于阶段零实测的评测吞吐。如果 B300 一夜能评 1–2 个 checkpoint，需要 6–12 夜；4090 能在白天并行评测的话可以减半 |

合计约 4–6 周。如果预算紧张，先把每个臂做 2 个 seed（8 个 run），结论不明确的比较再补第 3 个 seed。

---

## 5. 各阶段计划

### 阶段 0：环境、数据、评测链路、基线、协议冻结

**目标：** 所有后续实验需要的基础设施都就绪，基线有可信的数字。

**任务：**
1. **工作区。** 主分支是 `main`，每个阶段的开发在功能分支上做，例如 `stage1-probe`。`embodiedx/` 已经写进 `.gitignore`。按第 7 节规定目录。
2. **平台确认**（问运维）：
   - 夜间窗口是否还在，有没有可能白天也开放；
   - GPU 卡数配额；
   - PVC 余量和扩容方式（PVC 个数配额已满，只能扩现有卷或删卷再建）；
   - 有没有把 checkpoint 从 TAIP 传到 4090 的正规通道。
3. **代码同步通道。**
   - 代码托管在个人 GitHub 的私有仓库 `lingbot-vla-v2-jepa`。官方仓库作为 `upstream` 远端，用来同步官方更新。
   - 训练机上不放 GitHub 凭证。每次同步时，在本地用 `git archive` 打包要部署的提交（几 MB），通过 PVC 页面上传到卷里的 `embodiedx/code/`（Pod 里是 `$D/code/`），解压到 `$D/code/lingbot-vla-v2-jepa`，并把 commit 和 sha256 记入台账。`/mnt/vla` 在两个 Pod 里都是只读挂载，不能往里解压。
   - Pod 里 `source research/tools/pod/pod.env`，本仓库会排在 `PYTHONPATH` 最前面，覆盖镜像里固定在 `be969b8` 的 `/opt/lingbot-vla-v2`。不需要重新打镜像。
4. **评测链路 E1**（沿用 `embodiedx/plans/2026-10-22-robotwin-eval-env.plan.md`）：
   - 在 `$D/envs/RoboTwin` 建仿真环境：Python 3.10，torch 2.8 cu129，sapien 3.0.0b1，mplib 0.2.1，curobo v0.7.8 按 sm_103 编译，两处 `sed` 补丁照做；
   - 运行 `oidn_swap.sh` 并验收：重跑 `rt+oidn` 时不再出现 `OIDN Error`，图像噪声明显低于 `rt+none`；
   - 启动器补丁：episode 数可指定，`instruction_type` 显式传入，任务清单可指定，输出可以被 `parse_eval_run.py` 解析。
   - B300 验收不过，就转路线 B：4090、BF16，只做 checkpoint 之间的相对比较。
5. **评测链路校准（G0 的一半）。** 用官方 `lingbot-vla-v2-6b-robotwin` 权重（clean 与 randomized 混训），在方向性档上复现官方成绩（93.52 / 92.80）。差距在约 3 个百分点以内，就认为链路正确。这个权重只做环境校验，不进训练、不进选择。
6. **A0 基线评测。**
   - 对 `train_c50_v1` 的最终 checkpoint（10735 步）跑方向性档；
   - 同一个 checkpoint 换一组评测 seed 再跑一次，估计评测噪声；
   - 用这两个数据做功效分析，决定每个任务的 episode 数是 10 还是 20。
7. **冻结协议。** 把 3.4 节填完整，写成 `plan/protocol-eval.md`，记入台账。
8. **数据完整性。** 检查 clean_50 的 LeRobot 数据集、`train_c50.txt`、`robotwin_c50.json`（sha256 `424112547ad9dd…`）在 PVC 上都没变。

**产出：** 可用的评测链路；校准报告；A0 seed 42 的成绩与噪声估计；冻结的评测协议；平台确认清单。

**闸门 G0：** 评测链路校准通过；A0 seed 42 有成绩；功效分析表明预算内可以检测到约 5 个百分点的差异，检测不到就在这里重新定预算。

**说明：** 阶段一不依赖评测链路，可以与阶段零的第 4–7 项并行。

### 阶段 1：离线探针（详见 `01-stage1-offline-probe.md`）

**目标：** 在不训练 VLA 的前提下回答三个问题：
- 哪种 JEPA 特征最值得作为蒸馏目标（模型版本、clip 构造、层、分辨率）；
- 它是否比 DINO-Video 含更多动作相关信息（H1），两者是否互补（H3）；
- 每样本的前向开销是多少。

**产出：** 探针结果表；选定的 JEPA 配置（写进阶段二的配置）；教师吞吐实测；一份可在阶段二直接复用的 JEPA 教师包装代码。

**闸门 G1：** 见阶段一文档第 7 节。不通过就不做 A2/A3 的训练，转为"负结果 + A1 对照"的小结，或者换方案（例如改用 JEPA 的 predictor 损失，而不是替换教师）。

### 阶段 2：代码改造

**v0.2 状态：**
- 动作条件 head（B0/B1）的代码、配置开关、加载器的新模块初始化、训练日志项、配置生成和 CPU 单元测试都已完成。改动清单和 Pod 冒烟步骤见 `02` 文档第 3、4 节。2026-11-16 的 Pod 冒烟通过：A0 回归一致，B1 稳定、无额外开销，head 在学；1 倍档权重约 0.004。2026-11-17 的 1 个 epoch 试点（B0@0.004，B1@0.002/0.004/0.008）：都稳定，VLA loss 不受影响；B1 用上了动作，但同权重下预测误差和 B0 几乎一样，原因很可能是 head 上下文里的未来 task token 已经带着未来画面的预测。暂不进主消融。已加 `context: image`（head 只读当前图像 token、状态和动作），第二轮试点待跑，见 `02` 文档 4.3 节。
- 下面第 3 项（A1 开关）仍要做。第 1、2、4、5 项属于换教师路线，暂缓。

**v0.1 的任务：**
1. 新增 `lingbotvla/models/vla/vision_models/jepa/`。
   - 实现 `VJepaTeacher`，有两个后端：HF 的 `VJEPA2Model`（V-JEPA 2）和官方仓库的 vendored 代码（V-JEPA 2.1）。
   - 接口与 `DinoVideoTeacher.get_future_feature(video, return_cls, return_current, current_index, fps)` 相同，返回 `(future_patches[B,256,1024], current_patches[B,256,1024])`。
   - 阶段一写的包装直接迁移过来。
2. `build_video_model` 和 `get_video_target` 按 `video.teacher_type` 分派。新增配置项：`jepa_ckpt_path`、`jepa_layer`、`jepa_input_size`、`jepa_clip_mode`、`jepa_num_frames`。
3. **A1 开关：** `teacher_type: none` 时不构建教师，视频 loss 返回 0，query 和 head 照常参与前向。输出元组长度不变。
4. **A3：**
   - 新增 `jepa_future_head`（`TaskTokenResampler`），放在子模块里，对齐 query 从 `future_video_align_embs` 复制初始化；
   - 新增独立的 loss 权重；
   - 加载器加显式的允许缺失前缀，只放行 `jepa_` 开头的 key；
   - 更新训练脚本里的拆包和日志项，新增 `FutureJepa_Loss`。
5. **数据：** 如果阶段一选了多帧 clip（例如 8 帧），扩展 `get_video_delta_timestamps`，加入中间帧，并实测解码开销。`agx_run.py` 的补丁只加速了非图像列，中间帧仍要解码 PNG。
6. **测试：**
   - 单元测试：形状、dtype，以及教师输出与阶段一离线特征一致；
   - A0 回归：用 `smoke_stats` 比较 step-1 的 loss 与 `train_c50_v1` / S3b 冒烟，各项相对差 < 2%；
   - 保存、加载和续训的往返测试；
   - 推理路径 `sample_actions` 不调用教师，确认推理与 A0 完全一致；
   - 4 卡吞吐：每个臂跑 30 步冒烟，记录 StepTime 和显存。

**闸门 G2：**
- A0 回归通过；
- A1、A2、A3 的 30 步冒烟没有非有限值；
- A2 的 StepTime 不超过 A0 的 1.1 倍，A3 不超过 1.25 倍。

### 阶段 3：试点

**任务：**
1. A1、A2、A3 各用 seed 42 跑 1 个 epoch（2147 步，约 2.2 小时），一夜可以排 3–4 个。
2. **校准 loss 权重。** JEPA 目标的数值尺度和 DINO-Video 不同。
   - 用阶段一统计的 token 方差，把权重换算到"初始加权 loss 与 A0 的 FutureVideo 项量级相同"；
   - 同时跑一个原样 0.004 的版本作对照，共 2 档。
3. 对照 `train_c50_v1` 在 2147 步时的 VLA loss（`metrics_by_500.csv`）。
4. 对 1 个 epoch 的 checkpoint 做冒烟评测：50 个任务 × 2 条 × 2 个设置。

**闸门 G3：**
- 没有发散；
- JEPA loss 单调下降；
- VLA loss 不高于 A0 同步数的 1.05 倍；
- 冒烟评测成功率不低于 A0 同步数 10 个百分点以上；
- 选定 loss 权重。

### 阶段 4：主消融

按第 3 节执行：11 个 run，每个 run 评测最终 checkpoint，按预注册规则判定。中途不看评测结果去改配方：发现 bug 就整组重跑，并记入台账。

**产出：** 主结果表（4 臂 × 3 seed × 2 个设置），以及对 H0、H2、H3 的判定。

### 阶段 5：机制分析与扩展消融（按阶段四结论选做）

- **学生端探针：** 用阶段一的探针，读取已训练 checkpoint 中 `Q_{t+T}` 和 future head 的输出，比较各臂学到的表征里动作信息的多少。
- **推理时的 query 消融：** 把 `Q_t` / `Q_{t+T}` 置零或打乱，看成功率下降多少，衡量 action expert 对 query 的依赖。
- **注意力分析：** action expert 对 query token 的注意力占比。
- **扩展消融：**
  - loss 权重扫描；
  - JEPA 的层和 clip 变体；
  - head 重新初始化；
  - ViT-g 教师（输出 1408 维，要改 head）；
  - 数据效率曲线（每个任务 10、25、50 条 episode）。

### 阶段 6：扩展数据档

**数据：**
- 下载 randomized_500（约 391 GB），沿用方案 B：在 Pod 里用 `mirror_fetch.sh` 经 hf-mirror 下载，用 pi0 转换器补丁转换，再从 v2.1 转成 v3.0；
- 重算归一化统计；
- 先扩容 PVC。

**训练：**
- 只做 A0 和阶段四胜出的臂；
- 固定预算，例如 1000 万样本，约 37 小时，3–4 夜；
- 官方完整配方每个 run 约 16 夜，只在结论非常值得时考虑。

### 阶段 7（可选）：继续预训练

用 JEPA 教师在开源大规模数据上继续预训练 6B 模型，以回答"预训练遗留"的问题。按 76 样本/秒算，每 1 亿样本约 15 天，超出当前预算，只作为后续方向记录。

---

## 6. 可复用的资产

`embodiedx/` 只留在本地，不进代码仓库：里面有内部平台信息、第三方仓库和本地环境。可复用的工具已经去掉内部信息、迁移到 `research/tools/`，测试一并迁移；其余内容需要时到 `embodiedx/` 查阅。

**已迁移到本仓库：**

| 资产 | 位置 | 用在哪 | 说明 |
|---|---|---|---|
| Pod 环境与启动函数 | `research/tools/pod/pod.env`：`lb_run`、`smoke_stats`、`step_wall` | 阶段 1–6 | 由 `taip.env` 改来：`LB` 默认取 pod.env 所在的仓库目录（即同步到 PVC 的本仓库，`$D/code/lingbot-vla-v2-jepa`），并放在 `PYTHONPATH` 最前面，覆盖镜像里的旧代码；各变量都可以先 export 覆盖 |
| 数据读取补丁 | `research/tools/pod/agx_run.py` | 阶段 1–6 | 训练必须带上（`lb_run` 已经内置）；阶段一读标签也照它的做法，不解码图像列 |
| 配置生成 | `research/tools/pod/make_train_configs.py` | 阶段 2–4 | 已有 `--arm A0/B0/B1`、`--seed`、`--ac_weight`；A1 做好后再加；`--teacher_type` 随 A2/A3 暂缓 |
| 训练指标解析 | `research/tools/pod/train_metrics.py` | 阶段 3–4 | 已解析 `AC_Loss`、`AC_Shuffle_Gap`，`--ac_calibrate` 给出权重建议；`FutureJepa_Loss` 列随 A3 暂缓 |
| 评测解析与 A/B 判定 | `research/tools/parse_eval_run.py`、`research/tools/ab_test.py` | 阶段 0、3–6 | 新增 `--instruction_types`（默认 `clean=seen,randomized=unseen`），阶段零冻结协议时按协议设置 |
| 台账 | `research/tools/ledger.py` | 全部阶段 | 默认写到 `research/ledger/` |

**仍在 `embodiedx/`（本地参考）：**

| 资产 | 位置 | 用在哪 | 说明 |
|---|---|---|---|
| 训练镜像 | `embodiedx/docker/lingbotvla/`（含 Dockerfile、`build_image.sh`、`check_image.sh`） | 全部阶段 | 尽量不重新打镜像；需要新依赖时装到 PVC 上的 overlay 目录 |
| 夜间自动停止 | PVC 上的 `$D/bin/stop_after_save.sh`（台账 #12、#20）；仓库里的 `train.stop_step`（`make_train_configs.py --stop_step`） | 阶段 3–6 | 脚本已在跨夜续训中验证过。`stop_step` 在训练代码内部停下并存盘，学习率调度不变，续训时如果已越过会报错；还没在 Pod 上实测，用法见 `02` 文档 4.4 节 |
| 渲染探针与 OIDN 替换 | `embodiedx/taip/rt_probe.py`、`embodiedx/taip/oidn_swap.sh` | 阶段 0 | OIDN 替换待验收 |
| 下载与校验 | `embodiedx/tools/hf_manifest.py`、`verify_downloads.py`、`extract_archives.py`、`mirror_fetch.sh`（训练 runbook 5.9） | 阶段 6 | hf-mirror 的 API 会卡住，按清单用 curl 直接下载；阶段六用到时再迁移 |
| 数据转换链路 | `embodiedx/patches/robotwin-13c3c47-pi0-convert-deterministic.patch`，训练 runbook 5.3–5.5 | 阶段 6 | v2.1 → v3.0 必须转换 |
| 平台操作说明 | `embodiedx/plans/` 下的各 runbook、`embodiedx/gpu/` | 全部阶段 | 含内部地址，不进仓库 |
| 现成的数据与统计 | `/mnt/work/lerobot`（clean_50 v3.0）、`/mnt/work/lists/train_c50.txt`、`/mnt/work/lingbot_cfg/` | 阶段 1–5 | 与 `train_c50_v1` 完全一致 |
| 现成的基线 | `/mnt/work/runs/train_c50_v1`（最终 checkpoint 10735 步，以及各 epoch 末的 `hf_ckpt`） | 阶段 0、4 | 作为 A0 seed 42，loss 曲线可作参照 |
| 权重清单 | `embodiedx/data/manifests/*.json` | 阶段 0 | 重新核对权重时用 |

---

## 7. 工程规范

**仓库目录：**
```
lingbot-vla-v2-jepa/          # 个人 GitHub 私有仓库；upstream = Robbyant/lingbot-vla-v2
├── plan/                     # 计划文档（本目录）
├── research/
│   ├── jepa_probe/           # 阶段一的探针脚本
│   ├── model_tests/          # 模型改动的 CPU 单元测试（阶段二起）
│   ├── tools/                # 评测解析、A/B 判定、台账（从 embodiedx 迁移）
│   │   └── pod/              # Pod 环境、数据读取补丁、配置生成、训练指标
│   └── ledger/               # 本项目台账（ledger.jsonl + ledger.md）
├── lingbotvla/models/vla/vision_models/jepa/   # 阶段一写、阶段二接入的 JEPA 教师
└── embodiedx/                # 本地参考，已写进 .gitignore，不提交
```

**PVC 目录（PVC `lingbot-vla`：子目录 `embodiedx` 可写，挂载为 `$D=/mnt/work`；整个卷另外只读挂载为 `/mnt/vla`）：**
```
$D/code/lingbot-vla-v2-jepa/  # 同步上来的分支代码（每次同步记 sha256）
$D/weights/vjepa/             # V-JEPA 2 与 2.1 的权重和官方代码快照
$D/jepa/probe/                # 阶段一：样本索引、离线特征、探针结果
$D/runs/jepa_<臂>_s<seed>/    # 阶段三、四的训练 run
$D/reports/jepa/              # 汇总报告
```

**命名：** run 名为 `jepa_<A0|A1|A2|A3|B0|B1>_s<seed>[_<变体>]`。配置文件与 run 同名，放在 `$D/configs/`。

**台账：** 每次数据变更、训练、评测、A/B 判定和决策都记一条，并附上配置、清单和结果文件的 sha256。

**沿用的约束：**
- 训练机不装 AgenticX，不放任何 LLM 密钥；
- 训练时设置 `HF_HUB_OFFLINE=1`；
- 只有下载时临时打开代理；
- 不修改 `embodiedx/refs/`。

---

## 8. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 评测链路在 B300 上打不通（OIDN 或 curobo 卡住） | 阶段三以后都没有结论 | 尽早做 E1；不行就转路线 B（4090、BF16、相对比较），并先解决 checkpoint 的传输通道 |
| 评测噪声太大，10 条 episode 测不出差异 | 主消融结论 INCONCLUSIVE | 阶段零做功效分析，必要时提高到 20 条，或者减少臂数（先做 A0、A1、A2） |
| 后训练里视频教师几乎不起作用（A0 ≈ A1） | 替换教师的收益上限低 | 这本身是结论；可以转向"在学生端加 JEPA 式预测损失"，或阶段七 |
| JEPA 特征在探针上不优于 DINO-Video | G1 不通过 | 不进入 A2/A3 训练，节省约 8 夜；把探针结果整理成负结果 |
| JEPA 目标尺度不同，导致 loss 失衡 | 训练不稳定或者没效果 | 阶段一统计目标方差，阶段三校准权重 |
| 多帧 clip 让数据读取变慢 | 训练吞吐下降 | 阶段一先测解码开销；优先选 2 帧变体（与现有数据管线完全兼容） |
| PVC 空间不足 | run 无法保存 | 每个 run 只留最终的 `hf_ckpt`；阶段零申请扩容 |
| 夜间窗口被占用或缩短 | 进度延后 | `stop_after_save.sh` 加自动续训；把短任务（探针、冒烟）排在同一夜的空隙里 |
| V-JEPA 2.1 依赖 `timm`，镜像里没有 | 阶段一卡住 | 用 `pip install --target $D/pylib` 装进 PVC overlay，或者只 vendor 用到的 `drop_path` |

---

## 9. 待确认事项

1. 夜间窗口（18:30–09:00）现在是否还在，有没有可能整天使用。（运维）
2. PVC 余量与扩容方式；能否新增 PVC（个数配额已满）。（运维）
3. 从 TAIP 导出 checkpoint 到 4090 的正规通道。备选是用 `oras` 推到镜像仓库。（运维）
4. 评测的指令类型：沿用官方默认（全部 unseen），还是 clean 用 seen、randomized 用 unseen。（你来定；建议沿用官方默认，便于和官方成绩对照）
5. 这个研究的产出形式（论文、内部报告或比赛），决定阶段五做到多深、阶段六是否必做。（你来定）

---

## 10. 粗略时间线（假设每周 5–7 个夜间窗口）

| 周 | 夜间（GPU） | 白天（CPU、4090、写代码） |
|---|---|---|
| W1 | 阶段一：教师权重就位，提取特征 | 阶段零：E1 仿真环境、启动器补丁；阶段一：写探针脚本 |
| W2 | 阶段一：剩余特征与吞吐测量；阶段零：评测链路校准，A0 基线评测 | 阶段一：训练探针、出报告，过 G1；阶段二：写代码 |
| W3 | 阶段二：冒烟与回归；阶段三：试点 | 阶段二、三的分析；冻结协议 |
| W4–W8 | 阶段四：11 个 run 加评测 | 评测解析、统计、阶段五准备 |
| W9 以后 | 阶段五、六 | 写作 |
