# 阶段一：离线探针 —— 详细任务规划与执行步骤

- 版本：v0.1（2026-11-10）
- 上级文档：[`00-overall-plan.md`](00-overall-plan.md)（阶段 1，闸门 G1）
- 代码位置：`research/jepa_probe/`（探针脚本）；`lingbotvla/models/vla/vision_models/jepa/`（JEPA 教师包装，阶段二直接接入训练）
- 运行位置：TAIP。`lingbot-train` 夜间用 1 张 B300；`lingbot-data` 白天跑 CPU 任务。数据、权重和结果都放在 PVC `$D/jepa/`

---

## 0. 目标

不训练 VLA，只用冻结的教师模型和小型探针，回答下面四个问题：

| # | 问题 | 对应的探针 |
|---|---|---|
| Q1 | 在 LingBot 实际使用的"当前帧 + 49 帧后未来帧"构造下，V-JEPA 的特征比 DINO-Video 含更多**动作相关**信息吗？（H1） | P1 逆动力学、P1s 逆动力学 + 本体状态 |
| Q2 | V-JEPA 的特征丢失了**位姿信息**吗？位姿信息是 DINO-Video 和深度目标的强项 | P2 状态探针 |
| Q3 | 作为 `Q_{t+T}` 的蒸馏目标，学生只看当前帧时，这个目标有多少可以预测？其中有多少是"动作带来的变化"？ | P3 目标可预测性 |
| Q4 | DINO-Video 和 V-JEPA 互补吗？（H3） | P4 拼接探针 |

另外还要定下阶段二要用的 **JEPA 配置**：模型版本、clip 构造、层、分辨率；并实测**教师开销**（P5）。

**不在阶段一做的：**

- 不改训练代码；
- 不训练 VLA；
- 不做仿真评测；
- 不下载 randomized 数据（鲁棒性探针列为可选的 1b，见第 9 节）。

---

## 1. 前提与已有资产

| 需要的东西 | 现状 | 来源 |
|---|---|---|
| clean_50 数据（LeRobot v3.0，图像模式，PNG 存在 parquet 里，640×480） | 已在 `/mnt/work/lerobot`：50 个数据集，2500 条 episode，549,787 帧 | 台账 #15 前后，训练 runbook 5.9 |
| 训练清单 / 机器人配置 / 归一化统计 | `/mnt/work/lists/train_c50.txt`、`/mnt/work/lingbot_cfg/robot_configs/robotwin_c50.yaml`、`/mnt/work/lingbot_cfg/norm_stats/robotwin_c50.json`（sha256 `424112547ad9dd…`） | 台账 #20 |
| DINO-Video 教师 | `/mnt/work/weights/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth`、`config.yaml` | 训练 runbook 7.2 |
| 训练镜像 | `lingbotvla-train:20261017-be969b8-r1`：torch 2.8 cu129、**transformers 4.57.3**（已含 `VJEPA2Model`）、lerobot 0.4.2、datasets 3.6、einops、av、torchcodec；**没有 timm** | `requirements.txt`，台账 #1 |
| Pod 环境 | `source /mnt/work/code/lingbot-vla-v2-jepa/research/tools/pod/pod.env`：变量 `D`、`W`、`CFG`、`R`、`HF_HOME`；函数 `lb_run`。`LB` 自动取 pod.env 所在的仓库目录 | `research/tools/pod/pod.env` |
| 读数据时不解码图片的做法 | `agx_run.py` 的 `_non_image_view`：`hf_dataset.remove_columns(图像列)` | `research/tools/pod/agx_run.py` |
| 用 curl 按清单下载 | `mirror_fetch.sh`，配合 `tools/hf_manifest.py`、`tools/verify_downloads.py` | 训练 runbook 5.9 |
| 台账工具 | `research/tools/ledger.py`，默认写到 `research/ledger/` | — |

**来自现有数据的关键常数：**

- 每条 episode 平均约 220 帧。
- 数据集 fps 预期为 50，以各数据集 `meta/info.json` 为准，脚本里读取，不写死。
- 动作块长度 `chunk_size = 50`；未来帧偏移 `(chunk_size − 1)/fps`，即第 t+49 帧。
- 动作 14 维：两臂各 6 个关节加 1 个夹爪；状态同为 14 维。
- 归一化方式 `bounds_99_woclip`：用 q01/q99 映射到 [−1, 1]，不截断。
- 顶部相机键：`observation.images.cam_high`（LingBot 内部叫 `camera_top`）。

---

## 2. 实验设计

### 2.1 样本构造（与 LingBot 训练时一致）

1. **只用顶部相机。** 当前帧取第 t 帧，未来帧取第 t+49 帧。只保留 t+49 ≤ L−1 的 t（L 为 episode 长度），避开数据加载器在 episode 末尾的补齐。
2. **图像预处理链** 与训练完全一致：
   - PNG 解码，得到 640×480 的 uint8 图像；
   - `Resize((256, 256))`，即 LingBot 数据集里的 `image_transforms=Resize(image_size)`，会压扁纵横比。尺寸来自 `img_size`：训练脚本的 `MyDataArguments` 默认 256，`robotwin.yaml` 没有改它；
   - 转回 uint8 后，教师侧再用 `F.interpolate` 双线性缩放到教师的输入尺寸（256 时不缩放，V3h 为 384），然后做 ImageNet 均值方差归一化。这一步与 `module_utils.get_video_target` 相同。
   - 所有教师看到的像素完全一样。**不用** HF 自带的 `AutoVideoProcessor`：它先放大再中心裁剪，会让不同教师看到的像素内容不同。
3. **采样：**
   - 每条 episode 从 t=0 开始，每 10 帧取一个 t；
   - 预计每条 episode 约 17 个样本，共约 4.3 万个。
4. **按 episode 划分，所有变体共用同一份划分：**
   | 划分 | 每个任务的 episode 编号 | 条数 | 样本数（约） |
    |---|---|---|---|
    | 训练 | 0–34 | 1750 | 3.0 万 |
    | 验证 | 35–39 | 250 | 0.43 万 |
    | 测试 | 40–49 | 500 | 0.86 万 |

    验证集只用来选学习率和做早停；所有结论都看测试集。
5. **"关键时刻"子集：** 动作块内夹爪开合状态发生变化的样本，即抓取和放下前后。P1/P1s 另外单独报告这个子集上的误差。这些时刻对成功率最关键，也最需要"预见未来"的信息。

### 2.2 教师变体

所有变体都输出当前和未来两组 16×16×1024 的 token，与 DINO-Video 的目标槽位同形状。

| 代号 | 教师 | 输入尺寸 | clip 构造（f_k 表示第 k 帧） | 当前目标 | 未来目标 | 层 |
|---|---|---|---|---|---|---|
| **V0** | DINO-Video（LingBot 原版） | 256 | `[f_t, f_t, f_{t+49}]`，fps 取视频配置里的 `effective_fps`（1.0）：训练 batch 里没有 `future_video_effective_fps`，`convert_features` 会把它丢掉 | 第 1 帧 | 第 2 帧 | 最后一层 + norm。直接调用仓库里的 `get_video_target` |
| **V1** | V-JEPA 2 ViT-L（HF） | 256 | `[f_t, f_t, f_{t+49}, f_{t+49}]`，2 个 tubelet，一次前向 | tubelet 0 | tubelet 1 | `last_hidden_state`（含最终 LayerNorm） |
| **V1i** | 同上 | 256 | 当前与未来**分别**前向：`[f_t, f_t]`、`[f_{t+49}, f_{t+49}]` | 单独前向的结果 | 单独前向的结果 | 同上 |
| **V1m** | 同上 | 256 | 与 V1 同一次前向 | tubelet 0 | tubelet 1 | 第 16 层（共 24 层）hidden state，再做不带仿射参数的 LayerNorm |
| **V2** | 同上 | 256 | 8 帧：`[f_t, f_t, f_{t+8}, f_{t+16}, f_{t+24}, f_{t+33}, f_{t+41}, f_{t+49}]`，4 个 tubelet | tubelet 0（只含 f_t，不泄露未来） | tubelet 3 | 最后一层 |
| **V3** | V-JEPA 2.1 ViT-L（官方代码） | 256 | 同 V1（RoPE 自动插值） | tubelet 0 | tubelet 1 | 最后一层 + norm |
| **V3h** | 同上 | 384（256 放大到 384） | 同 V1；输出 24×24 个 token，再按面积插值到 16×16 | tubelet 0 | tubelet 1 | 同上 |
| V4（可选） | DINOv3 ViT-L/16（逐帧） | 256 | 单帧 | f_t | f_{t+49} | 最后一层 |

**设计说明：**

- **V1 和 V1i 对比：** 看"当前帧和未来帧放进同一个 clip 联合编码"有没有好处。DINO-Video 用的是联合编码（块因果）；V-JEPA 是双向注意力，联合编码时未来 token 也能看到当前帧。
- **V2 的用途：** 测试多帧中间运动信息的价值。代价是训练时数据管线每个样本要多解码 5 帧（7 张不同的帧减去训练已解码的 2 张；P5 实测）。即使 V2 最好，也要和它带来的成本一起权衡。
- **V3h 的用途：** V-JEPA 2.1 是在 384 分辨率上训练的。看在原生分辨率下取特征再降采样，是否比直接用 256 好。
- **时间尺度：** V-JEPA 的时间 RoPE 只按 tubelet 序号编码，2 个 tubelet 的 clip 会被模型当成相邻约 0.5 秒（预训练约 4 fps）。实际间隔约 1 秒，两者有偏差，但对所有 V-JEPA 变体都一样，所以不额外处理。
- **V4 默认不做。** 只有在 DINOv3 的 gated 权重能拿到，而且需要解释"DINO-Video 的视频训练本身带来了多少"时才做。

### 2.3 探针任务

| 代号           | 输入                            | 标签（都已按 `robotwin_c50.json` 的 `bounds_99_woclip` 归一化） | 指标                                                  |
| ------------ | ----------------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| **P1**       | 当前 token + 未来 token           | 动作块 `a[t:t+50]`（50×14）                               | 测试集 nMSE（= MSE / 测试集标签方差，即 1 − R²）；手臂和夹爪分别报告        |
| **P1s**（主指标） | P1 的输入 + `state[t]`           | 同 P1                                                 | 同 P1。衡量**本体状态之外**的视觉信息：action expert 本来就能拿到状态       |
| P1Δ          | 当前 token + 未来 token           | `state[t+49] − state[t]`                             | nMSE。去掉绝对位姿，只看变化                                    |
| **P2**       | 只用当前 token                    | `state[t]`（14 维）                                     | nMSE                                                |
| **P3**       | 见下                            | 未来 token（降到 4×4 池化，再用 PCA 降到 256 维）                  | R²cur、R²cur+act、ΔR²                                 |
| **P4**       | V0 与 Vj 的 token 按通道拼接（2048 维） | 同 P1s                                                | nMSE，与两者各自的结果比较                                     |
| **P5**       | —                             | —                                                    | 教师前向的耗时和显存（batch 64，bf16，1 张 B300）；V2 多解码的帧的 CPU 耗时 |

**P3 的定义与解读：**

- **做法：** 用闭式 ridge 回归预测未来 token 的 PCA 表示，有两种输入：
  - (a) 只用当前 token；
  - (b) 当前 token + 整个动作块。
- **ΔR² = R²cur+act − R²cur：** 未来目标里有多少是"由动作决定的变化"。学生预测 `Q_{t+T}` 时看不到未来的动作，ΔR² 越大，说明蒸馏这个目标越会逼着模型在内部"想象动作的后果"。
- **R²cur+act 过低：** 说明目标里有大量既无法从当前帧推出、也和动作无关的成分，对学生来说是噪声。
- **用途：** P3 只作辅助证据，不进闸门。

**对照基线（所有探针都要跑，用来确认探针本身是否灵敏）：**

- **B-mean：** 直接预测训练集均值，nMSE 应接近 1。
- **B-state：** 用 `state[t]` + 任务 one-hot，经 MLP 预测动作块。这是 P1s 必须明显超过的下限。
- **B-pixel：** 两帧 RGB 各缩到 16×16 当作 token，送进同一个探针。用来回答"至少比原始像素好吗"。

### 2.4 探针结构与训练协议

所有变体用**完全相同**的探针结构和超参搜索范围。

**输入：**

- 每帧把 16×16 的 token 按 2×2 平均池化成 8×8，共 64 个 token，以 fp16 存盘。
- 每个变体按通道做 z-score 标准化（统计量只用训练集），消除教师之间的尺度差异。

**注意力探针（P1、P1s、P1Δ、P2、P4）：**

- `LayerNorm → Linear(C→256)`，加上可学习的 8×8 位置嵌入和流类型嵌入（当前 / 未来）。
- P1s 再加一个状态 token：`Linear(14→256)`。
- 2 层 Transformer 编码器：d=256，4 头，MLP 为 4 倍宽，dropout 0.1。
- 1 个可学习 query 做交叉注意力池化，再经 MLP 输出 50×14 或 14 维。
- 参数量约 2M。

**训练：**

- 损失为归一化空间里的 MSE；AdamW，wd 0.05，batch 256，最多 40 个 epoch，余弦学习率。
- 按验证集 nMSE 早停，耐心值 5 个 epoch。
- 学习率在 {3e-4, 1e-3} 中按验证集选。
- 3 个随机种子，报告测试集的均值 ± 标准差。

**线性探针（参照）：**

- 当前和未来 token 各自全局平均池化后拼接（2048 维），用闭式 ridge 回归，λ 在验证集上选。
- 目的是确认结论不依赖探针结构。

**统计：**

- 两个变体比较时，按测试集 episode 做配对 bootstrap（10,000 次）：每条 episode 的 nMSE 差，取 3 个种子的平均。
- 给出相对差异的 95% 置信区间。

**算力估计：**

- 每个变体的特征文件约 4.3 万 × 2 × 64 × 1024 × 2 B ≈ 11 GB，可整体放进 GPU 显存。
- 一次探针训练约 1–3 分钟。
- 全部探针约 4 种任务 × 10 种输入 × 2 个学习率 × 3 个种子 ≈ 240 次，单卡约 6–8 小时，4 卡并行不到 2 小时。

---

## 3. 准备工作

### T1.0 代码骨架与本地测试（Mac，白天）

1. 从 `main` 拉出分支 `stage1-probe`。
2. 建目录：
   ```
    research/jepa_probe/
      build_index.py       # 样本索引 + 标签（只读非图像列）
      frames.py            # 读顶部相机帧，并按 LingBot 的方式做 256 缩放
      extract_features.py  # 按变体提取特征，分片保存，可断点续跑
      train_probe.py       # 注意力探针（P1/P1s/P1Δ/P2/P4，以及 B-* 基线）
      ridge_probe.py       # 线性探针与 P3
      teachers.py          # 各变体的教师封装：训练同款 uint8 帧进，当前 / 未来 token 出
      selfcheck_teacher.py # T1.2：教师加载与形状自检
      check_pipeline.py    # 步骤 1 自检与步骤 2：标签、像素、V0 目标与 LingBot 训练数据路径逐项对照
      bench_teacher.py     # P5：教师前向和解码耗时
      report.py            # 汇总表格、bootstrap 置信区间、PCA 可视化
      configs/variants.yaml
      configs/sweep_main.yaml
      tests/
    lingbotvla/models/vla/vision_models/jepa/
      __init__.py
      teacher.py           # VJepaTeacher：hf_vjepa2 / vjepa2_1 两个后端，接口与 DinoVideoTeacher 一致
   ```
3. 本地单元测试，只用 CPU：
   - 参照 `research/tools/tests/helpers.py` 造一个小的假 LeRobot v3.0 数据集，测 `build_index` 的划分、标签归一化和 t+49 边界；
   - 测 `frames.py` 的缩放与 LingBot 的 `Resize` 一致；
   - 用随机初始化的小 ViT 测 `teacher.py` 的形状契约：输入 `[B, 3, T, H, W]`，输出 `(future[B, 256, 1024], current[B, 256, 1024])`。
4. 打包上传（训练机上不放 GitHub 凭证，所以不在 Pod 里 `git clone`）：
   - 本地执行 `git archive --format=tar.gz -o lingbot-$(git rev-parse --short HEAD).tar.gz HEAD`。`embodiedx/` 被 `.gitignore` 排除，不会打进包里；
   - 包只有几十 MB，从 PVC 页面上传到卷里的 `embodiedx/code/`，即 Pod 里的 `/mnt/work/code/`（卷根目录在 Pod 里只读挂载为 `/mnt/vla`，可以从那里读包，但不能往里解压）；
   - 解压到 `/mnt/work/code/lingbot-vla-v2-jepa`，记录 sha256 和 commit（`export LB_COMMIT=<commit>`，`extract_features` 会写进 `meta.json`）。

### T1.1 教师权重与依赖就位（Pod，白天用 `lingbot-data`，经代理）

代理由 `lingbot-data` 的 Deployment 预先设好（大小写两套变量和 `no_proxy`），不要再 `export` 覆盖。地址不写进仓库。先测连通性：

```bash
source /mnt/work/code/lingbot-vla-v2-jepa/research/tools/pod/pod.env
env | grep -iE '^(https?_proxy|no_proxy)='    # 应看到 4 个代理变量和 2 个 no_proxy
V=$D/weights/vjepa; mkdir -p $V
for u in https://hf-mirror.com/facebook/vjepa2-vitl-fpc64-256/resolve/b3c1679b7c34d3255ef3547f27c7b226aefab26f/config.json \
         https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
         https://codeload.github.com/facebookresearch/vjepa2/tar.gz/204698b45b3712590f06245fbfba32d3be539812; do
  curl -sSL -m 60 -o /dev/null -r 0-0 -w "http=%{http_code} proxy_connect=%{http_connect} $u\n" "$u"
done
```

判据：`proxy_connect=200`；前两个返回 206 或 200，codeload 现场打包、不支持按范围请求，返回 200 即正常。11-11 实测三个都通（200 / 206 / 200）。

| 物件 | 来源（固定版本） | 大小 | 校验 |
|---|---|---|---|
| V-JEPA 2 ViT-L（HF） | `facebook/vjepa2-vitl-fpc64-256` @ `b3c1679b7c34d3255ef3547f27c7b226aefab26f`：只下 `config.json`、`model.safetensors`、`video_preprocessor_config.json`，**不下** `original/model.pth`（5.1 GB） | 1.30 GB | `model.safetensors` 的 sha256 为 `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b` |
| V-JEPA 2.1 ViT-L | `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt` | 5,151,198,524 B | 下完后在本地算 sha256 并记入台账 |
| V-JEPA 官方代码快照 | GitHub `facebookresearch/vjepa2` @ `204698b45b3712590f06245fbfba32d3be539812`（2026-03-23），tar.gz | 很小 | 记 sha256 |
| timm（2.1 的代码只用到 `timm.models.layers.drop_path`） | `pip install --no-deps --target $D/pylib timm==1.0.15` | 很小 | — |

**下载命令：**

- HF 文件：hf-mirror 的 API 曾经卡住不返回（训练 runbook 5.9），所以**不用** `hf download`，用 curl 按上表路径直接下载到 `$V/vjepa2-vitl-fpc64-256/`。下完用 `sha256sum` 对照。
- 2.1 权重：用 `curl -C - --retry 5 -o $V/vjepa2_1_vitl_dist_vitG_384.pt <url>` 断点续传。单连接 3–5 MB/s，约 20–30 分钟。想更快可以用 4 段 `-r` 分段并行下载后再拼接。
- 代码快照：解压到 `$V/vjepa2-src/`。

**下载后的处理：**

1. **抽出 2.1 的编码器权重。** 只保留 `ema_encoder` 这个键（hubconf 里 `vjepa2_1_vit_large_384` 用的就是它），去掉 `module.` / `backbone.` 前缀，存成 `$V/vjepa2_1_vitl_384_ema_encoder.pt`（约 1.2 GB），记 sha256。原始的 5.1 GB 文件可以留作备份。
2. **注意官方 hub 代码的坑。** 在固定的这个提交里，`src/hub/backbones.py` 把 `VJEPA_BASE_URL` 改成了 `http://localhost:8300`（"for testing"），所以 `pretrained=True` 会失败。我们的包装统一用 `pretrained=False` 建模型，再 `load_state_dict(strict=True)` 加载本地权重。
3. **`PYTHONPATH` 约定：** `pod.env` 已经把本仓库放在最前面，不需要再手动追加。2.1 的代码快照由 `VJepaTeacher.from_vjepa2_1` 加进 `sys.path`；`$D/pylib`（timm）写在 `variants.yaml` 中 V3/V3h 的 `extra_paths` 里，由脚本追加到 `sys.path` 末尾，不会覆盖镜像自带的包。
4. **如果 `dl.fbaipublicfiles.com` 经代理连不通：**
   - 先问运维走大文件通道（4090 下载后交给运维放进 PVC）；
   - 暂时拿不到的话，阶段一先只做 V0、V1、V1i、V1m、V2，V3 和 V3h 等补上权重后追加。闸门可以先按 V-JEPA 2 判定，在台账里写明。
5. 权重下完后 `unset` 代理变量，后续运行都带离线变量（`HF_HUB_OFFLINE=1` 等）。

### T1.2 教师加载自检（Pod，夜间 1 张 GPU，约 15 分钟）

1. **V0：** 用 `build_video_model`（读 `robotwin_c50` 训练配置里的 `align_params.video` 段）建 DINO-Video，送入随机 clip `[2, 3, 3, 256, 256]`。确认 `get_future_feature` 返回两个 `[2, 256, 1024]`，没有 NaN。
2. **V1：**
   - `VJEPA2Model.from_pretrained($V/vjepa2-vitl-fpc64-256, torch_dtype=bfloat16)`，送入 `pixel_values_videos` `[B, T=4, 3, 256, 256]`，用 `skip_predictor=True`；
   - 确认 `last_hidden_state` 的形状是 `[B, 2×256, 1024]`；
   - 确认我们包装输出的 tubelet 0 和 tubelet 1 分别等于 `last_hidden_state[:, :256]` 和 `[:, 256:]`。
3. **V3：**
   - 用官方的 `app.vjepa_2_1.models.vision_transformer.vit_large`，参数照抄 hubconf 里的 `_make_vjepa2_1_model`（`img_size=384`、`use_rope=True`、`interpolate_rope=True`、`img_temporal_dim_size=1` 等）；
   - 加载抽出的 `ema_encoder`，`strict=True`；
   - 分别以 256 和 384 输入，确认 token 数是 512（2×16×16）和 1152（2×24×24）。
4. 结果写入 `$R/jepa/s1_teacher_selfcheck.txt`，记一条台账。

---

## 4. 执行步骤

### 步骤 1：建样本索引和标签（`lingbot-data`，白天，CPU）

```bash
source /mnt/work/code/lingbot-vla-v2-jepa/research/tools/pod/pod.env   # 已把本仓库放到 PYTHONPATH 最前面
export OMP_NUM_THREADS=1
P=$D/jepa/probe; mkdir -p $P
python -m research.jepa_probe.build_index \
  --train_list $D/lists/train_c50.txt \
  --robot_config $CFG/robot_configs/robotwin_c50.yaml \
  --norm_stats $CFG/norm_stats/robotwin_c50.json \
  --stride 10 --future_offset 49 --chunk 50 \
  --split "train=0-34,val=35-39,test=40-49" \
  --out $P/index
```

**产出：**

- `index.parquet`，每个样本一行：`sample_id, task, task_id, dataset_dir, episode_index, t, row_t, row_future, rows_v2[8], split, fps, gripper_event`；
- `labels.npz`：`state_t[14]`、`state_t49[14]`、`action_chunk[50,14]`（已归一化）、`task_id`；
- `index_meta.json`：每个数据集的 fps、样本数，以及输入文件的 sha256。

**做法：** 只读非图像列，照 `agx_run.py` 的做法用 `hf_dataset.remove_columns(相机列)`。整体只需几分钟。

`--train_list` 每行是 `robotwin_c50 <数据集目录>`，脚本取最后一列。

**自检（CPU，约 10 分钟）：**

```bash
cd $LB
python -m research.jepa_probe.check_pipeline data --index $P/index --samples 50 --out $R/jepa/s1_check_data.json
```

随机抽 50 个样本，用 LingBot 自己的 `VLADataset`（`robotwin_c50` 机器人配置、训练配置的 `data` 段、`agx_run.py` 补丁）构造训练样本，逐项比较。每项打印一行 PASS/FAIL，有失败时退出码为 1：

- **labels：** 归一化后的动作块和状态与 `labels.npz` 一致，最大绝对误差 ≤ 1e-5（float32 舍入，实测约 2e-7）；
- **rows：** episode、帧号对得上，动作块里没有补齐的步；
- **pixels：** 训练送进教师的 `pil_images` / `future_pil_images`（顶部相机，排第 0 位）与 `frames.py` 的输出 uint8 逐位相等。这是步骤 2 的第一条判据，白天在 CPU 上就能先验掉；
- **fps：** 训练样本里没有 `future_video_effective_fps`，所以 DINO-Video 用配置里的 `effective_fps`，V0 提取也照此不传 fps；
- **config：** `variants.yaml` 里 V0 的视频参数与训练配置一致，机器人配置指向的 norm stats 与建索引时用的是同一个文件。

`build_index` 打印的各划分样本数和夹爪事件比例记入台账。

### 步骤 2：确认 DINO-Video 目标与训练管线一致（夜间，1 张 GPU，约 20 分钟）

这一步决定探针测的是不是**训练时实际用的目标**，必须先通过。

```bash
cd $LB
python -m research.jepa_probe.check_pipeline video --index $P/index --batches 4 --batch 16 --out $R/jepa/s1_check_video.json
```

1. 从训练集抽 4 个 batch，每个 16 个样本，用与步骤 1 自检相同的方式得到训练的 `pil_images`，像训练循环那样调用 `get_video_target`，得到目标 A。
2. 对同样的样本，用 `extract_features` 的取帧代码（`ShardFrames`，内部是 `frames.py`）和 V0 的封装计算目标 B。
3. 判据：
   - 输入像素 `pil_images` 与 `frames.py` 的输出完全相同（uint8 逐位相等）；
   - A 与 B 按 token 计算的余弦相似度 ≥ 0.999，相对 L2 误差 ≤ 1%（误差来自 bf16）。
4. 不通过就停下排查（常见原因是缩放的插值方式、antialias 或 dtype 的顺序不同），不要继续。

### 步骤 3：提特征（4 张 GPU，约 30–60 分钟）

```bash
cd $LB
VS=V0,B-pixel,V1,V1i,V1m,V2,V3,V3h
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python -m research.jepa_probe.extract_features \
    --variant $VS --variants_cfg research/jepa_probe/configs/variants.yaml \
    --index $P/index --features $P/features --part $g/4 \
    --batch 64 --num_workers 16 --dtype bf16 --viz 1000 > $P/logs/extract_g$g.log 2>&1 &
done
wait
```

- **一次读帧、所有变体共用：** 瓶颈是读 parquet 和解 PNG，不是教师前向。parquet 按整个 row group 读，单个分片实际要读约 10 倍的帧；如果每个变体各跑一遍，同一批帧要读 8 次，4 个进程还会抢 PVC 带宽（实测每个分片从 34 秒拖到 4–5 分钟）。所以每个分片只读、只解码一次，再依次送进每个还缺这个分片的教师。
- **`--part k/n`：** 进程 k 只负责编号 k、k+n、k+2n…的分片，4 张卡之间不会重复做同一个分片。每个变体由最后补齐它的进程写 `stats.npz` 和 `meta.json`（先写临时文件再改名）。
- **先小规模试跑：** 加 `--max_shards 1` 只跑一个分片（2048 个样本），看耗时和输出形状，再去掉这个参数跑完。已完成的分片会自动跳过，单变体也可以用 `--variant V0 --out $P/features/V0`。
- **B-pixel** 也作为一个"变体"提取（两帧各缩到 16×16，按 2×2 组成 64 个 12 维 token），供探针基线使用。
- **V0** 用 `variants.yaml` 里抄自训练配置的 `video` 段，DINO 权重在 `${W}/lingbot-vla-v2-6b/dino_video/`。
- **输出：** `features/<v>/shard_*.npz`，每个样本含 `cur[64,1024]` 和 `fut[64,1024]`（fp16）；外加：
  - `stats.npz`：训练集上逐通道的均值和方差，以及整体 token 的方差。阶段三要用它来换算 loss 权重；
  - 可视化样本（每个任务第一条测试 episode 的全部样本，约 850 个）的完整 16×16 token，存在对应分片的 `viz_ids` / `viz_cur` / `viz_fut` 里；
  - `meta.json`：变体配置、代码 commit（取自 `LB_COMMIT`）、权重的 sha256、耗时。
- **断点续跑：** 已完成的分片会跳过；分片先写临时文件再改名，中途被杀不会留下半个分片。
- **耗时估计：** 每个分片读一次全部 7 个位置（V2 需要），8 个教师的前向合计约 2 秒/batch（见步骤 4 的 bench）；22 个分片分给 4 张卡，每张卡 5–6 个分片。
- **磁盘：** 7 个变体约 77 GB。先确认 PVC 余量 ≥ 150 GB（`df -h $D`）。

### 步骤 4：P5 开销测量（夜间，与步骤 3 同一晚，约 20 分钟）

```bash
CUDA_VISIBLE_DEVICES=1 python -m research.jepa_probe.bench_teacher \
  --variants V0,V1,V1i,V2,V3,V3h --variants_cfg research/jepa_probe/configs/variants.yaml \
  --index $P/index --batch 64 --iters 50 --warmup 10 --dtype bf16 --compile \
  --out $R/jepa/s1_bench.json
```

- **教师侧：** 报告每个 batch 64 的前向耗时中位数和 p90、峰值显存；加 `--compile` 时同时测 `torch.compile` 之后的耗时（编译失败只记录错误，不影响其余结果）。计时包含缩放和归一化，与训练时一致。
- **数据侧：** 测单进程解码并缩放一张 PNG 的耗时。V2 每个样本多出 5 帧，据此算出这些额外解码占每张卡 8 个 worker 的比例，判断数据加载会不会跟不上。参照：训练全局约 76 样本/秒。
- **换算：** 按 3.36 s/step、micro batch 64 计算 StepTime 的增加。基线已经包含 DINO-Video，所以分两种：替换（A2）增加的是与 V0 的差值；两者都要（A3）增加的是完整的前向耗时。

### 步骤 5：训练探针（夜间 4 张 GPU 并行，或白天用 CPU 跑线性探针和 P3）

```bash
# 注意力探针：任务 × 输入变体 × 学习率 × 种子，按 GPU 分组并行
python -m research.jepa_probe.train_probe --sweep research/jepa_probe/configs/sweep_main.yaml \
  --features $P/features --index $P/index --out $P/results --gpus 0,1,2,3
# 线性探针与 P3（闭式解，CPU 或 GPU 都可以）
python -m research.jepa_probe.ridge_probe --variants V0,B-pixel,V1,V1i,V1m,V2,V3,V3h \
  --features $P/features --index $P/index --out $P/results/ridge
```

- 没有提完特征的输入会被跳过并打印提示，所以 V3/V3h 暂时缺席也能先跑其余部分。
- 已有结果文件的运行会跳过，中断后重跑同一条命令即可续上。每张卡的日志在 `$P/results/_jobs/`。
- P4 在主 sweep 跑完后自动进行：按 P1s 验证集 nMSE 选出最好的两个 JEPA 变体，与 V0 拼接后再训练。

**`sweep_main.yaml` 的内容：**

- 任务：P1、P1s、P1Δ、P2；
- 输入：B-mean、B-state、B-pixel、V0、V1、V1i、V1m、V2、V3、V3h；
- P4：V0⊕（P1s 最好的两个 JEPA 变体）；
- 学习率 {3e-4, 1e-3}，种子 {0, 1, 2}。

每次运行保存 `results/<task>/<input>/lr<lr>_s<seed>.json`：

- 验证集和测试集的 nMSE（总体、手臂、夹爪、关键时刻子集）；
- 每条测试 episode 的误差（供 bootstrap 使用）；
- 训练曲线。

### 步骤 6：汇总报告（白天）

```bash
python -m research.jepa_probe.report --results $P/results --bench $R/jepa/s1_bench.json \
  --features $P/features --index $P/index --out $R/jepa/stage1_report.md
```

PCA 图写在报告旁边的 `stage1_report_viz/` 目录里。

报告内容：

1. **主表：** 每种输入在 P1s、P1、P1Δ、P2 上的测试 nMSE（均值 ± 标准差），以及相对 V0 的变化和 95% 置信区间。
2. **关键时刻子集**上的同一张表。
3. **P3：** 每个变体的 R²cur、R²cur+act、ΔR²。
4. **P4：** 拼接相对单独 V0 的增益。
5. **P5：** 开销表。
6. **可视化：** viz1000 中几条 episode 的 PCA 着色图（前三个主成分映射成 RGB），V0 与 V1/V3 并排，看时间一致性和对机械臂、物体的关注。
7. **线性探针与注意力探针的排序是否一致**（Spearman 相关）。

把报告文件下载到本地，放到 `research/jepa_probe/reports/`，与代码一起提交。

---

## 5. 时间安排（以"夜"为单位，假设夜间窗口不变）

| 时段 | 任务 | 机器 |
|---|---|---|
| 第 1–2 天（白天） | T1.0 写代码和本地测试，打包上传 | Mac |
| 第 2 天（白天） | T1.1 下载权重和依赖；步骤 1 建索引 | `lingbot-data` |
| 第 2 晚 | T1.2 自检；步骤 2 一致性验证；步骤 3 提取 V0、V1、V1i、V1m；步骤 4 测开销 | `lingbot-train`，1–2 张卡 |
| 第 3 天（白天） | 线性探针和 P3（CPU）；检查特征统计 | `lingbot-data` |
| 第 3 晚 | 步骤 3 提取 V2、V3、V3h；步骤 5 跑注意力探针 | `lingbot-train`，4 张卡 |
| 第 4 天（白天） | 步骤 6 汇总报告，G1 评审，写阶段二配置 | Mac |
| 第 4 晚 | 机动：补跑、或者追加 V4 | — |

阶段一每晚只占 GPU 2–4 小时，其余时间可以给阶段零的评测链路验证用。两者错开：评测在前半夜，探针在后半夜。

---

## 6. 产出物

| 产出 | 位置 |
|---|---|
| JEPA 教师包装（阶段二直接用） | `lingbotvla/models/vla/vision_models/jepa/teacher.py` |
| 探针代码与测试 | `research/jepa_probe/` |
| 教师权重和代码快照，以及 sha256 | `$D/weights/vjepa/` |
| 样本索引与标签 | `$D/jepa/probe/index/` |
| 离线特征（含训练集统计） | `$D/jepa/probe/features/<变体>/` |
| 探针结果 | `$D/jepa/probe/results/` |
| 自检、一致性、开销报告 | `$R/jepa/s1_teacher_selfcheck.txt`、`s1_check_data.json`、`s1_check_video.json`、`s1_bench.json` |
| 阶段一总报告 | `$R/jepa/stage1_report.md`，同时复制到 `research/jepa_probe/reports/` |
| 阶段二的冻结配置 | `plan/02-stage2-config-freeze.md`，记录选定变体的 `teacher_type`、权重、层、输入尺寸、clip 构造、目标统计量和开销 |

---

## 7. 闸门 G1（预先登记，看结果前冻结）

**结果有效的前提**（任何一条不满足，先修，再看后面的判据）：

1. 步骤 2 的一致性验证通过。
2. 探针灵敏：B-mean 的 nMSE ≈ 1；V0 在 P1s 上明显好于 B-state，bootstrap 置信区间不含 0。
3. 线性探针与注意力探针的变体排序大体一致（Spearman ρ ≥ 0.6）。不一致时以注意力探针为准，但要在报告里讨论。

**判据：** 主指标为 P1s 测试 nMSE。每个变体在验证集上选学习率，取 3 个种子的平均。

| 结论 | 条件 |
|---|---|
| **G1-A：支持替换（A2）** | 存在 JEPA 变体 J，同时满足：① `nMSE_P1s(J) ≤ 0.95 × nMSE_P1s(V0)`，相对差的 95% 置信区间不含 0；② `nMSE_P2(J) ≤ 1.10 × nMSE_P2(V0)`，即位姿信息没有明显变差；③ 按 P5 换算，J 让 StepTime 增加 ≤ 10% |
| **G1-B：支持互补（A3）** | `nMSE_P1s(V0⊕J) ≤ 0.95 × nMSE_P1s(V0)`，置信区间不含 0；P5 换算后 StepTime 增加 ≤ 25% |
| 辅助证据（不作闸门） | P3 的 ΔR²(J) ≥ ΔR²(V0)；P1 和关键时刻子集上的方向与 P1s 一致 |

**根据判据决定后续：**

| G1-A | G1-B | 阶段四做哪些臂 |
|---|---|---|
| 通过 | 通过 | A0、A1、A2、A3 全做 |
| 通过 | 不通过 | A0、A1、A2；A3 降为阶段五的可选项 |
| 不通过 | 通过 | A0、A1、A3；A2 只做 1 个 seed 作参考 |
| 不通过 | 不通过 | 不做 A2/A3 训练。阶段四只做 A0 对 A1，回答 H0；阶段一结果写成负结果；重新评估方向，例如在学生端加 JEPA 式的预测损失 |

**选哪个 JEPA 变体：**

1. 在满足 G1-A 的变体里选 P1s 最低的。
2. 如果几个变体的差异在置信区间以内，依次优先：
   - 2 帧变体（与现有数据管线完全兼容）；
   - V-JEPA 2 的 HF 后端（依赖最少）；
   - 开销更低的。

---

## 8. 风险与应对

| 风险 | 应对 |
|---|---|
| 代理连不上 `dl.fbaipublicfiles.com` 或 GitHub | 先按 T1.1 测 206；连不上就走运维的大文件通道；暂时只用 V-JEPA 2 判定，2.1 之后追加 |
| `timm` 装不上 | 在 `$D/pylib/timm/models/layers.py` 放一个只有 `drop_path` 的极小实现（推理时 drop_path 为 0，行为不变），并写进报告 |
| 一致性验证不通过 | 逐个对齐：PNG 解码、Resize 的插值和 antialias 设置、uint8 → float 的顺序、ImageNet 归一化、clip 的拼接顺序和 fps；仍不通过就直接调用 LingBot 的 dataloader 来提 V0 特征 |
| 探针饱和：所有变体都接近 B-state | 换主指标为 P1Δ 或关键时刻子集；把采样步长加密到 5；降低池化（16×16 全分辨率，只做 P1s） |
| 不同教师的 token 方差差别大，影响探针 | 已经按通道做 z-score；另外报告不做标准化的线性探针结果作对照 |
| 特征文件太大 | 只保留 8×8 池化的 token；viz1000 单独存；原始 16×16 token 不全量保存 |
| `lingbot-data` 与 `lingbot-train` 的 CPU 配额冲突（两者 limit 合计超额） | 白天训练 Pod 本来就缩到 0；夜间只用训练 Pod |

---

## 9. 可选扩展 1b：鲁棒性探针（不阻塞 G1）

如果 G1 的结论不明确，或者要提前看分布外表现：

1. 下载 5 个任务的 randomized_500 压缩包，约 40 GB，沿用训练 runbook 5.9 的流程。
2. 按 clean_50 的方式转换：pi0 补丁转成 v2.1，再转 v3.0。
3. 用在 clean 上训练好的探针直接在 randomized 样本上测试（不重新训练），比较各变体的 nMSE 退化幅度。

退化越小，说明这种教师特征对杂物、纹理、光照越不敏感，对应 randomized 设置下可能的收益。

---

## 10. 台账条目（每条都附上相关文件的 sha256）

| 类型 | 标题示例 |
|---|---|
| data | `S1 V-JEPA weights on PVC`（V-JEPA 2 的 HF 权重、2.1 原始与抽出的编码器、代码快照、timm） |
| data | `S1 probe index built`（样本数、划分、fps、夹爪事件比例；index 和 labels 的 sha256） |
| run | `S1 teacher self-check` |
| run | `S1 V0 consistency vs training pipeline`（余弦相似度和相对误差） |
| run | `S1 features <变体>`（耗时、磁盘占用、`meta.json` 的 sha256） |
| run | `S1 teacher bench`（P5 结果） |
| eval | `S1 probe results`（主表摘要和报告的 sha256） |
| decision | `G1: <通过/不通过>, 选定 <变体>`（列出判据数值和阶段四做哪些臂） |
