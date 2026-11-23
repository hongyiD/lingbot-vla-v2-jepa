# 评测链路：在 B300 上打通 RoboTwin 闭环评测

关联文档：
- 总计划 `plan/00-overall-plan.md`：阶段零第 4–7 项、3.4 节评测协议、闸门 G0。
- 10 月的评测环境计划和实测：`embodiedx/plans/2026-10-22-robotwin-eval-env.plan.md`，`embodiedx` 台账 #25–#28。

## 0. 为什么现在做

A0（`train_c50_v1`）、B1（`jepa_B1_s42`）、A2（`jepa_A2_s42`）三个完整 checkpoint 的训练 loss 各有长短，已经分不出高下（台账 #12、#20）：
- B1 主任务不变，当前深度差 3.1%；
- A2 当前深度好 0.4%，主任务差 3.3%。

只能靠成功率判定。2026-11-23 决定先打通评测链路，暂停新的训练臂。

## 1. 10 月已经确认的事实

| 事实 | 出处 |
|---|---|
| B300 能用 SAPIEN 光追出图，前提是 `VK_ICD_FILENAMES=/mnt/work/envs/nvidia_egl_icd.json`（EGL 版 ICD），而且 Pod 要设 `NVIDIA_DRIVER_CAPABILITIES=all` | embodiedx 台账 #26–#28 |
| 光追每个控制步（3 台相机）约 42 ms，渲染不是瓶颈 | 同上 |
| sapien 3.0.0b1 自带的 OIDN 2.0.1 不支持 SM 10.3：报 `OIDN Error: unsupported device type: CUDA`，出图没降噪，和 `rt+none` 完全一样。原探针只查"不是空白图"，所以误判为 PASS | 台账 #28 |
| OIDN 2.3.3 含 sm_100/sm_120 的 kernel，C API 在大版本 2 内向后兼容；`oidn_swap.sh` 把 2.0.1 的两个文件名软链接到 2.3.3，可 `--restore` | 台账 #28 |
| RoboTwin `13c3c47` 已在 `$D/robotwin_eval`，资产（background_texture、embodiments、objects）已下载解压 | 台账 #25 |
| 官方启动器起 `num_gpus × num_per_gpu` 个推理服务，仿真任务按队列分配；客户端 `test_num = 100` 写死，指令类型取自 `policy/ACT/deploy_policy.yml`（`unseen`） | `experiment/robotwin/` |
| 仿真依赖：Python 3.10，`sapien==3.0.0b1`、`mplib==0.2.1`、`open3d==0.18.0`、`scipy==1.10.1`，curobo `v0.7.8` 可编辑安装；`_install.sh` 对 sapien 和 mplib 各有一处 `sed` 修改，会改变规划行为，必须照做 | RoboTwin `13c3c47` `script/` |

## 2. 步骤

| 步骤 | 内容 | 资源 | 通过标准 |
|---|---|---|---|
| E1a | OIDN 替换验收：先在未替换的 venv 上跑新探针，确认它能识别出降噪失败；再替换成 2.3.3 重跑 | 1 张卡，约 10 分钟 | 替换前 FAIL；替换后 `RT_PROBE PASS`，没有 `OIDN Error`，每台相机的噪声比都 ≤ 0.7 |
| E1b | 环境摸底：conda、Python 3.10、各下载源能否连通、nvcc、CPU、内存、磁盘、`robotwin_eval` 和资产的状态 | CPU，几分钟 | 拿到写建环境脚本需要的信息 |
| E1c | 在 `$D/envs/RoboTwin` 建仿真环境；curobo 按 sm_103 编译；两处 `sed` 照做并断言各命中一次；仿真环境里也跑一次 `oidn_swap.sh` | CPU 为主，约 1 小时 | 能 import sapien、mplib、open3d、curobo；curobo 的 `.so` 含 sm_103 |
| E1d | 启动器和客户端补丁：episode 数、指令类型、任务清单可以指定；输出能被 `parse_eval_run.py` 解析 | 本地 | 本地测试通过 |
| E1e | 冒烟：1 个任务 × 10 条 episode，FP32 | 1 张卡 | 有 `Success rate` 行；记录每条 episode 的耗时 |
| G0 | 校准：官方 `lingbot-vla-v2-6b-robotwin` 在方向性档上复现官方成绩（93.52 / 92.80），差距 ≤ 3 个百分点；A0 评两次估评测噪声 | 4 卡 | 见总计划 3.4 节 |

E1a 和 E1b 互不依赖，可以同时做。

## 3. 风险

- **OIDN 2.3.3 的降噪结果和 2.0.1 可能略有差别。** 训练数据是用 2.0.1 渲染的。差别再大，也比完全不降噪小得多。在 4090 上可以用 2.0.1 和 2.3.3 渲染同一帧做对比，量化这个差别。
- **sm_100 的 kernel 在 SM 10.3 上能不能跑，没有实测过。** 跑不了就退回路线 B：在 4090 上用 BF16 评测，只做 checkpoint 之间的相对比较。
- **curobo 按 sm_103 编译，需要 CUDA 12.9 的 nvcc。** 镜像里有 `/opt/cuda-ptxas-12.9`，但运行时头文件可能要另外凑。

## 4. 进度记录

| 日期 | 事项 | 结果 |
|---|---|---|
| 2026-11-23 | 决定先打通评测链路（台账 #21） | — |
| 2026-11-23 | 探针和 `oidn_swap.sh` 迁入 `research/tools/pod/`。探针新增两条判据：没有 `OIDN Error`；每台相机 `rt+oidn` 的噪声（灰度图 4 邻域拉普拉斯绝对值的均值）≤ 0.7 × `rt+none` | 本地测试通过 |
| — | E1a、E1b | 待执行 |
