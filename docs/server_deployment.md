# 服务器部署与运行

1. 解压后进入`SemConsFL_Semantic_Utility_Probe`，创建Python3.11环境，安装服务器CUDA驱动对应的torch/torchvision及requirements。
2. 设置`export CIFAR100_ROOT=/data/cifar100_raw`。该目录包含`cifar-100-python/train`、`test`、`meta`。脚本不下载，不重新生成client split。
3. 执行`bash scripts/check_environment.sh`，然后`bash scripts/check_gpu1.sh`，确认物理GPU1存在和其他进程占用。
4. 执行`bash scripts/run_two_gpu1.sh`：自动先做短程显存测量，再启动A/B各50轮。若资源不足会退出，不隐式串行。也可只先执行`bash scripts/validate_gpu1.sh`了解资源；正式并行入口仍会重新检查当时空闲量。
5. 显式串行选择`bash scripts/run_serial_gpu1.sh`。独立运行A/B脚本见README。
6. `tail -f logs/original_*.log`及`logs/fixed_*.log`查看输出；每组`status.json`保存当前完成轮数；`error.log`保留失败堆栈。脚本自身保存两个PID和exit code。Ctrl+C仅终止本次shell仍运行的child jobs，不杀其他GPU进程。
7. 两组完成后`bash scripts/analyze_results.sh`。`results/comparison/report.md`汇总，JSON包含完整统计和不足样本标识。保留集不被分析程序调用。

所有输出目录已存在即拒绝覆盖，失败后也一样。新尝试需明确指定新目录，默认不自动retry/resume。`latest_state.pt`包含private/Anchor/history以保留边界，不宣称当前有无缝恢复API；不要只加载Full然后称连续训练。

压缩包排除本地验证的大checkpoint和临时结果，但保留`docs/local_validation/`的简短真实日志、摘要和比较结果。服务器两个50轮输出重新独立产生，绝不继承本机两轮权重。

本地仅验证GPU0，Bash语法可离线检查；服务器GPU1是否能并行由实际资源预检决定。`PROJECT_READY_FOR_SERVER`表示代码与依赖已完整打包、已有本地真实执行证据，不表示已在未知服务器或GPU1完成训练。
