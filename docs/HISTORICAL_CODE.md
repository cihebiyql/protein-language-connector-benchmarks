# 历史代码与复现范围

`historical/scripts/`与`historical/tests/`保存原有算法、试验程序和历史检查。公开副本只规范化本机路径／身份等字符串，原source SHA与public SHA由`publication/source_to_public_manifest.json`对应；这不保持源代码文件的字节SHA，不能以公开副本替代原合同绑定的文件。

原训练使用WSL Python 3.10.12、PyTorch 2.12.1+cu130、Transformers 4.57.6、NumPy 2.2.6。精确观测包版本另列`historical/requirements-observed.txt`，它是观测记录，不是跨平台安装承诺。默认CPU复算不需要这些模型依赖。

原模型入口依赖冻结ESM残基缓存、指定底座版本、原task-contract与本地统一验收器、资源探测和已结束的执行时窗。原数值策略包括BF16底座、FP32 Reader／消息、无autocast、TF32关闭及严格确定性；这些策略没有为公共发布改动。

历史训练代码有明确SHA门槛和2026-10-16结束的时限。当前目录不提供绕过门槛／改日期继续原运行的命令。未来重新实施训练须建立清楚的新执行身份、许可及资源合同，并区分科学超参／划分／选点与工程路径变更。

本公开发布没有新模型前向、训练、拟合或重新选择checkpoint。`analysis/`只读取既有公开文件，验证原预测、覆盖、MAE、Spearman及配对seed统计。CPU纯tensor导出仅封装已有Reader权重，拒收pool案例始终标记为`rejected_not_formal`。

V2/B4与V3/B2的TRAIN资格失败保留在历史源码／报告中；它们没有被选作本轮科学测试通道。只因实现测试通过不能宣称GPU资格或论文效果通过。

