# MotifAE证据接口：独立探索验证

当前接法：冻结的ESM2-650M逐残基末层 → 冻结的官方MotifAE k3 → 只用TRAIN建立、独立资格验证的特征关联 → 带位置证据卡 → 同一冻结Qwen3-4B回答不同问题。

完整约定见 [PROTOCOL.md](PROTOCOL.md)，接入选择见 [reports/INTEGRATION_PLAN.md](reports/INTEGRATION_PLAN.md)。原v6实验继续运行，模型、划分、评分和暂停条件均保留。本轮不训练Qwen，不把特征激活自动当成已验证生物学功能。

私有面板从旧ELM TRAIN中派生：307个位点，训练201、资格39、探索测试67。源样本按父蛋白、同源组、局部序列家族、共同文献做连通组隔离。旧validation和TEST不参与。完整蛋白和变侧翼截出短序列的表示分别提取。

对照包括相同650M骨干的原始H33探针、原始基序正则、仅短序列、错配证据卡及确定性模板。以定位、已知阳性恢复、引用与证据忠实度分别评分。预设的探索增益信号只用于判断本轮是否有额外证据，不改变原v6的方向停止判断。

`resources/pipeline.py --scope <scope> --stage launch` 在node1独立完成数值资格测试、特征提取、词典冻结、独立测试、Qwen问答、聚合报告和GitHub发布。仅使用第四张项目GPU4；连同v6的3、5、6合计不超过4张。有限流程，不创建定时监控，不要求本机保持在线。

代码、协议、聚合报告和指纹可公开；原始ELM材料、逐条序列和位置、特征、词典、证据卡与回答均保留私有。权重及参考资料从固定作者发布下载，不直接上传大模型。

来源：[论文](https://shenlab.github.io/pdfs/papers/Hou-et-al-2026-MotifAE.pdf)、[作者代码](https://github.com/CHAOHOU-97/MotifAE/tree/8dd6a7588d30726fef98c789dac3042370e829c5)、[权重](https://zenodo.org/records/20547228)。
