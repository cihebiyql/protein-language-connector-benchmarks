# 门控初值扩展6项真实执行独立复核

结论：通过，无执行阻断。固定30快照JSON SHA2712ffcb…、MD SHA c6d12ca2…，只读取六个near_unit完整终态及对应六个mid／real／balanced／zero来源；核心24个result SHA全部与前证相同。未读取活跃global16或操作任何进程，未改旧报告／FINAL／协议／源码／结果。

## 实际检查范围

新增1782条严格生成、原MAE／并列平均秩全局Spearman及1782条C1按行G、类别共同full-token分母和约简残差独立重算一致。1782／1782有效，0失败；连同未改变的核心24既证7128，覆盖30项8910条，不声称本次重跑旧7128或任何forward。

六项均有真实3×516＝1548条连续更新，按原完整validation full-answer NLL最小值／最早并列选点；同对应zero源的训练超参、选择器、解码规则及冻结data gate一致。实际strictpolicy、fresh idle probe、环境freeze、原worker／代码／cache／projection／初态pins、标准合同准入与result验收、exit0和自有PGID退出归档均通过。

仅CPU2、CUDA_VISIBLE_DEVICES为空读取六个自产selected reader及三seed各两份未训练初态：每selected reader11个tensor实际464001元素，全部float32／CPU／有限，逐名称参数数、完整registration及所选epoch匹配。每seed十个非gate初态tensor逐位相等，只有alpha由0精确覆盖为3；两初态均标明未训练且不使用标签。未构造、加载或forward底座／旧selected模型。

alpha3表示tanh(3)≈0.99505475的近单位门控，不是ungated。二者名义参数完全相同，初始梯度条件不同正是处理的一部分。每例每forward16×128 float32＝8192逻辑字节；实际train／validation／B1生成的calls、queries、payload、attention edges及residual字节账本独立一致，不能据此声称总EOS读取／FLOPs相等。

## 资源与两个时钟

| 条件 | 选epoch | monotonic秒 | UTC跨度秒 | 差值秒 |
|---|---:|---:|---:|---:|
| known/17 | 1 | 553.126 | 613.187 | 60.061 |
| double-new/17 | 1 | 885.086 | 981.883 | 96.797 |
| known/42 | 3 | 567.218 | 630.620 | 63.403 |
| double-new/42 | 1 | 880.538 | 980.654 | 100.116 |
| known/2026 | 1 | 624.802 | 691.541 | 66.739 |
| double-new/2026 | 1 | 949.261 | 1052.689 | 103.428 |

六项monotonic与UTC跨度分别都≤1800秒，结束UTC均早于原23:20绝对截止，未重置时窗。最大采样whole-PGID RSS5.967GiB（上限12GiB）、Torch allocated5.929GiB（上限11.5GiB），reserved11.203GiB单列。UTC与monotonic差异保留，不据较短值宣称wall-time加速；采样峰值不证明瞬时全设备峰值。

## 原gate对照算术

冻结方向为MAE(alpha3)−MAE(alpha0)，正值才有利于zero初始化。两个三seed mean与样本SD（ddof1）独立重算一致：

| 场景 | seed17 | seed42 | seed2026 | mean | sample SD |
|---|---:|---:|---:|---:|---:|
| new_peptide_known_target | 0.477121 | -0.604664 | -0.273417 | -0.133653 | 0.554270 |
| new_peptide_new_target | -0.009364 | 0.245128 | -0.338927 | -0.034388 | 0.292830 |

两场景都随seed变号，平均符号不足以确立zero或alpha3的稳定优势。此对照包含各自训练轨迹和原validation选出的不同θ，不是同一checkpoint改变alpha的推理干预，也不能单独解释mid优势为何不稳定。未新增bootstrap／fit／评价阈值／选择条件。global16尚在本报告范围之外，不能把此固定30项称作36项完成。
