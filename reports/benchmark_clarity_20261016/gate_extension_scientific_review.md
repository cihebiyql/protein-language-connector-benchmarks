# 门控六项扩展及30项阶段草稿的科学复核

**结论：通过，无新增科学解释阻断或P0—P3问题。** 原门控初始化contrast的方向、逐seed值和三seed统计一致；两个场景都有方向翻转，不支持alpha0或alpha3的稳定单向优势。此结论限固定30项阶段，池化与全轮终态仍待验收。

只读22:57固定快照、门控／核心机制报告、冻结协议、既有源码审计及当前草稿；没有读取活跃global16目录，没有forward、GPU、fit、新bootstrap、下载或进程操作。仅新增本文件与同名JSON，未改原报告／FINAL或评分。

## 方向、计数与统计通过

门控公式保持`MAE(near_unit alpha3)−MAE(zero alpha0)`，正值有利零初始化，单位pKD，全部seed和完整总体保留：

| 场景 | seed17／42／2026 | mean ± sampleSD |
|---|---|---|
| 已见靶点 | ＋0.477121／−0.604664／−0.273417 | −0.133653 ± 0.554270 |
| 双新 | −0.009364／＋0.245128／−0.338927 | −0.034388 ± 0.292830 |

SD分母为3−1，不是CI；原组bootstrap仅保留、未重新抽样。负均值不等于near-unit普遍更好，不从扩展挑最优模型替换核心24主比较，也不补推未执行的early-alpha3或neutral-alpha3效果。

新增训练为6个near-unit模型，输出1,782条；和既有6个zero模型配对分析时涉及12个系统、3,564条输出。zero已经计入核心24，不能再计为新训练。唯一系统累计为24＋6＝30，唯一输出总量8,910条；不是把24＋12算成36或把3,564全部当新增输出。

## 初态、选择与机制范围通过

六对均为mid／real／balanced，11个登记初始tensor中alpha不同、其余10个SHA相同。相同Reader结构与464,001名义参数、16×128 float32／8,192逻辑字节、三epoch／1,548更新和原完整validation答案NLL选择保持。登记初态指纹及已存文件核验由门控证据报告承接；本次不加载初态或训练后权重。

alpha3仍经tanh，初始近单位且可学习，不是无门控；它同时改变初始残差幅度和gate优化参数化。初始化contrast跨了训练／验证选点／评测的全过程，不是固定θ的gate clamp。C1仍是固定所选θ、整个分支off／on的teacher-forced作用，不能替代均值memory干预或分子特异机制；donor为0时P／H不被补齐。

所选validation的raw、gated和hidden范数范围与表格相符；near-unit较大的gate和较小raw分支只支持参数化与输出尺度共同变化的观察，不证明主动补偿、唯一尺度原因、BF16舍入原因或更多语义信息。统计仍含prompt／teacher-forced答案，gated δ是BF16相加前量，不是实际状态差。C1与自主MAE、Spearman的差异和事后输出分布账本没有被换成新的主要端点。

## 当前FINAL草稿的窄核对

所审MD SHA为 `c9d00a7202dac7d063b30f77bcb21d48db364838d6d8cc14f2783bf270ac33d0`，JSON自检一致。草稿正确标为`core24_plus_gate6_frozen30`，新增1782条与门控公式／统计一致；池化pending、S／W未来未实施、最终资产保全／清理／发布待终态的身份没有被提前关闭。

此前范数范围建议已落实为“各case所选epoch的validation”，原数值及含gold／相加前限制保留。新增冻结ESM2-150M第30层640维残基缓存、Qwen2.5-1.5B-Instruct的28层／1536维，以及≤1022残基非重叠chunk覆盖全部残基但无跨chunk上下文，均与contract／budget／原缓存及runner源码一致。没有把分块全覆盖说成整条蛋白始终联合上下文化。

科学冻结SHA仍为 `9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`；固定30快照SHA为 `2712ffcb60f2b8736c6aff623f5e0c39fb219c164e5b077ab9083277acea386a`。本次是科学解释和已存算术的复核，不替代未来全36终态执行／资产／进程验收。完整来源SHA和逐项核查见[JSON记录](gate_extension_scientific_review.json)。
