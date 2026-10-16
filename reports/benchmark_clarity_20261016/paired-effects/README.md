# 固定核心配对效应图

此目录仅负责不可变summary的验证与制图，不读取活跃case、不训练模型、不重算主结果或bootstrap。当前准备阶段只允许验证20项快照；等待父任务提供完成core24的新snapshot路径及SHA后，才实际渲染。

固定设计为2场景×5contrast×3seed＝30行：depth_real、depth_neutral、content_early、content_mid、DiD。所有原null保留；k不足3不计算或显示均值/SD，不删除seed。黑菱形和误差条仅引用原summary的mean±sample_SD，SD不是CI。门控/池化属于预登记扩展阶段，原summary全部保留，不因分数而选择。

配置在`render_paired_effects.py`顶部。使用既有Python及matplotlib/pandas/openpyxl，Agg；线程环境及MPLCONFIGDIR只在该子进程内设置到本次输出目录，不安装依赖。独立输出PDF/PNG/JPG、精确绘图XLSX、中文三段caption、数据健康检查及SHA manifest，不生成合并PDF报告。

验证20项快照（不会生成图或XLSX）：

```powershell
python -X utf8 reports/benchmark_clarity_20261016/paired-effects/render_paired_effects.py --summary reports/benchmark_clarity_20261016/interim_20cases_20261016_2020/matched_summary.json --sha256 4ca600c4b0d8ebf90948d98c60c2bbd2af4c3423acaf34903cafd7022f19ab3d --output-dir reports/benchmark_clarity_20261016/paired-effects/validate_20 --validate-only
```

正式渲染需父任务提供新的core24不可变路径和SHA，执行同一入口的`--render`并使用新的输出目录。脚本拒绝不足24核心完成项、错误SHA、schema/算术不一致、duplicate及已存在非空输出目录；不在原snapshot目录写文件，不自动修复数据。

XLSX保留30行原始配对值与源mean/SD、原null、case MAE元数据、JSON指针、source及script SHA；除Excel数字列外还保留exact JSON和binary64 hex字符串，避免Excel显示精度被误当原数值修改。每页冻结表头、设置列宽，无展示性着色或格式筛选。

DiD同时等于depth_real−depth_neutral和content_mid−content_early，是代数依赖量；正DiD不保证mid绝对好。content比较分别训练θ_R/θ_N的系统总效应，不是固定θ推理干预。N＝3配对seed，故不实施正态、VIF、p值、优胜阈值或其他推断检验；不做清洗、插补、异常值过滤。

准备阶段已有8项CPU检查通过，包括重复/类型/原均值SD错误拒绝、共享轴范围、Excel精确字符串与null、以及20项输入的render阻断。最终准备版验证记录在`validate_20_final/`：30行中25个原值、5个原null，双新2026五格没有均值或补值，实际图形生成数为0。初版`validate_20/`独立保留。等待父任务提供core24的正式不可变路径及SHA。
