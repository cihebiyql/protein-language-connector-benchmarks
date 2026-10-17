# 第三类机制评价：固定条件探针

本模块检验已选M3接收端表示在给定冻结分子摘要后，是否仍有线性可用的pKD预测信号。它是**预登记单seed机制诊断**，不是三seed优势检验，也不是互信息估计。当前只完成代码、CPU夹具与CLI门禁；没有运行真实表征抽取或真实probe拟合。

## 文献与迁移

[Hewitt等，Conditional probing: measuring usable information beyond a baseline](https://aclanthology.org/2021.emnlp-main.122/)物理p2 §2.3 Eq.2直接比较`[B;R]`与`[B;0]`；p3 §2.6将对照解释为给定baseline后可由指定预测族读取的附加信息。已保存原文为项目`papers/continuation_20261016/evaluation/conditional-probing.pdf`，SHA256 `0d44d3b5c26ab2dbbffbc506e6d2535520bf46af79d061807fbfd7550f0545d1`。来源、页码与范围也见[既有文献审查](literature_review.md)。

原论文的条件V-information联系使用logloss。本实现迁移为固定ridge回归误差差值，单位分别为pKD²和pKD，**不报告bits或总互信息**。随机压缩和受限线性族读不到信号也不等于完整表示无信息；正增益也不单独证明预训练语言知识复用。

## 冻结方案

协议文件：`configs/probe_protocol.json`，由`mm_probe.load_protocol`与代码内完整常量逐项比对。禁止修改lambda、projection seed、维度或选seed后继续沿用该诊断名称。若确需新方案，应作为独立协议重新登记。

- 两个原场景独立进行；每场景只用**M3、seed17、原validation NLL已选的selected.pt**。不从三seed或多个epoch中再挑probe较好的模型。
- 抽取batch size固定4；不增加监督目标，不更新θ，不生成答案。
- `R`为既有numeric_head的**输入**：冻结LM最终norm后的prompt_end 1536维状态。forward_pre_hook只捕获该张量，忽略numeric scalar输出，成功/异常均finally移除hook并恢复外层train状态。
- 所有前向只用`collate(..., supervised=False)`。抽取先通过`input_view(..., labelled=False)`移除TRAIN/validation标签；test仅用无gold视图。禁止传labels或任何teacher suffix。

## B、R的确切维度

|部分|固定操作|输出维度|
|---|---|---:|
|冻结分子基线|p15、p30、t15、t30，各对8槽取mean得到640维，按该顺序拼接2560；PCG64 seed0标准Gaussian矩阵除sqrt(2560)，无训练|64|
|条件摘要|仅observed槽：SHA256(slot_name + NUL + condition_id十进制字符串)，前8byte大端mod16定桶，第9byte偶数为正、奇数为负，累加±1/sqrt(10)|16|
|缺失/观测|固定十槽observed bool转0/1；不做填补|10|
|序列长度|log1p(肽长度)、log1p(完整靶标长度)|2|
|**B总计**|不使用样本ID、split、DOI、pkd或数值head输出|**92**|
|接收端R摘要|1536维prompt状态；PCG64 seed1 Gaussian矩阵除sqrt(1536)，无训练|64|
|**探针输入**|两臂分别`[B92;R64]`、`[B92;zero64]`|**156**|

Gaussian矩阵固定生成后保存为`projections.npz`，同时记录文件SHA和每矩阵raw bytes SHA；fit时重新生成并逐元素检查。特征文件按train/validation/test各自保存CPU float32 `B[N,92]`与原始`R[N,1536]`，不保留其它token hidden states。原始分子输入依赖已验证的paired ESM cache，H15/H30定义、长序列全覆盖后有损pooling继续沿用数据合同。

## 拟合与评价

两臂均用numpy float64求解ridge，固定λ=1000，截距不惩罚，无自适应jitter。仅TRAIN的逐列均值与人口std（floor 1e-8）用于标准化；拟合标签仅为原精度TRAIN pKD。validation只报告，不选λ；test只评分，不调参、不重选θ。

两臂各156列、157个名义系数。`[B;0]`后64列为零，系数应精确为0；这不等于实际识别自由度相同。保存每臂非恒定TRAIN列数、中心化设计矩阵rank，以及`1+sum(s²/(s²+λ))`的ridge有效自由度（含截距）。同宽度不能抹去动态/零列带来的统计容量差异。

先拟合并保存`probe_states.json`，再固定train/validation/test的所有预测文件，**完成这些后才打开已锁SHA的test_gold.jsonl**。保留原sample_id顺序与所有行，不裁剪预测、不丢弃失败。两臂分别报告MSE、MAE、target宏MAE、RMSE、Spearman；常数或单样本Spearman输出null并标记undefined。

报告`Δ = baseline error − conditioned error`，正值表示附加R降低该误差。ΔMSE、ΔMAE与Δtarget宏MAE分开，validation/test分开；此单seed诊断不产生三seed显著性、CI或优越性宣称。接收端表示来自已用TRAIN标签联合训练的θ，因此诊断对象包含训练所获得的表征，不能直接归因为原始冻结LLM的生物知识。

## 原资产与执行门禁

readiness只读schema与SHA，不反序列化selected.pt、不加载torch/model、不解析混合原数据或test gold。必须匹配completed fit身份、原选点记录、checkpoint SHA、config SHA、prepared-view manifest、cache manifest及fit当时的四个core源码SHA（mm_model/mm_data/mm_extract/mm_train）。

extract和fit各需独立正规合同，使用已支持的透传字段：

```text
training_run_binding = {
  "kind": "conditional_probe",
  "probe": <mm_probe.execution_binding(args)["probe"]>
}
```

该binding固定stage、两种protocol/config SHA、selected checkpoint和fit manifest/selection SHA、数据/视图/cache SHA、六个实现源码SHA（含复用的mm_baselines指标实现）、scenario/M3/17、device、全部输入根和输出路径；fit另外固定feature manifest SHA与路径。标准validator本身也须以SHA固定并返回valid_for_execution=true。任何不匹配先拒绝，再有模型加载/拟合；沿用正规合同deadline/timeout，外层supervisor负责进程树、RSS及GPU准入。本模块不更改既有合同构造器。

完成的表征目录含projections、三split特征/收据及manifest。完成的拟合目录含两臂状态、三split逐条预测和result.json（全部SHA、覆盖、分数与限制）。未完成抽取不生成complete manifest，不能进入fit。

## CLI

```text
python -m src.mm_probe --help
python -m src.mm_probe readiness --probe-protocol configs/probe_protocol.json
```

尚未提供真实selected fit路径时，第二条返回`waiting_selected_fit_assets`和exit3，表示协议/代码可检查但资产尚不齐，不代表允许执行。完整readiness可再传`--planned-stage extract|fit`及下列同名资产参数；不会开始计算。

```text
python -m src.mm_probe extract --probe-protocol configs/probe_protocol.json \
  --config <冻结训练协议> --checkpoint <该场景M3_seed17/selected.pt> \
  --scenario <原场景> --view-root <新scope/data_views> --cache-root <完整paired_cache> \
  --project-root <项目根> --output <全新特征目录> --device <已准入设备> \
  --contract <正规合同> --validator <标准validator> --validator-sha256 <实际SHA>

python -m src.mm_probe fit --probe-protocol configs/probe_protocol.json \
  --config <同一冻结训练协议> --checkpoint <同一个selected.pt> \
  --scenario <同一原场景> --view-root <同一data_views> --cache-root <同一paired_cache> \
  --project-root <项目根> --features <完成的特征目录> --output <全新probe拟合目录> \
  --contract <独立CPU正规合同> --validator <标准validator> --validator-sha256 <实际SHA>
```

以上带尖括号命令是待绑定模板。真实extract/fit本次未执行，父执行器取得正规合同后才运行。

## CPU验证

现有WSL环境，CUDA_VISIBLE_DEVICES为空，OMP/MKL/OpenBLAS各2线程；无安装/依赖变更。11个独立测试通过：协议不可调参、固定92/156维、gold/ID无特征路径、pre-head捕获与异常清理、teacher suffix拒绝、TRAIN标准化、零列系数/rank/有效自由度、undefined Spearman、SHA/seed/core-code绑定、标准validator拒绝与device不匹配、完整预测先于test gold。集成拟合只使用临时合成8/3/2行CPU夹具，不能作为真实实验成绩。

当前全scope CPU回归共**151项通过，10.845秒**；CLI `--help`正常，最小readiness实测返回waiting_selected_fit_assets、model_loaded=false、fit_performed=false。协议文件SHA256为`077bc025aa26daed683c5b5439fe43debc04bbfe3697cc722c842bc678cd474d`。未真实抽取或拟合，正式使用仍依赖实际selected fit资产和合同。
