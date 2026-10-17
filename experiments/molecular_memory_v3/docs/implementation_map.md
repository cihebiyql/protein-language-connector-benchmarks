# 分子记忆V3：现有实现与输入精确映射

结论：可复用冻结ESM/Qwen资产、完整L30残基cache、严格A1解析／C1算术、原split门禁与hook／冻结梯度模式；附件所需L15、可训练分层投影／类型位置身份／多bank／关系／数值head需要新scope实现。旧matched fixed-memory wrapper不是可直接接入新训练sender的接口。此文只做静态映射，未启动任何模型、forward、训练、安装、迁移或remoteprobe。

新授权scope：2026-10-17 00:29:40+08起，10:29:40+08止；优先<GPU_HOST>但实时资源及环境由资源lane核验。原2026-10-16模型／数据／合同／截止时间保持原样。

## 四个先决兼容边界

1. 旧MatchedConnectorLM拒memory.requires_grad、typeIDs，并锁[B,16,128]；旧collate.memory.detach和FixedMessageSender.no_grad会切断新W15/W30／关系／条件模块的梯度。新wrapper应在model.forward内生成可训练memory，仅冻结rawESM与base参数。
2. 数值head必须取答案前prompt末有效位置。完整teacherforcing序列的hidden[:,-1]会读gold/EOS；旧C2capture是no_grad/detach且取最后decoder块pre-finalnorm，不能直接当训练head。建议明确post-Qwen.model.norm的1536维promptstate。
3. 旧reader_state保存仅reader，遗漏新sender／condition／relation／numericheads。新checkpoint必须记录全部trainable modules与TRAIN scalers，不更新或打包底座。
4. 两A1scenario不能合并TRAIN或共享训练后checkpoint：known TRAIN含double TEST262／VAL245；double TRAIN含known TEST116／VAL125。按scenario×seed×variant独立训练，冻结backbone／rawcache／未训练pairedinit可共享。

## 输入与缓存事实

| 对象 | 直接证据 | 当前事实 |
|---|---|---|
| ESM2-150M | modelconfig＋revision a695f604… |30块、hidden640、20heads、max_position_embeddings1026；原weights float32，旧GPU抽取bf16 |
| Qwen2.5-1.5B | modelconfig＋revision989aa798… |28块、hidden1536、12heads／2KVheads、vocab151936、context32768 |
| 全L30cache | data/cache/bench_v02/panel_features_20261015/manifest.json |4920序列／31文本；max8525；153多chunk；本任务A1仅需2506序列 |
| A1 cache子集 | 两锁数据序列SHA join cachemanifest |408342残基、2625chunk、max3419、101多chunk、535,331,442字节，缺cache0；来源均fresh frozenESMforward |
| 残基payload | cache_panel_features.py:74 |residues30=[L,640] float16，mean30=[640] float32，sequence／SHA／length／chunk_spans／modelrevision／BOS-EOS removed／frozen标签 |
| chunk语义 | cache_panel_features.py:60,135 |不重叠连续≤1022残基；逐chunk去BOS/EOS后按原序拼全长，无跨chunk注意力；完整覆盖不等于全长联合上下文 |

A1 known：2062／147／152；double：2063／431／442。两场景都只有assay观测（96／125种），温度、pH、buffer、实际construct及verified construct观测均0；严格同条件pair为0。不能把缺失mask称作补全实验条件，也不能据此造匹配pair强弱标签。当前支持broad reference-target pKD，非已验证实际实验构建体条件下因果判断。

### L15新增方式

旧cache没有L15，不能由L30反推。建立新目录experiments/molecular_memory_v3_20261017/cache/esm2_150m_layers15_30_<extractor_sha>/，原cache不覆盖。优先同次冻结ESM pass抓encoder.layer[14]的第15块后／finalLN前状态，以及模型last_hidden_state的第30块后／finalLN后状态。保存每层相同完整chunk坐标、source／model／tokenizer／extractor／安装库SHA、dtype／backend及无标签声明。

本机安装库静态源码为Transformers4.57.6：ESM encoder:536–568，generic.check_model_inputs:1023/1093–1101先保留输入并把最后状态替换成normed last_hidden_state。因此hidden_states[15]与[30]的具体norm语义不同；<GPU_HOST>必须另核实际库源码／API，不能只凭版本名盲取索引。只抓两个指定输出可避免保留全部31份hiddenstates；hooks用finally清理。

L15单层FP16裸tensor约522,677,760B；同次15＋30约1,045,355,520B，尚不含文件／metadata。保存前后均需finite检查。L15-only sidecar可以引用旧L30 SHA，但需声明不同提取pass／批形／dtype政策，不能说完全同次；新配对L30对照更清楚。

## 可复用组件和接口

| 组件／函数 | 文件起点 | 复用范围 |
|---|---|---|
| sequence identity/chunks/feature provenance | scripts/bench_v02/cache_panel_features.py:18 | source-level utility and schema reference |
| existing raw feature loader | scripts/bench_v02/train_panel_baselines.py:84 | read-only L30 accessor, new layer-aware loader needed |
| chemistry/context baseline | scripts/bench_v02/train_panel_baselines.py:40 | TRAIN-fit148dim encoder and readable allowlist reference |
| specialized predictor | scripts/bench_v02/train_panel_baselines.py:129 | strong supervised reference, new relation-feature-matched variant needed |
| frozen Qwen hooks/gradient and off control | scripts/matched_budget_20261016/connectors.py:69 | pattern and tests, not fixed-memory class drop-in |
| raw-residue typed older connector | scripts/foundation_20261016/connectors.py:97 | variable-length masks/type pattern only |
| LM serialization/scoring | scripts/foundation_20261016/run_connectors.py:75 | unchangedA1 prompt-input allowlist/answer precision/parser/causal loss as separate functions |
| C1 and diagnostics | scripts/foundation_20261016/information_diagnostics.py:39 | NLL arithmetic and exact correct/off definition; not training-state capture |
| panel split gate and metric channels | scripts/bench_v02/panel_execution.py:32 | original physical data checks and separate generation/readout/predictor semantics |

建议的新接口（尚未实现）：raw peptide／target各层[B,L,640]＋bool残基mask＋global_residue_position／chunk_id＋对象／来源layerID＋allowedchemistry／context及missingmask；训练projection每层640→d（128／256），每bank[B,L,d]。关系U=Attention(peptide,target,target)，关系MLP输入[Zp,U,Zp*U]宽3d；缺target的RT直接关闭该bank，不能全masked softmax变NaN。

Qwen query[B,T,1536]→[B,T,d]，每bank读出后soft组合并投回[B,T,1536]，门控残差cast回base dtype。postnorm prompt末状态[B,1536]交给任务独立numerichead[B]；LM生成logits和head预测独立评分。类型／层／全局位置必须显式提供，8＋8次序本身不是身份；位置也不是结构坐标。

数值head若复用同一teacherforced forward，leftpad下gather index=Lmax−(prefix_len+answer_len)+prefix_len−1；纯prompt推理用最后active位置。rawcache可detach，可训练memory不可detach；冻结Qwen用requires_gradFalse＋eval，接入后计算仍保留autograd。base／reader dtype分别处理，不能wrapper整体转bf16而把原FP32 trainable接口悄悄cast。

原ContextEncoder为148维：2个TRAIN标准化数值＋6flags＋12chemonehot＋128allowedtext hash。允许chemical_state七个既有字段、assay／buffer／target_role；禁止sampleID／DOI／publication／rawmetadata／gold。new schema需严格nested whitelist及finite-aware missing。actual construct sequence当前未作为输入，只有missing／verified标志；有新construct数据也须新明确resolver，不能改旧row含义。

严格A1评分复用foundation.run_connectors.parse_answer，不换成更宽松的bench_v02.contracts.json_generation（后者可接markdown／extra fields等）。原A1teacher答案两位小数＋EOS，但test score用原精度pKD、不clip；C1是固定θ branch off−on的逐row token-mean NLL。当前同θ mean替换仍是额外干预，不由重新训练real／neutral系统差或C1自动得出。RT正式通道旧schema键retention_time且unit=min，不混pKD head标定。

### 不可沿用的旧执行入口

run_trial.main硬锁旧protocol SHA和2026-10-16T23:20截止；execute_remaining_v1固定<HOST>、/mnt/d、WSLvenv、sibling科研工作台标准根、singleGPU空闲判定。rt_baselines更有<WORKBENCH>绝对STANDARD。复制函数时ROOT=parents[2]在新nested位置会改变语义。新scope必须独立executor/config/contract，显式project/model/cache/data/schema roots，不篡改旧常量／源码／合同／deadline。

旧第31次pool已发生“实际提前结束但声明timeout超过actualstart剩余窗口”而拒收。新executor应在昂贵hash／验证后合法冻结预算，spawn前再次核剩余；若冻结budget已不容纳则拒绝启动并保存not-started收据，不能事后改合同。保留UTC与monotonic双计时，新的10:29:40绝对截止不回拨。多GPU按已分配GPU身份分别probe，不把其他GPU上的独立seed误判成侵占本GPU。

## 最小remote资产逻辑闭包

静态保守A1 bundle：2639个文件，共4,311,226,001B（约4.015GiB）。逐文件路径／字节／既有SHA全部在JSON。实际网络传输可在资源lane核<GPU_HOST>已有同SHA后去重，本lane未探测或迁移。

| 组 | 逻辑大小／控制 |
|---|---|
| ESM完整modelmanifest闭包6文件 |595,260,503B |
| Qwen完整modelmanifest闭包9文件 |3,098,971,928B |
| A1所需L30 2506payload |535,331,442B |
| 两A1锁pin并集48文件 |78,547,471B；包含旧a1_locked provenance／alignment，旧gate逐一rehash，不能只带samples和lock |
| 其他小文件 |原cache/model manifests、两scenario lock/manifest、64个旧Python静态包约1.1MB、标准模板／validator约53KB；新scope代码尚待owner生成 |

完整cachemanifest列4920条但A1只传2506，要建派生subset／overlay manifest引用原SHA并如实声明子集，不能把缺文件称完整cache。保持relative hierarchy可复用旧splitgate，标准schema复制到新scope --standard-root，不依赖D路径。无需传旧训练套件、publication、巨型ProtNote/PepInter/OPUS/ProtT3、papers或authorweights。底座只读复用，不导入已训练旧reader作新消融初态。

RT optional：a2_target_march.jsonl为10,459,078B、7070行／6929独特序列，现cache缺6929；另有原rt_target_diagnostic分组与化学检查可参考，原A2 source iRT gate仍HOLD。B1有588行／558局部区域序列已缓存，任务是蛋白局部功能区域，不是自由肽药效或已审QA；若预训练使用需要跨任务暴露审计。

## 静态性能与CPU测试边界

旧训练是manual list batch4，没有DataLoader管线；每batch重新JSON/chat tokenize、cache首次miss rehash和torch.load；WSL /mnt/d多小文件IO、每step float／finite／gate同步、全[B,T,151936] logits和CE都可能是瓶颈。旧B1逐例生成每token完整重算prompt，KV cache关闭；C1另加correct／off两次teacherforcing。以上是代码推断，不是新模型吞吐实测。

可研究预tokenize、rawfeature只读memmap／bounded cache、CPU collate＋pinmemory；trainable sender仍在forward。numeric-only可直取Qwen.model finalhidden省vocabhead；jointCE必要logit子集只有因果shift验证后才可用。batch／bucketing／accumulation／AMP／cache都需新共同政策和TRAIN资格，不能为了test分数选速度版本；旧B4/B2资格失败记录不能抹掉。旧hook按fullT mask写入，直接use_cacheTrue遇增量T1会形状不符，不能偷开。

可复用test_fixture思想（本lane未执行）：test_foundation_connectors冻结／off／hookcleanup／pad；test_matched_budget_core causal／firstgate／dtype；runner严格collate和validation-before-test；tokens类别分母；information prompt-only状态禁止labels。新增CPU边界必须覆盖 learnedW/condition/relation/head梯度通路、答案变更不影响pre-answerhead、全masked可选bank、typedposition交换、全部trainable checkpoint roundtrip、128/256真实参数/通信数、nested标签泄漏和source pin篡改。CPU通过不代替<GPU_HOST>真实TRAIN数值／显存／吞吐qualification。

## 未决与证据范围

新模型文件ownership、完整V0/V1/V2矩阵／loss权重／选择器／课程及任务标签覆盖由父任务冻结；本lane不写模型／trainer／executor。<GPU_HOST>实时GPU、CPU、软件版本、路径与已有同SHA资产由资源lane负责，未SSH／probe。附件论文动机本lane未重新外检；本文件依据附件＋本地code/config/manifest，明确区分事实、接口提案和未知。

本lane只新增implementation_map.md/json；所有原科学源／缓存／数据／模型／合同未修改，没有forward／fit／GPU／安装／实际迁移或递归委派。

## 新接口预算不能沿用旧固定16槽数字

例如肽16＋靶点240、两层、128维float32的完整memory，仅原始分层bank就262144B／对象／forward，是旧16×128的32倍，尚未计关系／条件bank。一般为4×d×所有可用对象／层残基数，再加实际新增bank。若需要同通信控制，要预先固定总K并统一分配；否则如实列token／query／key／byte／参数／FLOPs，不称为旧8192B匹配。数学W为640×d，nn.Linear.weight为d×640。

新增条件、关系、type／layer／position、query-gate和numerichead后必须重新统计；旧464001和附件约60万都不是新模型总数。零门控首步其他分支梯度为0及attention key-bias零梯度不应误判死路；不能同时零门控和零输出分支，需有限且非退化toy fixture检查路径。

精确AST函数／类起止行与全部静态输入／迁移指纹在JSON，旧15个源码SHA在交付时再次核对一致。
