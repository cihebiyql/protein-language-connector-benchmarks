# V3自动评价协议与接口

实现为`src/mm_evaluate.py`，只导入Python标准库。生成批量允许正整数，首次正式test前统一冻结并与训练config.training.generation_batch_size双绑定，不能逐case改变；greedy、24新token和严格评分保持原规则。当前JSON协议训练预算为**UNFROZEN**；开发评价可运行，formal评价和formal协议验证强制失败。父级完成TRAIN canary后，须在首次正式测试前同时冻结训练config与本协议。本scope截止2026-10-17 10:29:40+08:00，旧窗口保持关闭。

## 冻结对象与两个SHA

固定6臂M0/M1/M2/M3/A_L/A_Q、seed17/42/2026、两个独立scenario，共36个注册tuple。两个A1原划分分别2062/147/152与2063/431/442；samples/test_gold/manifest/lock共8个SHA直接写入代码与协议，任一差异拒绝。原gold精度、ID、顺序、范围与test人口保持。旧30完成模型仅历史参考。

`run.protocol_sha256`是**训练config文件SHA**。`evaluation_protocol.json.training_budget.config_sha256`绑定该值，config_path相对scope根；评价器重新读实际config的training字典，核canonical JSON签名。`training_budget_sha256`的编码为`json.dumps(training,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)`的UTF-8 SHA256。评价协议自身SHA独立记为result.evaluation_protocol_sha256，绝不与训练config混同。

正式评价还核evaluator_source_sha256与实际评分源码一致。正式冻结需填：training_budget.status=FROZEN、training、training_budget_sha256、config_path/config_sha256，以及两个scenario各自完整的expected_training={epochs_completed,optimizer_steps,examples_seen}。新预算由canary后协议确定，不默认沿用旧1548更新。正式budget.training的generation_batch_size与generation.batch_size必须精确相同，实际config文件还会独立重核；零、负数、bool和非整数批量均拒绝。producer_source_sha256记录当前mm_train.py字节，正式冻结时再与最终producer源码复核。

## API与命令

纯函数API：`parse_answer(text)`、`generation_metrics(gold_rows,predictions)`、`numeric_metrics(gold_rows,predictions)`、`select_checkpoint(history)`、`seed_statistics(per_seed)`、`c1_metrics(gold_rows,records)`、`summarize_runs(results,training_config_sha)`。文件入口为`evaluate_run(run_manifest_path,evaluation_protocol_path,data_root)`。gold_rows采用旧sample_id/split/pkd/groups.target_sequence_sha256结构；TRAIN canary可用这两个metric函数，不能作为formal结果。

```text
python src/mm_evaluate.py validate-data --data-root <old_a1_locked_v2>
python src/mm_evaluate.py validate-protocol --protocol docs/evaluation_protocol.json --development
python src/mm_evaluate.py validate-protocol --protocol docs/evaluation_protocol.json
python src/mm_evaluate.py evaluate --run-manifest <job/run_manifest.json> --protocol docs/evaluation_protocol.json --data-root <old_a1_locked_v2> --output <NEW_result.json>
python src/mm_evaluate.py summarize --run-manifests <run1.json> <run2.json> ... --protocol docs/evaluation_protocol.json --data-root <old_a1_locked_v2> --output <NEW_summary.json>
```

输出文件使用exclusive创建，已存在则退出2且保留原字节。invalid input退出2，错误不回显凭证或整份输入。summarize从原run manifest重做每run校验与计分，不依赖未经核验的汇总分数。未提供的tuple列missing；开发、partial和失败run保留状态，不能填formal得分。

## 严格run与逐行格式

run_manifest必需keys：schema_version=1、scope_id、run_id、phase(development/formal)、status(complete/partial/training_failed/evaluation_failed)、scenario、arm、seed、protocol_sha256、training_budget_sha256、data_sha256、training、test_labels_used_for_fit=false、artifacts。允许的额外元数据名见JSON run_manifest_schema，未知key拒绝。训练仅包含epochs_completed/optimizer_steps/examples_seen三个实际计数，性能字段放runtime。

artifacts固定6键history、selection、checkpoint、lm_predictions、numeric_predictions、c1；各值为{path,sha256}或null，路径必须解析在该run目录内且无symlink越界。ledger、numeric_stats等另放provenance.audit_artifacts。complete必需前5项；formal还按已冻结C1计划要求第6项。fit产物不是可评分evaluate manifest。

history每epoch至少：epoch、validation_answer_loss_sum、validation_answer_token_count、validation_rows、validation_ordered_ids_sha256；可附诊断。ID SHA使用原validation顺序的sample_id数组按同一canonical JSON编码。每epoch必须全validation覆盖，epoch1..epochs_completed连续；全token loss_sum/token_count最小、最早精确tie选点。重构容差1e-12只核存储值，不改变tie或胜负。selection必需且仅含best_epoch、validation_answer_token_nll、checkpoint_sha256、test_labels_used_for_selection=false、written_before_test=true。

LM JSONL每行必需sample_id、split、gold_pkd、raw_text；可含parse、raw_generated_token_ids、first_eos_length、generation_error和run_id/protocol_sha256/checkpoint_sha256。saved parse存在时必须与独立重解析一致。raw token IDs是非负int，最多24个，first_eos_length含首EOS且不超过实际IDs；标准库评价器核类型/长度，原tokenizer精确decode由被pin的producer承担，不能宣称本工具独立解码验证。

numeric JSONL每行必需sample_id、split、gold_pkd、prediction_pkd、error；可附相同三项身份。成功时error=null且预测有限实数；失败时prediction_pkd=null并有明确error。数值必须已用TRAIN mean/std(ddof0)反变换回原pKD；评价器不round、不clip、不把numeric值填进LM文本。

完整输出必须逐行严格匹配原test ID顺序与gold；重复、错序、缺失、额外行、错split、篡改gold/parse/文件或不同checkpoint均拒绝。partial只保留实际写出的原顺序前缀，列出缺失ID和人口分母，正式分数为null，绝不制造未尝试行。真实生成报错可记录raw_text=null及generation_error，仍计失败。

## 指标和多seed

LM parser保持旧语义与3种reason：只能解析一个JSON object、唯一pKD键、有限实数；普通JSON空白允许，解释/code fence/重复键/多键/字符串数值/bool/NaN/Infinity拒绝。任何格式失败使full_sample_mae=null；parse_valid_rate和valid-only分母照实输出，不换成功子集为主分数。主人口固定152/442。

numeric独立报告MAE、RMSE、Spearman、target_macro_mae。Spearman用平均秩处理tie，常数或小于2行时null。target_macro按精确target_sequence_sha256先算每target MAE，再等权平均，完整分母48/157；它不是操作group_id macro。numeric任一失败使full_population=null，有限子集仅诊断。

summary每scenario/arm/metric列全部seed、k/3、缺失seed；仅3/3指标有值才给mean及样本SD(ddof1)。complete运行数与full-MAE有定义数分开。M0配对误差差值=MAE_M0−MAE_arm，任何一侧缺失使该seed差值null，不挑seed。C1同样逐seed汇总，尚未运行的mean为0/3而不是0效应。

## C1与条件语义

本轮正式C1计划是同θ的on/off NLL及numeric on/off；TRAIN-mean和uniform标not_run。每行C1必需sample_id、answer_tokens、correct/off:{loss_sum,token_count}、checkpoint_sha256，numeric_correct_pkd/numeric_off_pkd成对有限。G_off=mean_i(off_loss_i/n_i−correct_loss_i/n_i)，先求每行token均值再平均行；不能用全token加权替换。numeric效应=mean_i(|y−off|−|y−on|)。mean若将来实际注册并提供，才核TRAIN统计SHA、同checkpoint和fit_split=train。关闭LM branch对numeric的含义由模型边界决定，报告具体计算干预；P/H与严格同条件pair仍undefined/0，不能填成生物效应。

与data/model owner核准的10槽固定顺序如下：

| 槽 | 精确字段 |
|---|---|
| 0–6 | c_terminal、chemical_modification、cyclization_method、explicitly_representable、n_terminal、residue_modification、topology |
| 7 | assay |
| 8 measurement_context | temperature_kelvin、ph、buffer的精确canonical组合 |
| 9 target_context | target_sequence_role、actual_construct_verified、actual_construct_sequence_missing的精确canonical组合 |

类别hash不fit测试类别，missing显式表示而非伪造测量0。槽8任一真实观测才observed；槽9的已知角色/状态不是实际构建物核实证据。当前温度/pH/buffer/实际构建物全缺失，分层报告n_missing/n_observed=0及undefined_empty_stratum。gold/ID/DOI/raw metadata不进条件；numeric只用memory或答案前prompt state，标签只进训练损失。

## 验证与边界

`python -S -m unittest discover -s tests -p test_evaluate.py -v`已通过53项纯CPU fixture；模型和训练权重从未载入。另对31份旧归档预测、9062行（包含非正式case31）逐一复核新parser与原stored parse，零差异，不将这些行称新baseline。正式新结果只能由完整冻结manifest及真实执行产物产生。既有test已用于历史架构迭代，当前评价保留这一暴露事实。
