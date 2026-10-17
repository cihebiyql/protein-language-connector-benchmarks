# 真实TRAIN canary独立只读审查

结论：**PASS_TRAIN_ENGINEERING_QUALIFICATION_NO_TECHNICAL_BLOCKER，38/38检查通过。** 可据TRAIN时间与内存证据冻结共同10epoch、训练batch4、生成batch8的正式预算。新36个case的训练、完整validation选点、TEST评分及科学效果仍待正式执行。

## 实际对象与身份

审查对象为<GPU_HOST> GPU3真实运行runs/train_canary_b8_v1。仅通过SSH读取已完成25个文件的SHA和22份JSON/JSONL元件，结束重算原25个SHA，字节全部保持一致。并发新增canonical_archive_v1下4个archive/preaccept/result/terminal行政归档文件，已记录其SHA与相对名；它们不改变原canary证据。source mm_train/mm_model/mm_data/mm_data_evaluation/mm_evaluate/mm_worker_limit/mm_supervise的本地、远端当前与run记录SHA逐项一致，输入/模型/代码合同启动校验通过。全部artifact ref、receipt→run_manifest、supervisor→contract/stdout/stderr的SHA链通过。

supervisor状态succeeded、exit0，结束无残余进程组；墙钟25.54365秒。worker状态complete、phase=development、execution_phase=train-canary；worker明确controller_pending，standard valid_for_execution=true而valid_for_acceptance=false。这是工程资格，未写成正式科研验收。

## TRAIN人口与防泄露

实际M3、new_peptide_known_target、seed17，4个optimizer update、16个优化样本曝光，完整epoch数0；history的epoch1是partial segment，epoch_fully_completed=false。没有validation选择，selection=null。固定TRAIN前8行做生成、numeric及on/off C1，8个ID、split、gold与不可变A1 TRAIN逐一相等，未出现heldout ID或validation/test预测文件。

numeric scaler按**全TRAIN2062行**拟合，count/source_split、mean与ddof0 std(下限1e-8)已用原冻结samples独立重算，与numeric_stats.json及checkpoint逐字数值相等。16是优化曝光数，不能描述成只读过16个TRAIN标签。原known samples SHA为d1d9ce44db4aeefcc829f5ce7b50b3e288ad03a28af9d15bb9f1fd550a6b657e。

未来答案扰动资格实际改变80个未来token，prompt_end_index和attention_mask固定；8条numeric_z均逐位不变，max_abs_diff=0。8条原输出并非常数，且测试时gate已打开。该资格验证此真实TRAIN探针的答案前读出边界，未把测试标签用于调参。无TEST标签参与的判断来自被pin的phase/read guards、数据refs及实际ID/输出证据；本审查没有OS syscall文件访问追踪。预计算的label-free ESM缓存包含固定输入集合，不等于使用heldout标签监督。

## 活动梯度与完整保存

step0的gate任务梯度非零，下游sender/readers/router等梯度为0，符合单外零gate初始化。step1与step3在answer CE和numeric MSE两条任务路径中，projections、condition/missing embedding、query、relations、readers、view/bank router与output projection均有非零梯度；numeric_head只对numeric MSE有梯度，符合输出与loss连接定义。这些是weight decay/clipping前的任务梯度，未用参数发生变化冒充学习证据。

实际检查点文件是**artifacts/canary.pt**，不是验证集选中的selected.pt；SHA256为e327b9ddf6e56d57c579d865184dc97ca442041e4b3beb4785041385c974f26b，19,503,316字节。白名单metadata-only pickle读取没有导入torch、实例化模型或分配tensor storage。检查点含53个活动参数张量，共1,618,568参数，全部名字/shape齐全；另含slot_positions、static_anchor两个buffer，没有frozen LM。53组Adam state的exp_avg/exp_avg_sq与参数shape一致，53个step均为4；history、identity、scaler与外部JSON一致。

ZIP CRC无误；214个FloatStorage的4,858,573个原始float32逐一有限，storage字节长度全部匹配。保存内容覆盖当前trainable state、buffer、optimizer与scaler；这不是对精确中途续训/RNG恢复的承诺。

## 生成、计分与时间

生成cap24、batch8；8条均在第18token遇EOS，真实model call为18。无需凑24个dummy call。最初审计曾把cap误写成应恰好24次，已按真实EOS及源码停止条件更正，运行产物未改。

8/8答案未通过原严格JSON，parse_valid_rate=0、full_sample_mae=null；numeric8/8有限。LM、numeric及C1的所有存储指标均从原TRAIN逐行预测独立复算，完全一致，未修补code fence或替换LM答案。0/8解析是早期开发信号，不作为停止其它seed/对照或改变测试规则的理由。C1的通用字段n_test=8在本次函数调用中指8条TRAIN探针，实际split/ID已核验；它不表示打开了TEST。mean/uniform not_run，生物donor/P/H和严格pair保持undefined/0。

首update累计2.83412秒；后续三次更新差值0.40624、0.20250、0.34107秒。8条生成1.72976秒；单次4update pilot提供预算估计，不保证全矩阵相同吞吐。GPU峰值alloc6,505,563,648字节、reserved12,767,461,376字节，均符合当前20GiB allocator预算及物理容量；host process-group RSS峰值5,770,334,208字节，低于24GiB合同上限。

## 正式释放边界

父级可在未打开新TEST的前提下冻结共同预算与配置，逐case继续fresh resource admission、源码/资产合同校验、全validation最小NLL最早tie选点及相同评分。该canary仅覆盖M3/known/seed17及4update，后续6臂×2scenario×3seed仍须实际执行；不同方法的活动参数和通信计算差异仍需如实报告。本审查只新增real_canary_review.md/json，未改runtime源码、旧结果、原数据，也未运行模型forward、fit或GPU。
