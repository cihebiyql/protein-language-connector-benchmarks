# Molecular Memory v3：可实现性与模块合同

**结论：V0和V1可按现有ESM／Qwen底座逐步实现；V2配对训练及B2／C3事实解释训练目前缺金标准，须保持关闭。** 新目标窗口为2026-10-17 00:29:40—10:29:40，独立于已结束的10月16日合同。本文是实现前设计审查，没有实现、模型加载、forward、新fit或GPU作业。

已全文读取用户附件。旧核心24没有稳定mid优势，门控扩展也翻向；这支持检验新接口，不提供新架构必胜的先验。新模块和新增监督会改变系统，旧结果只能作为保留的历史参照；新scope不能沿用旧464001／“约60万”作为全部模型参数。

## 1．数据边界：先让未知保持未知

本次只读核算：known全部2361行、double全部2936行，pH、temperature、buffer和actual_construct_sequence均缺失，actual_construct_verified=true均为0；target角色都是数据库参考序列／实验构建体未核实。两个scenario的samples SHA仍为`d1d9ce44…`、`7ff5a672…`，TRAIN的最大P＋T均为3429。

- 附件“首版只限已知实际构建体”不能直接应用到旧A1，否则会改变或清空冻结test。V0继续使用全长参考靶标、原chemical／assay及显式unknown／reference-role标记；另有构建体核实子面板时再独立登记，不能静默删除旧test。
- 缺失条件不能填成真实pH=0、温度=0或同一已知条件。数值张量为存储使用0占位时，必须另带observed=false，并用missing状态编码；原数据仍为None。missingness token与padding mask是两个概念。
- 当前same-condition pair、严格独立donor均为0；B2 reviewed QA／C3裁决事实gold为0。V2、pair loss、事实／解释监督应显示`unavailable_gold`并拒绝启动，不从任意两行pKD造强弱、从未测对造负例或把证据登记表当gold。
- 可先只做已有A1定量任务，不要求虚构“事实预热”阶段。RT目标域诊断若另启用，仍保持target-only及原跨条件HOLD，不造一个靶点；B1不能冒充游离肽效力。
- 各scenario独立训练、fit条件词表／数值统计和选点；冻结无标签PLM缓存可共享，训练后的投影、Reader、head和condition统计不跨scenario接续。旧test已见，新结果仍是该固定基准的后续验证。

## 2．输入与memory接口

建议第一轮主路径选**固定bank**，同时让代码接口支持per-residue模式；两者有不同通信预算，不在同一budget组混名。

| 对象 | 张量合同 | 约束 |
|---|---|---|
| 冻结肽层表示 | `H_p[l]: [B,Pmax,640]`、`mask_p: [B,Pmax]` | l先30；V1增加15。mask=true仅实际残基，padding不参与pool／attention |
| 冻结参考target层表示 | `H_t[l]: [B,Tmax,640]`、`mask_t: [B,Tmax]` | 所有原残基覆盖；target_present区别于construct_verified |
| 条件 | 固定field顺序的原允许值＋`observed[B,F]`、`missing[B,F]`、padding／field IDs | TRAIN-fit类别／标准化；不含sample ID、DOI、标签、含答案描述。F及sentinel规则在协议固定 |
| 可训练分子投影 | `P_l: Linear(640,128)`，每来源层独立；建议肽／target共用同层投影 | 对固定LN后的H操作；type／source-layer／位置标记显式加入。位置用原残基坐标／固定sin-cos等，不解释为结构坐标 |
| 分子bank | per-residue为`[B,Pmax或Tmax,128]`；fixed-bank为`[B,Kp或Kt,128]` | 可从全部有效残基独立adaptive pool，Kp=Kt起步8可作一个预定预算。pool压缩不是无损，也不是截断 |
| 条件bank | `M_c: [B,Kc,128]`、mask及field语义 | Kc／F必须预登记；保留已知化学、assay和未知状态的区别，不以unknown宣称条件真实已知 |
| 可选关系bank | `M_r[l]: [B,Kr,128]` | 由当前肽／target生成，Kr通常等于Kp或P；无target时关闭，不提供伪靶点 |
| 文本状态 | `H_L: [B,Ttext,1536]`、attention mask、显式`prompt_end_index[B]` | training可teacher-force答案，但question路由anchor／numeric head只读答案前状态 |

`MemoryBundle`应是按bank和来源层命名的tensor／mask字典，另有`bank_present[B,R]`及原始位置／chunk映射。缺失target任务跳过相应bank或先按presence分组；不能把一整行全masked键传给softmax造成NaN。条件unknown可有显式token，但它不是虚构观测。

长target沿用原≤1022非重叠chunk、去BOS/EOS后按原位置拼接，覆盖全部序列，同时声明原PLM跨chunk上下文断开。新增分块读取也必须保留完整coverage／mask／原位置；不截前N残基冒充full target。位置编码不能补回已丢失的跨chunk PLM上下文。

V1的H15需要新scope的独立cache及layer语义／版本。固定一基编号“第15块后”，核HF hidden_states索引与hook对应；旧H30是原last_hidden_state定义，不能无记录换成别的pre-norm状态。新增H15不覆盖旧H30文件／manifest，两个layer的同一残基／chunk顺序必须严格对齐。

## 3．分开读取、问题路由与LM注入

可用共享`LN1536→Linear1536×128`产生Q，p／t／c各有自己的MHA128（4 heads），同一对象reader跨来源层复用；最后合成128维后只用一个`Linear128→1536`写回。V1关系bank增加独立reader。哪些模块共享要在config明写，不能靠类名推断参数。

每bank／layer的读出`R_b,l:[B,Ttext,128]`；路由可分解为来源层权重π和bank权重ρ，mask仅对应实际启用输入。默认用**原prompt末端、注入前**的隐藏状态作路由anchor，再广播到各文本位置；它与后续gold／生成答案隔离。也可预登记逐token路由，但那是另一计算版本，不得不记账切换。

softmax初始logits=0产生有效bank上的均匀非零权重；外层`h'=h+tanh(gamma)·W_o(sum ρπR)`，gamma初值0。投影／MHA／W_o保持正常非零初始化。不能同时zero gate与整个关键输出分支，形成双零死路。当前A1问题模板基本固定，query会随允许上下文变化，但仅此不证明“学会按不同问题意图路由”；多任务／问题变化行为需单独证据。

**迁移中的硬错误要避免：** 旧`FixedMessageSender.encode`有`@torch.no_grad`且预计算投影后的消息。新可训练640→128、condition encoder、relation、router和reader不能沿用这个no_grad，也不能跨optimizer step复用落盘的已投影消息。只缓存冻结ESM／无参数前处理；可训练路径每次forward重算并保留图。冻结LM用requires_grad=false和eval；接入后的LM计算不能整体no_grad，否则训练信号断掉。

gate0首步非gate任务梯度为0是正常现象；gate应能先得到信号，打开后再检查投影／各活动分支的任务梯度。AdamW造成的参数变化不是任务梯度证据。工程检查用预定义多步canary／非零gate输入敏感性，不要求每tensor每一步都非零。off须完全绕过注入，恢复同position／mask的base logits，hook异常也清理；门值、路由概率和attention不是置信度、接触图或生物证据。

## 4．两条输出通道与监督

生成通道继续原严格pKD JSON、greedy24、原未舍入test gold／完整失败分母；不为解释训练把原主输出改成长文本。numeric head从最终LM的**答案前prompt状态**`[B,1536]`读出`[B,1]`，A1一个Linear head为1537参数；未来RT另head，只有实际训练的head进入该任务的活动参数账本。

numeric训练标签只进损失，TRAIN-only标准化若启用则预先冻结。生成选点默认继续原完整validation答案NLL；numeric val MSE／MAE另报，head输出不回填语言生成或修复parse失败。若joint CE＋numeric loss训练，所有新对照必须有相同真实head、loss权重／曝光／更新；这不是与旧CE-only同训练目标的纯架构消融。

最直观的实现是独立prompt-only numeric forward；若为节省预算复用teacher-forced forward，只能取显式prompt_end_index，并通过“任意改后续gold tokens不改变该numeric状态／输出”的因果mask测试，不能误取最后答案token。额外forward／head／监督都计账。pair／事实／解释loss没有合格gold则不计算，更不能用伪0标签或语言模型自评代替。

## 5．参数与通信：怎样配对活动control

以d=128、H=1536、Linear含bias为例：单层分子投影82048；共享query LN＋projection199808；每个MHA66048；共享输出198144；外gate1。一个来源层、p/t/c三个reader的核心为**678145**，尚不含condition encoder、type/layer/位置、router或numeric head。两来源层再加82048；一个关系MHA＋`384→128→128`两层MLP例为131840，还需其独立reader66048。精确总数以实现逐tensor审计为准，不用示例数字冒充已实现容量。

逐模块分别记录registered／requires-grad／实际使用／任务梯度与optimizer状态；缺gold模块不注册进optimizer凑数。通信至少区分原残基→投影、bank边界`4×d×Σ有效slot`、每bank/layer的KV／query次数、padding及生成实际EOS调用。L15／condition／relation会增加slot和读取，不能继续声称全部只有8192字节或同FLOPs。

| 模块对照 | 可活动、可计账的capacity control | 解释边界 |
|---|---|---|
| 加L15 | 两独立投影视图都读H30，对照H15／H30；两投影与后续reader均使用 | 同名义参数／slot／调用，baseline是L30双视图，不冒充单视图V0 |
| query router | 同router作用于预定固定**非零**q0，对照真实prompt anchor；输出仍影响所有bank | 不用zero输入令矩阵永不学习；常量输入可辨识自由度较低，不能称有效统计容量完全相等。detach不等于去问题依赖 |
| cross relation | 同MHA／MLP对肽self-derived输入生成关系槽，对照肽query读取target；target仍通过其独立bank供两臂使用 | Kp=Kt固定bank时Q/KV形状可匹配；per-residue长度不同则计算不匹配。不是合法错配donor或不结合负例 |
| split／merge reader | 相同数目活动experts读合并memory，与各读独立bank比较；或固定K下预定混合分区 | 前者KV读取更多，不能同时称等通信；严格混合分区对照只解释该分区规则。不能挂不用的reader凑参数 |

类型embedding移除后，独立bank本身仍提供对象身份；“-type embedding”不能叫“完全无分子类型”。非仿射位置／type处理和head是否训练也应明示，任何同参数结论都只是所申明的结构／名义预算，不是可辨识自由度证明。

## 6．有限的累计＋单模块消融矩阵

建议先用真正单视图V0做CPU／TRAIN工程验证，不计为科学成功。正式预算匹配suite可用同一完整图的六配置：

| 配置 | 两逻辑来源视图 | Router | 关系槽 |
|---|---|---|---|
| M0，capacity baseline | H30／H30，独立可训练投影 | 固定非零q0经同router | peptide-self control |
| M1，累计加中间层 | H15／H30 | 同static router | 同self control |
| M2，累计加问题条件化 | H15／H30 | prompt-query router | 同self control |
| M3，完整V1 | H15／H30 | query router | p→target cross relation |
| A_L，full减中间层信息 | H30／H30 | query router | cross relation |
| A_Q，full减问题条件化 | H15／H30 | static router | cross relation |

full减cross relation就是M2，复用且不重复计训练。每配置相同bank形状／K、dtype、head／loss与训练预算；上述control确实参与前向和训练。M0不是旧固定sender基线，也不是原论文复现。若root选择更简单的逐模块增加矩阵，也可实现，但参数／消息增长必须分组报告，不能统称严格预算匹配。

六配置×两scenario×三seed17／42／2026为最多36新训练位置；优先固定M0—M3核心24，A_L／A_Q另12为预先登记扩展。资源不足按原顺序保留未启动／partial，不看test决定做哪一臂。全合并reader、per-residue对fixed-bank、d256、额外LM插层、LoRA均可在框架支持，但首轮不并行扩成无限组合；未执行条件明确留空。

## 7．dev学习信号与正式test分开

CPU／TRAIN canary先验证shape、mask、full-residue coverage、无标签输入、off恢复、梯度连接、数值有限、序列化与解析，再依据真实耗时／显存冻结新scope统一epochs／batch／累计／K／生成batch／超时／λ与完整任务表。不能假定旧时长适用于新多bank模型；新增H15缓存和numeric forward也消耗10小时窗口。原10月16日合同不重开，不通过改旧日期重新运行旧controller。

dev／validation可记录CE、numeric loss、gate／router、分支梯度和原／实际残差；这些是优化诊断和预先声明的选点依据。只要工程资格通过、损失有限，早期小增益、无显著性或seed方向不利都继续预登记训练／其他seed，不逐seed“必胜早停”，也不要求每一步改善。停止理由限明确资源／数值／合同／数据错误或截止。

正式test只在该配置选点冻结后按原评分完整执行，报告所有seed／失败／原MAE及条件分母；仍保持三seed未齐mean／SD为null，数值头和生成分表。不用test挑模块、λ、source层、seed、条件过滤或输出修复。content对照仍是训练与评测系统总效应；固定θ的mean／off／donor是各自不同估计量。

新调度须在昂贵哈希／准备后及实际启动边界核对剩余窗口，避免再次出现“声明timeout大于actual start剩余时间”的验收失配；不能因实际提前结束倒改合同。最终10:29:40未完成就保留全表partial。本文没有放行GPU执行；新实现、输入／cache／代码pins、资源探测及标准合同须由父任务独立验收。

附件新增近期论文链接在本审查中未逐一重新核验；不把其身份或效果当作本合同已证事实。已有Flamingo／BLIP-2／ProtT3等本地证据可支持研究动机，具体创新与新引用由文献核查链另行绑定。所有接口和预算示例均为新scope的设计建议，不是已跑结果。
