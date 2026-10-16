# 2026-10-16 文件与数据集落盘交付

原任务中的60份已核验PDF和58项仓库记录已在 `PROJECT_ROOT` 完成逐项本地覆盖；新增ProtT3真实数据、OPUS18项原始基准、两套OPUS完整训练语料，以及protein-uq的GB1/Meltome数据与通用评价源码。新语料独立放入quarantine，原冻结划分、test_gold和历史结果保持原样。

截至10:14，ProtNote完整包及PepInter两份权重均已通过官方MD5与本地SHA验收；ProtNote又在全新隔离目录安全解压71个文件，29,364,788,715字节，独立逐文件重读CRC/SHA全部通过。[解压文件清单](../docs/PUBLICATION_SCOPE.md)逐项列出可直接读取的路径；[独立解压验收](../docs/PUBLICATION_SCOPE.md)及[解压owner退出收据](../docs/PUBLICATION_SCOPE.md)均已落盘。该准备不改变训练准入，权重仅字节复制、未反序列化。

## 本地覆盖与验证

| 资产 | 本轮结果 |
|---|---|
| 60份PDF | 新复制48份；目标目录已有同SHA文件12份，逐篇关联本地路径、SHA、页数及原工作台ID。 |
| 58项仓库记录 | 新复制44份源文件快照；复用13份同commit目标仓库；protein-uq先转入原元数据，再补齐固定commit的通用评价源码。 |
| 原实验与证据 | 复制26份实际实验源文件及328份证据/清单文件。 |
| 物理校验 | 7,698个独特引用文件、7,139,703,687字节已逐文件SHA与大小复核，错误0项。后续新增数据各有官方hash或Git blob验证。 |

主清单：[asset_manifest.json](../docs/PUBLICATION_SCOPE.md)。验证回执：[transfer_verification.json](../docs/PUBLICATION_SCOPE.md)。

## 新取得的真实数据

| 数据 | 文件与实际内容 | 来源校验 |
|---|---|---|
| ProtT3 SwissProtV3 | ZIP148,410,120字节；解出JSONL配对train430,595、validation10,000、test10,000。文件名虽为.json，内容实际为每行一个[sequence,text]。 | 官方OSF SHA256一致；ZIP成员CRC已完整读取验证。 |
| ProtT3 proteinkg25 | ZIP138,380,545字节；解出tab分隔文本train422,315、validation10,000、test10,000，每行2字段。 | 官方OSF SHA256一致；ZIP成员路径/符号链接检查0异常，CRC通过。 |
| OPUS原始基准 | 18份JSON，共26,973文件记录、20,973,373字节。17项为instruction/input/output，MCQ为question/input/options/answer/description。 | revision29f7620de5db692c5cbff8fcc0db2b0117a5bbd2；18份官方Git blob全部一致。 |
| OPUS-InstructionCorpus | 1,534,241,673字节；状态downloaded_payload，完整文件保存在data/raw/continuation_20261016。 | revisionb42fea9b0b9f55ae1c6caedc74cb0d8bbc93226d；官方LFS SHA256验证True。 |
| OPUS-InstructionCorpus-Evol | 3,327,988,260字节；状态downloaded_payload，完整文件保存在data/raw/continuation_20261016。 | revision958f4ccb8e796f892f6740536481b7caa2ca86f1；官方LFS SHA256验证True。 |
| protein-uq GB1 | 原始CSV149,361行；五份切分各8,733行，字段sequence/target/set/validation。 | 固定commit5e7b2b9cd219805eaabe73b21d4d9955cf882448，Git blob核对通过。 |
| protein-uq Meltome | 原JSON221,203观测；FASTA201,283条；mixed_split27,951、human10,093、human_cell7,158行。 | Git blob一致；ZIP内JSON/FASTA/CSV已完整读取且CRC通过。 |

各任务投影、重复测量及不同split互有重叠，表内记录数分别报告。它们不相加充当独立蛋白或实验数。详细源文件SHA、schema和成员计数见 [ProtT3盘点](../docs/PUBLICATION_SCOPE.md)、[protein-uq盘点](../docs/PUBLICATION_SCOPE.md)。

## 准入与剩余文件

新增训练语料和公开benchmark保持原始split并独立隔离。对其做序列/家族、原文来源、类别和答案重叠审计后，才能登记进新的训练协议。当前A1/B1锁及RT源版本gate继续按原协议执行。文件落盘不改变原有训练准入状态。

代码快照保留源码、文档、许可证、来源commit和逐文件SHA。`.git`历史及指定二进制权重省略项逐文件列在manifest；Model Stitching、ProtNLM等原本稀疏/导出式记录保留真实范围。58项表示58个来源条目，完整原系统复现状态由各自执行回执报告。

初轮交付（2026-10-16 07:38）时，ProtNote原始data.zip为17,594,525,298字节，只有官方文件目录/MD5及失败的有界目录探测记录；PepInter两份原权重合计13,912,689,272字节，已有官方元数据/MD5。随后已启动这三个完整文件的补充下载，当前有效状态见末节及实时清单。旧任务中未取得的DependencyMap正式PDF和DeepLoc2017不计入这60份PDF。

本轮网络下载没有遇到需用户登录或验证码的分支。下载的仓库保持静态，未运行安装、训练或下载脚本。

## 目录入口

- [新增论文](../docs/PUBLICATION_SCOPE.md)
- [新增源代码快照](../docs/PUBLICATION_SCOPE.md)
- [新增数据](../docs/PUBLICATION_SCOPE.md)
- [来源、SHA与验证回执](../docs/PUBLICATION_SCOPE.md)


<!-- FULL_PRIMARY_ASSETS_STARTED_20261016 -->
## 原始完整包补充下载

2026-10-16补充下载启动阶段执行3项原始完整文件，共31,507,214,570字节：ProtNote data.zip、PepInter分类及回归checkpoint。官方Zenodo元数据已重新核对，3个完整GET均返回HTTP200及准确Content-Length。文件位于data/raw/continuation_20261016/full_primary_assets，下载中保留.partial和逐文件字节进度。

[补充下载实时清单](../docs/PUBLICATION_SCOPE.md)独立于前述已完成清单；该阶段要求完整文件核对官方MD5并记录SHA256，现已全部完成，终态见下节。下载数据继续隔离，原系统执行状态由后续运行回执给出。

初次完整流的历史回执保存在 [manifest_additional_full.json](../docs/PUBLICATION_SCOPE.md)；恢复和分段记录由实时清单合并，最终三个文件的校验、ZIP/schema盘点及进程终态已列于下节。


<!-- FULL_PRIMARY_STATUS_BEGIN -->
## 三个原始完整文件的当前状态

更新时间：2026-10-16 10:02 Asia/Shanghai。有效实时状态见[manifest_additional_live.json](../docs/PUBLICATION_SCOPE.md)；[初次完整流记录](../docs/PUBLICATION_SCOPE.md)保留原下载历史。

| 文件 | 当前状态 | 实际有效字节／预期字节 | 官方MD5核对 | 本地SHA256 |
|---|---|---:|---|---|
| ProtNote/data.zip | 完整下载且校验通过 | 17,594,525,298／17,594,525,298 | `703141e8eb7a89c83190122f0adfde35（一致）` | `a5efe8fef2411efc96bbaccdc872ed1d58b62d727b2a3d830a117d7db64e8f50` |
| PepInter/Pepinter_classification.ckpt | 完整下载且校验通过 | 6,900,954,326／6,900,954,326 | `759f7649ba6c4b8ff98921420c954dbf（一致）` | `210d6b8e64d86b81209aa203374b929c0e666c50a350c5792bf6c2d22b9919c2` |
| PepInter/pepinter_regression.ckpt | 完整下载且校验通过 | 7,011,734,946／7,011,734,946 | `69cc0f4afd28aff1c0c4a16b005716d1（一致）` | `fa0c5692d080f30308a2a8e2e194758fc97b40a28e96f4cf0c422dc8213a08a4` |

当前完整核验3/3项，共31,507,214,570字节。所有文件保持quarantine，两个ckpt仅做ZIP容器目录读取，未反序列化或执行。ProtNote纯ZIP目录及安全schema盘点已完成。

已完成的权重目录回执：[分类容器](../docs/PUBLICATION_SCOPE.md)、[回归容器](../docs/PUBLICATION_SCOPE.md)。

初轮60份PDF／58项仓库资产清单及冻结基准文件保持原样。普通提前EOF、分段和传输诊断收据保留在sources/continuation_20261016；诊断未修改系统代理或DNS配置。

### ProtNote归档内实际内容

ZIP共82个成员：71个文件和11个目录，声明解压总量29,364,788,715字节；路径/符号链接安全检查异常0项。25份FASTA与7份JSON已流式完整读取并校验成员CRC，读取约2.106GB明文；17个.pt和10个.pkl保持二进制，均未反序列化。

| 原始文件面板 | train | dev | test |
|---|---:|---:|---:|
| 原作者GO random | 418,015 | 52,841 | 51,751 |
| 原作者EC random | 208,823 | 26,724 | 25,892 |

FASTA结构为`>accession label1 label2 ...`加换行序列；GO/EC标签位于header，序列模型及LLM可见输入必须剥离这些答案字段。2024命名的GO训练/测试文件分别417,093／51,646条header记录；zero_shot目录另有63,594条GO和2,445条leaf-node记录。各面板、子集及作者fake_*替代切分分别保存，行数不相加为独立蛋白数。

parenthood JSON为`label_id -> list[label_id]`映射，2019／2024文件分别58,610／60,179个键；5份Proteinfer label-vocab JSON均为32,102个GO标签，逐文件SHA重复组已记录。OBO版本为2019-07-01及2024-06-17。

归档还包含作者ProtNote的5个checkpoint（seed12/22/32/42/52，每个3,412,866,491字节），位于`data/models/ProtNote/`。10:02归档盘点时尚未解压；随后按追加准备任务已完整字节复制到下述新隔离目录，均未加载或反序列化。

完整成员表、逐FASTA计数、JSON字段与SHA重复关系见[安全ZIP与schema盘点](../docs/PUBLICATION_SCOPE.md)。

### 两段下载工程时间线

任务起点07:18、下载起点07:40:26保持不变。原09:40工程cap未完成，保留15,383,190,100字节及原ledger；确认原owner自然退出后，于09:40:57启动独立续传段，绝对截止11:10。ProtNote整包于09:53:40通过MD5/SHA；累计任务耗时9,340.6秒、下载耗时7,994.6秒，17:18整体任务时钟未重置。该cap记录属于下载工程，不计为科学模型执行。

[原阶段cap/absence收据](../docs/PUBLICATION_SCOPE.md)；[最终三包验收](../docs/PUBLICATION_SCOPE.md)；[终态下载owner absence](../docs/PUBLICATION_SCOPE.md)。
<!-- FULL_PRIMARY_STATUS_END -->

## ProtNote追加安全解压验收（10:14）

新隔离根目录为 `PROJECT_ROOT/data/raw/continuation_20261016/full_primary_assets/ProtNote/unpacked_20261016/`，原ZIP保留。82个成员中11个目录、71个文件全部落盘，总29,364,788,715字节；每个目标路径resolve均位于新根目录下，无符号链接、reparse point、路径冲突或覆盖既有文件。文件通过独占临时文件写入、流式CRC/大小核验后改名；逐文件SHA和源member记录在独立清单中。25份FASTA、7份JSON、17个.pt、10个.pkl及其他源文件均保留原始字节和FASTA header。

独立验证重新读取全部71个落盘文件，大小、CRC、SHA与解压清单及归档声明全部一致；32份FASTA/JSON的SHA与先前直接读取ZIP的记录32/32匹配。5份作者checkpoint已可按原member路径定位，权重及pickle内容未反序列化，未启动模型。BLAST索引伴随文件中的.pdb扩展名仍按原数据库文件保存。

解压前空闲424,795,287,552字节，超过29,364,788,715字节目标量加20GiB余量。解压使用2逻辑CPU亲和[18,19]、2流、每流8MiB缓冲，89.188秒，峰值RSS96,915,456字节；独立全量重读验证15.297秒，峰值RSS54,267,904字节。两进程共享本阶段10:27:02截止（启动后15分钟且早于11:10绝对cap），均以0退出；10:14:28检查owner PID35168/63668及对应命令均不存在，无需终止进程。

原下载终态回执及两段下载cap历史完整保留。这次解压是已下载资料的隔离落盘准备；原冻结基准、gold、训练锁与科学模型执行记录保持原状态。[逐文件清单](../docs/PUBLICATION_SCOPE.md)、[独立全量重读验收](../docs/PUBLICATION_SCOPE.md)、[独立owner absence](../docs/PUBLICATION_SCOPE.md)。
