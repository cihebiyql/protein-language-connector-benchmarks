# 公开内容与复现范围

本目录公开多任务基础测试框架、固定任务登记、CPU代码与测试、完整小型结果、许可及来源清单。新建的Meltome均值／组成Ridge基线与四个历史预测面板的独立复算分开记录；这些结果不代表V4连接器已经完成多任务训练或取得多任务提升。

Meltome使用[FLIP官方仓库固定版本](https://github.com/J-SNACKKB/FLIP/tree/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/meltome)。上游版本为 `62cace8735f5610e2743cf06ce0f944b37fffaa6`，保留官方human划分7287／861／1945，共10093行。原始研究来源为[Meltome研究](https://doi.org/10.1038/s41592-020-0801-4)。下载URL、字节数、SHA及压缩包成员详见 `data/meltome/download_manifest.json`。

以下原始数据保留在研究工作目录，未重复上传到本仓库。公开下载命令从固定官方URL获取并核对SHA，不重新划分或删行。

| 文件 | SHA-256 |
|---|---|
| `splits.zip` | `6da88cb6f699f814054a998e7b2a5c6796a6907de391f50aa156a5b4c1ed7152` |
| `human.csv`（包内 `splits/human.csv`） | `cfafab67d5e29188fe0846ab48cdf6aff36aaf4ccd635b7b66bb4002a939822c` |
| `meltome_PIDE20_clusters.tsv` | `681bfa23a56c3faf8c8e380ddbb66efc19bbbe22e28e3e58fdc32ba3ccdd3866` |

在本目录、已有NumPy的Python环境中可运行：

```bash
python src/meltome.py download --config configs/meltome_human_v1.json
python src/meltome.py audit --config configs/meltome_human_v1.json
python src/meltome.py run --config configs/meltome_human_v1.json --run-id github_reproduction_v1
```

使用新的单段run-id；已有结果目录拒绝覆盖。模型仅用训练集拟合，Ridge参数由完整验证集选择，固定模型后才评价完整测试集。公开的原始基线执行源码 `results/meltome_human_v1/cpu_baseline_v1/source_meltome.py` SHA为 `90c163bbe0fec947a08837d3804f60d1661fb1bfd78cb00b32ad71fa4955fb43`；现行 `src/meltome.py` SHA为 `c410ff15f1b93cbd595156fee5e03c4ee1f4854ad4a60a59235897103d2ac093`。后者增加固定版本下载入口与输入检查，原始run快照和双版本验收记录均保留。

完整统一入口 `run_suite.py` 同时包含历史面板复算。历史复算依赖原项目canonical数据及预测布局；本公开目录保留复算结果和来源文件SHA，未附带完整私有历史输入。不能据此声称公开clone单独即可完整重放四个历史面板。具有原始布局时，按README指定 `--project-root` 和新run-id运行。

FLIP修改后划分的许可声明及归属按[官方Meltome说明](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/meltome/README.md)保留；[FLIP许可原文](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/LICENSE)为AFL-3.0。本目录包含未修改的 `FLIP_meltome_README.md` 与 `FLIP_LICENSE`。原始研究数据的许可声明与FLIP修改后划分的许可分别保留，不扩展为本项目所有资料的统一新许可。

`publication_manifest.json` 列出每个源文件与公开文件的SHA、字节数，以及排除文件及理由。代码、配置、现有验收收据和结果均保持源字节；未发布Python缓存。定时检查已经按用户要求关闭，既有node1训练队列继续按其冻结计划执行。
