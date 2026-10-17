# node1共享资源策略：独立运行版本

本目录保存用户明确授权共享资源后的独立运行源码。旧版根目录的源码、合同、数据、评分和已发表结果均保持原样；只有监督器、队列和合同三个运行管理模块与旧版本不同，其余九个模型／数据／评分源码逐字节一致。

新策略最多同时使用三个GPU worker，所选GPU为4、5、3。CPU空闲率、I/O等待、GPU利用率及已有计算进程只作观察，不阻止启动；保留主机内存、每worker RSS、GPU剩余显存、存储、独立进程组监督及原始绝对截止。不能终止其他工作负载。

`USER_RESOURCE_POLICY.json` 是公开路径规范化副本，唯一替换是私有远端输出根目录。真实执行原件SHA为 `56a404606b5a66883c16f638209144d37f0bdf9d9e0bf958f13b57975ee62e31`；公开字节SHA另见 `publication_manifest.json`。该副本不能冒充原合同绑定的执行字节；核验原始合同使用原件SHA。

本目录configs保留开发／TRAIN资格版本；实际5epoch冻结配置、评价协议及12:29状态位于snapshots/started_v1。正式队列已经启动，尚无通过正式验收的完成组；最新解释见STATUS.md。原03:26:57北京时间截止不因版本变化重置。旧本地工程失败只属于原尝试，本版本未使用尚在开发的本地低内存加载器。

本目录直接使用共享版 `freeze_formal_shared.py` 和 `analyze_v4_shared.py`。旧的非共享冻结／分析测试继续保留在上级版本，不混入本目录；新增共享策略及冻结夹具只验证资源政策，不代表科学效果。

在本目录使用Linux或WSL，可执行以下无模型CPU检查：

```bash
export MM_STANDARD_ROOT="$PWD/standard"
export CUDA_VISIBLE_DEVICES=""
python -S -B -m unittest tests.test_proc_race tests.test_supervise tests.test_queue tests.test_node_shared_policy tests.test_shared_freezer -v
python -S -B -m unittest discover -s tests -p test_evaluate.py -v
python -S -B -m unittest discover -s tests -p test_archive.py -v
```

原始模型版本和输入SHA保持不变。复现时将该源码快照部署为独立的新执行scope、显式配置模型与旧冻结输入路径，并依据实际原始政策和新收据重新冻结；不能把公开路径副本直接当作原远端运行目录。
