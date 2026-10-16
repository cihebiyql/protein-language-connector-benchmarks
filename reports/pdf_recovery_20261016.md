# 两篇缺失 PDF 的有界复核

2026-10-16 按 paper-lab 流程重新核查官方、开放获取、作者和机构来源。本轮新增有效 PDF **0 篇**。原有 60 份有效 PDF 的统计不变；网页、错误页和开放获取元数据均未计作 PDF。

| 论文 | 本轮实际结果 | 停止条件 |
|---|---|---|
| DependencyMap：A proteome-wide dependency map of protein interaction motifs，DOI 10.1038/s41594-026-01762-2 | 正式版页面触发 Nature 身份服务重定向，工具未能读取。官方预印本 DOI 成功解析到 bioRxiv v2 页面，标题、DOI、作者及 2025-11-15 日期相符；页面给出的真实 PDF 链接返回 HTTP 429、HTML 内容。 | 收到 429 后立即停止 bioRxiv 主机访问，未猜测其他下载端点。正式版及预印本 PDF 均仍缺。 |
| DeepLoc 2017：DeepLoc: prediction of protein subcellular localization using deep learning，DOI 10.1093/bioinformatics/btx431 | OpenAlex 标注 bronze OA，但其提供的 OUP 官方 PDF 地址实际返回 HTTP 403、标题为 “Just a moment...” 的 HTML。哥本哈根大学记录仅链接 DOI；作者主页未提供该篇 PDF；既有 DeepLoc 原始源码快照中无 PDF。 | 停止 OUP 主机重试，未操作验证码或机构登录。原始 2017 PDF 仍缺。 |

bioRxiv 普通文章页包含“转发给朋友”表单的 CAPTCHA 字样，最初关键词检测误把它标成访问验证页。核对文章元数据与表单位置后已在回执中保留原判定及更正说明；本轮只读取了页面公开给出的 PDF 链接，没有操作该表单。随后的 PDF 请求确实返回 429，因此停止。

本轮保留了精确 URL、重定向、HTTP 状态、响应类型、大小及 SHA256，详见[机器回执](../docs/PUBLICATION_SCOPE.md)和[原始响应证据](../docs/PUBLICATION_SCOPE.md)。有效 PDF 的页数与标题绑定验证均为未完成，未伪造校验值，也未执行工作台导入。

DependencyMap 的既有[正式版网页全文文本](../docs/PUBLICATION_SCOPE.md)可继续阅读；它的来源与 PDF 交付状态分别记录。DeepLoc 2.0 和 2017 更正页也未替代原论文。

本次仅新增上述回执、报告和证据目录文件。旧 PDF、主资产清单、冻结数据、模型及运行合同均未修改。

来源：[DependencyMap 正式页面](https://www.nature.com/articles/s41594-026-01762-2)、[官方预印本](https://doi.org/10.1101/2024.09.11.612445)、[DeepLoc 机构记录](https://researchprofiles.ku.dk/en/publications/deeploc-prediction-of-protein-subcellular-localization-using-deep/)、[作者主页](https://jjalmagro.github.io/)。
