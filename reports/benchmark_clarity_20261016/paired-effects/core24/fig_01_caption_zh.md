图1。两个冻结A1场景的五项预先固定核心配对效应：depth_real、depth_neutral、content_early、content_mid及DiD，横轴为MAE差（pKD）；E/M分别表示early/mid，R/N分别表示real/neutral训练系统。

灰度圆、方、三角分别表示seed17、42、2026的同seed配对差值，黑菱形及误差条引用原summary的三seed均值±sample_SD（k＝3），SD不是CI；所有30个原值均保留，无清洗、插补、p值或推断检验。同一冻结test测量不因seed重复成为新的独立生物样本，门控与池化属于预登记扩展阶段而未纳入此核心主图，原summary保留全部记录。

两个场景的depth_real均为两seed负、一seed正，原均值分别为−0.1091与−0.0372 pKD，当前三seed未表现出mid方向稳定优势；双新content_mid和DiD均在seed17、42为正、seed2026为负。content是分别训练θ_R与θ_N系统的总效应，不能解释为固定θ的推理消息使用；DiD与其余contrast存在代数依赖，正DiD也不保证mid绝对更好。
