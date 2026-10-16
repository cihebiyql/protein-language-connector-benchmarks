图1。两个冻结A1场景下五项预先固定核心配对contrast：depth_real、depth_neutral、content_early、content_mid与DiD，横轴为MAE差（pKD）。

灰度圆、方、三角分别表示同一seed下两系统的配对差值（17、42、2026）；仅k＝3时以黑菱形和误差条显示原summary均值±sample_SD，误差条为seed间SD而非CI。原null保留coverage，未插补、删seed或实施推断检验；冻结test测量不因seed重复变成新增独立生物样本。

已见靶点depth_real的原三seed均值为-0.1091；双新depth_real的原三seed均值为-0.0372。深度正值表示相应内容条件下mid的MAE更低；content是独立训练θ_R与θ_N系统的总效应，不是固定θ推理消息使用。DiD由其他contrast代数决定，正值不保证mid绝对更好；门控和池化属于预登记扩展阶段，因此未纳入此核心主图，原summary保留全部记录。
