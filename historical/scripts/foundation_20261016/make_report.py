"""Derive the readable foundation report from real result artifacts.

Original GPU controller reports remain immutable. This reader uses the actual
BASE03 schema and keeps coverage, conditional error, and full-population error
separate. It never substitutes zero for an unavailable metric.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

from foundation import ROOT, OUT, build_index, load, write_json

LABELS = {'input_mlp': '普通输入MLP', 'qformer_style': 'Q-Former风格',
          'internal_cross_attention': '内部跨注意力', 'constant': '独立训练常数消息',
          'BASE03': '预测器→冻结Qwen'}

def number(value, digits=5):
    return '未定义' if value is None else f'{value:.{digits}f}'

def normalized_generation(metrics, *, pipeline=False):
    """Validate denominators before exposing a conditional or full metric."""
    names = ('n_total', 'n_valid', 'n_failed')
    counts = [metrics.get(key) for key in names]
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
        raise ValueError('Generation report requires explicit nonnegative integer denominators')
    total, valid, failed = counts
    if total == 0 or valid + failed != total:
        raise ValueError('Generation report has inconsistent denominator or empty population')
    conditional = metrics.get('generated_pKD_mae_valid_only') if pipeline else (metrics.get('regression_valid_only') or {}).get('mae')
    full = metrics.get('full_sample_generated_mae' if pipeline else 'full_sample_mae')
    for value in (conditional, full):
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                                  or not math.isfinite(value) or value < 0):
            raise ValueError('MAE must be finite nonnegative or undefined')
    if valid == 0 and conditional is not None:
        raise ValueError('Conditional MAE cannot exist without valid predictions')
    if failed and full is not None:
        raise ValueError('Full-population MAE cannot ignore failed predictions')
    if valid == total and conditional is not None and full is not None and not math.isclose(conditional, full, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError('Full and conditional metrics disagree on identical population')
    return {'n_total': total, 'n_valid': valid, 'n_failed': failed,
            'valid_only_mae': conditional, 'full_sample_mae': full}

def paired_valid(directory):
    paths = [directory / ('test_predictions_' + mode + '.jsonl') for mode in ('correct', 'off')]
    if not all(path.exists() for path in paths):
        return None
    values = [{row['sample_id']: row for row in (json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip())}
              for path in paths]
    assert set(values[0]) == set(values[1]), 'Correct and off populations differ'
    retained = [key for key in values[0] if all(items[key]['parse']['valid'] for items in values)]
    if not retained:
        return {'n': 0, 'correct_mae': None, 'off_mae': None}
    scores = [sum(abs(items[key]['parse']['value'] - items[key]['gold_pkd']) for key in retained) / len(retained)
              for items in values]
    return {'n': len(retained), 'correct_mae': scores[0], 'off_mae': scores[1],
            'scope': 'Conditional on both modes returning valid JSON; not full test population'}

def group_error_interval(gold, candidate, baseline, groups, draws=2000, seed=2026):
    """Descriptive paired resampling of physical source/family components."""
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 2:
        raise ValueError('At least two bootstrap draws are required')
    if not gold or len({len(gold), len(candidate), len(baseline), len(groups)}) != 1:
        raise ValueError('A common nonempty paired population is required')
    if any(not group for group in groups):
        raise ValueError('Physical independent groups are required')
    if any(not math.isfinite(float(x)) for values in (gold, candidate, baseline) for x in values):
        raise ValueError('Finite paired predictions required')
    components = {}
    for target, pred, control, group in zip(gold, candidate, baseline, groups):
        entry = components.setdefault(group, [0., 0])
        entry[0] += abs(control - target) - abs(pred - target)
        entry[1] += 1
    ids = sorted(components)
    if len(ids) < 2:
        return {'status': 'undefined_fewer_than_two_independent_groups', 'n': len(gold), 'groups': len(ids)}
    rng = random.Random(seed)
    sampled = []
    for _ in range(draws):
        chosen = rng.choices(ids, k=len(ids))
        sampled.append(sum(components[g][0] for g in chosen) / sum(components[g][1] for g in chosen))
    sampled.sort()
    def quantile(q):
        value = q * (len(sampled) - 1)
        left = int(value)
        return sampled[left] + (sampled[min(left + 1, len(sampled) - 1)] - sampled[left]) * (value - left)
    return {'status': 'complete_descriptive_fixed_checkpoint', 'n': len(gold), 'groups': len(ids),
            'baseline_minus_candidate_MAE': sum(value[0] for value in components.values()) / len(gold),
            'paired_group_95_interval': [quantile(.025), quantile(.975)], 'draws': draws, 'seed': seed,
            'training_seed_uncertainty_included': False, 'multiple_comparison_control': False}

def cross_method_intervals(rows):
    result = []
    for scenario in sorted({r['scenario'] for r in rows}):
        samples = ROOT / 'data/processed/bench_v02/a1_locked_v2' / scenario / 'samples.jsonl'
        if not samples.exists():
            continue
        gold_rows = [json.loads(line) for line in samples.read_text(encoding='utf8').splitlines() if line.strip()]
        gold_rows = [row for row in gold_rows if row['split'] == 'test']
        ids = [row['sample_id'] for row in gold_rows]
        values = {}
        for row in rows:
            if row['scenario'] != scenario or row['mode'] not in {'correct', 'source_predictor_numeric'}:
                continue
            directory = Path(row['result_path']).parent
            path = directory / ('test_predictions.jsonl' if row['mode'] == 'source_predictor_numeric' else 'test_predictions_correct.jsonl')
            if not path.exists():
                continue
            predictions = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
            if [p['sample_id'] for p in predictions] != ids:
                raise ValueError('Cross-method paired population differs from physical gold')
            if row['mode'] == 'source_predictor_numeric':
                values['source_predictor_numeric'] = [p['source_predictor_pKD'] for p in predictions]
            elif all(p['parse']['valid'] for p in predictions):
                values[row['method']] = [p['parse']['value'] for p in predictions]
        for candidate, baseline in [('qformer_style', 'input_mlp'), ('internal_cross_attention', 'input_mlp'),
                                     ('input_mlp', 'constant'), ('qformer_style', 'constant'),
                                     ('internal_cross_attention', 'constant'), ('qformer_style', 'source_predictor_numeric'),
                                     ('input_mlp', 'source_predictor_numeric'), ('internal_cross_attention', 'source_predictor_numeric')]:
            if candidate in values and baseline in values:
                result.append({'scenario': scenario, 'candidate': candidate, 'baseline': baseline,
                    **group_error_interval([r['pkd'] for r in gold_rows], values[candidate], values[baseline],
                                           [r['group_id'] for r in gold_rows])})
    return result

def main():
    index = build_index()
    rows = []
    for suite in sorted(OUT.glob('gpu_suite*')):
        if not suite.exists():
            continue
        for path in sorted(suite.rglob('result.json')):
            result = load(path)
            if not result or result.get('status') != 'complete':
                continue
            registration = load(path.parent / 'run_preregistration.json') or {}
            config = registration.get('configuration', {})
            method = result.get('method') or registration.get('method') or config.get('method')
            scenario = result.get('scenario') or registration.get('scenario') or config.get('scenario')
            if method not in LABELS:
                continue
            if method == 'BASE03':
                metrics = result['metrics'].get('test', {})
                row = {'method': method, 'scenario': scenario, 'mode': 'structured_predictor',
                       **normalized_generation(metrics, pipeline=True), 'new_optimizer_steps': 0,
                       'new_trainable_parameters': 0, 'source_predictor_selected_architecture': result.get('selected_architecture'),
                       'result_path': str(path), 'scope': result.get('scope'),
                       'metrics_raw': metrics}
                rows.append(row)
                # The numerical predictor channel is a separate real baseline;
                # its value is not substituted into the LM's generated answer.
                rows.append({'method': method, 'scenario': scenario, 'mode': 'source_predictor_numeric',
                    'n_total': metrics['n_total'], 'n_valid': metrics['n_total'], 'n_failed': 0,
                    'valid_only_mae': metrics['source_predictor_mae_same_all_examples'],
                    'full_sample_mae': metrics['source_predictor_mae_same_all_examples'],
                    'new_optimizer_steps': 0, 'new_trainable_parameters': 0,
                    'source_predictor_selected_architecture': result.get('selected_architecture'),
                    'result_path': str(path), 'scope': 'Numerical output of the frozen predictor; not generated by the LM'})
            else:
                for mode, metrics in result.get('test_generation', {}).items():
                    rows.append({'method': method, 'scenario': scenario, 'mode': mode,
                        **normalized_generation(metrics),
                        'new_optimizer_steps': result.get('optimizer_steps'),
                        'new_trainable_parameters': registration.get('capacity', {}).get('trainable_parameters'),
                        'best_epoch': result.get('best_epoch'), 'actual_epochs': result.get('actual_epochs'),
                        'scope': registration.get('scope'), 'result_path': str(path),
                        'paired_valid_comparison': paired_valid(path.parent)})
    updated = datetime.now(timezone.utc).isoformat()
    intervals = cross_method_intervals(rows)
    derived = {'updated_utc': updated, 'results': rows, 'descriptive_paired_intervals': intervals,
               'information_results': index.get('information_results', []),
               'science_counts_exclude_cpu_fixtures': True,
               'budget_comparison': {
                   'comparison_group': 'four newly trained connector/control methods; BASE03 remains a separate frozen-predictor inference pipeline',
                   'four_neural_methods_same_backbones_data_allowed_context_epochs_batch_and_seed': True,
                   'parameters_and_communication_and_flops_matched': False,
                   'input_mlp_and_qformer_prefix_tokens': 16,
                   'internal_attention_memory': 'all valid concatenated peptide and target residue keys; prompt queries read this memory',
                   'long_target_pooling': 'ordinary input MLP pools concatenated memory; peptide detail can be diluted by a long target',
                   'constant_parameter_budget': '16 learned LM-space tokens; not parameter-matched to the nonlinear connectors',
                   'sole_insertion_location_causal_attribution': False},
               'scope': 'One seed, three epochs, A1 broad-reference-target generation foundation; prior 30-epoch predictors and original published systems remain separate'}
    write_json(OUT / 'derived_comparison.json', derived)

    asset_notes = []
    if (index.get('additional_full_asset_verification') or {}).get('passed') is True:
        asset_notes += ['三个补充完整包共31,507,214,570字节已通过官方MD5和本地SHA校验；[最终验收](../sources/continuation_20261016/additional_full_verification.json)保留原阶段未完及续传身份。', '']
    if (index.get('protnote_unpack_verification') or {}).get('passed') is True:
        asset_notes += ['ProtNote已安全解压71文件、29,364,788,715字节并独立重读验收，其中含五个作者模型；[逐文件清单](../sources/continuation_20261016/unpack_protnote_manifest.json)。FASTA header含答案标签，正式模型输入需分离这些标签。', '']
    text = ['# 蛋白质语言连接器基础测试', '', '本页从真实运行文件生成，历史结果与新运行分别记录。', '',
            '## 资料与入口', '',
            '[资产落盘报告](assets_continuation_20261016.md) · [逐文件清单](../sources/continuation_20261016/asset_manifest.json) · [完整包实时进度](../sources/continuation_20261016/manifest_additional_live.json)', '',
            '[工作台正文与检索核验](workbench_continuation_20261016.md)：60/60份PDF正文已导入，60/60题名检索命中；本轮修复15篇历史失败的语义索引，补建1339条向量，逐篇无缺失候选。方法笔记审核与检索就绪分别记录。', '',
            '60份PDF与58项仓库资源已在本项目建立本地对应。新增语料保持隔离，不进入当前冻结划分。完整ProtNote/PepInter包的状态以实时清单和最终校验为准。', '',
            *asset_notes,
            '## 本轮实际执行', '',
            '- 重放18个既有A1/B1检查点，并重新拟合6个控制；这是本轮复验，原18次训练保持历史身份。',
            '- 新A1生成式基础阶段使用固定ESM2-150M残基缓存与冻结Qwen2.5-1.5B；四种连接器按相同数据、种子17、3轮和相同批量执行。',
            '- BASE03使用验证集选定的既有预测器，再由同一个冻结Qwen生成数值；本轮不重新训练该预测器或LLM。',
            '- RT目标域诊断已使用7,070条真实记录，测试1,063条；原始跨条件RT任务继续保留数据准入缺项。', '',
            '## 新运行的生成与数值通道结果', '',
            'MAE单位为pKD。全样本MAE仅在该模式全部输出有效时定义；有失败时保留条件误差与失败数。各场景独立报告。', '']
    for scenario, title in [('new_peptide_known_target', '新肽与已见精确靶点'), ('new_peptide_new_target', '新肽与新靶点家族')]:
        text += ['### ' + title, '', '| 方法 | 消息模式 | 有效／总数 | 全样本MAE | 有效子集MAE | 新可训练参数 |', '|---|---|---:|---:|---:|---:|']
        selected = [row for row in rows if row['scenario'] == scenario]
        if not selected:
            text += ['| 尚无完成结果 | — | — | 未定义 | 未定义 | — |']
        for row in selected:
            text.append(f"| {LABELS[row['method']]} | {row['mode']} | {row['n_valid']}/{row['n_total']} | {number(row['full_sample_mae'])} | {number(row['valid_only_mae'])} | {row['new_trainable_parameters']} |")
        pairs = {row['method']: row.get('paired_valid_comparison') for row in selected
                 if row['mode'] == 'correct' and row.get('paired_valid_comparison')}
        if pairs:
            text += ['', '同一批两种模式都有效的样本，可作条件配对比较：', '', '| 方法 | 共同有效数 | correct MAE | off MAE |', '|---|---:|---:|---:|']
            for method, paired in pairs.items():
                text.append(f"| {LABELS[method]} | {paired['n']} | {number(paired['correct_mae'])} | {number(paired['off_mae'])} |")
        text += ['']
    text += ['### 比较范围', '',
             '四种新训练方法匹配数据、底座、允许上下文、训练轮数、批量与种子。普通MLP和Q-Former风格使用16个前缀，内部注意力读取全部有效残基键；连接器参数量与计算量也不同。长靶点可能稀释普通MLP对肽的池化信息。因此当前表格检验这些具体实现，单独解释为接入层位置的因果收益还需要额外预算控制。常数控制使用16个可训练软token，其参数量小于非线性连接器。', '',
             '新生成式结果为单种子、3轮基础运行。原有预测器为3种子、30轮，训练目标也不同，分别列示，不拼成严格同预算排行榜。', '',
             '双新场景首次运行在训练第28批出现CUBLAS内部错误，失败记录独立保留。恢复运行保持数据、算法、批量、种子和训练轮数，启用进程内同步CUDA错误定位；未调整测试答案、提示词或全局环境。成功重放不能单独确定原故障原因，bf16轨迹也不保证逐位相同。', '',
             'BASE03另报源预测器的真实数值通道。源预测值未用于修正或替换LLM原始回答；若LLM重算单位或输出失败，其错误仍计入生成通道。这种零样本呈现失败不能证明新连接器超过了充分优化的预测器与LLM系统。', '',
             'BASE03采用逐样本生成，新连接器采用批量生成；其来源预测器属于既有训练，当前只做零训练推理。原始耗时保留在收据中，不能把不同批量与执行流程的耗时直接当作模型架构的吞吐优势。', '',
             '### 固定检查点的描述性配对区间', '',
             '以下按冻结来源／家族连通组重采样2000次，seed2026。正差值表示候选MAE较低。区间仅反映当前固定检查点的测试组抽样，未包含训练种子波动或多重比较控制。', '',
             '| 场景 | 候选 | 对照 | 对照减候选MAE | 95%区间 | 独立组 |', '|---|---|---|---:|---|---:|']
    for interval in intervals:
        ci = interval.get('paired_group_95_interval')
        text.append(f"| {interval['scenario']} | {interval['candidate']} | {interval['baseline']} | {number(interval.get('baseline_minus_candidate_MAE'))} | {str([round(v, 5) for v in ci]) if ci else '未定义'} | {interval['groups']} |")
    text += ['', '## 消息作用与条件可读出性诊断', '',
             'C1逐样本计算正确消息与关闭消息时的答案token NLL差，覆盖完整测试集。C2仅提取提示词状态，训练集拟合投影、标准化及岭回归，验证集选正则；每种表示增加相同128维。B为组成、化学属性及完整允许实验上下文。正ΔMSE表示在B之外降低测试误差，单位pKD²；它不等于互信息比特数或生物因果作用。', '',
             '| 场景 | 方法 | C1测试数 | G（nats/token） | B测试MSE | raw ΔMSE | message ΔMSE | receiver ΔMSE |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    accepted_information = index.get('information_results', [])
    for entry in accepted_information:
        actual = load(entry['result_path'])
        c1, probes = actual.get('C1', {}), actual.get('C2', {}).get('models', {})
        deltas = [probes.get(name, {}).get('delta_MSE_vs_B') for name in ('raw', 'message', 'receiver')]
        text.append(f"| {entry['scenario']} | {LABELS.get(entry['method'], entry['method'])} | {c1.get('n_test', '—')} | {number(c1.get('G_all_test'))} | {number(probes.get('baseline_zero128', {}).get('test', {}).get('mse'))} | " + ' | '.join(number(v) for v in deltas) + ' |')
    if not accepted_information:
        text.append('| 尚无通过执行验收的真实产物 | — | — | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |')
    text += ['', f"已通过执行验收{len(accepted_information)}/8项；[信息诊断汇总](information_foundation_20261016.md)和[原始产物目录](../results/foundation_20261016/information_diagnostics)保留分组区间、协议和收据。严格独立匹配供体两场景均为0，P/H保持未定义。", '']
    text += ['raw是肽与靶点的ESM平均向量拼接；message是前缀均值，或内部注意力在末提示位置的输出投影；receiver是末解码层的末提示位置状态。这些摘要再分别投影至128维。本轮衡量这些摘要的线性可读出性，不等于完整残基表示的信息保留率。', '']
    text += ['B中的文本上下文采用固定维哈希编码，可能有损。receiver读出收益也可能来自语言模型对提示词上下文的重新编码；需结合raw、message和常数控制解读，不能单凭receiver误差下降归因为蛋白来源信息。', '']
    text += ['本轮覆盖A1消息关闭与线性条件读出的基础档。条件探针、控制任务、交换干预、证据删除和原子事实评价的原论文及页码见[评价原文证据](../sources/continuation_20261016/prior_share_campaign/evaluation/evidence.md)。随机标签控制、MLP探针容量曲线、slot删除曲线和生物反事实／残基证据评价尚未在本轮执行，常数消息控制不替代这些实验。', '']
    text += ['',
             '## 已验证的预测器与数值可靠性', '',
             '[18个检查点重放、6个重新拟合及12个校准诊断](predictor_replay_20261016.md)。已见靶点的名义90%区间实测覆盖86.18%—89.47%；双新场景区间更宽。验证集曾参与检查点选择，校准结果按描述性诊断报告。', '',
             '## 保留时间目标域诊断', '',
             '[数据与运行收据](rt_target_foundation_20261016.md)。在1,063条测试记录上，Ridge MAE3.9404分钟、RMSE5.2754、Spearman0.92128；kNN MAE7.5596，常数10.9167。Ridge名义90%区间实测覆盖87.77%，保留欠覆盖。此测试使用PXD064311 March目标条件，原源域iRT与化学身份问题保持隔离。', '',
             '## 尚待金标准或原系统准入的项目', '',
             '| 项目 | 当前边界 |', '|---|---|',
             '| A1严格同条件肽对 | 当前合格金标准0条；广覆盖pKD结果不替代同条件方向验证 |',
             '| B1 | 已复验既有预测器；测试仅两个未见功能类 |',
             '| B2知识组合 | 证据登记不等于问答金标准，已核准QA为0 |',
             '| C1/C2 | correct/off及独立信息诊断按实际产物记录；严格不同连通组的匹配供体为0，错配P/H不补造 |',
             '| C3原子事实 | 尚缺独立仲裁的事实标注 |',
             '| OPUS/ProtT3/ProtNote/DeepLC/PepInter原系统 | 数据与权重获取和原系统真实复现分别记录，未运行者不计为复现 |', '',
             '[五套原作者系统的逐项准入矩阵](original_system_readiness_20261016.md)分别列出已存在的真实数据和模型、缺失权重、环境包、来源门槛与作者文档。硬件最低需求未经本地原系统实测的项目保留未知。', '',
             '## 复用命令', '',
             '在项目根目录运行：', '', '```powershell',
             'python scripts/foundation_20261016/foundation.py --step status',
             'python scripts/foundation_20261016/foundation.py --step tests',
             'python scripts/foundation_20261016/make_report.py', '```', '',
             'GPU运行由独立控制器先冻结合同、实时检查资源并监督进程组；既有输出目录拒绝覆盖。', '',
             '[机器索引](../results/foundation_20261016/foundation_index.json) · [派生比较数据](../results/foundation_20261016/derived_comparison.json) · [独立审计](foundation_independent_review_20261016.md)']
    path = ROOT / 'reports/foundation_results_20261016.md'
    path.write_text('\n'.join(text) + '\n', encoding='utf8')
    start = ['# 蛋白质语言连接器 基础测试入口', '',
             '[当前实测与范围](reports/foundation_results_20261016.md)', '',
             '[文件及数据集落盘](reports/assets_continuation_20261016.md)', '',
             '[工作台正文与检索状态](reports/workbench_continuation_20261016.md)', '',
             '[逐方法机器状态](results/foundation_20261016/foundation_index.json)', '',
             '[实际运行合同与记录](results/foundation_20261016)', '',
             '本次新增文件与历史v0.1/v0.2结果分开保存，既有README与冻结结果保持原样。']
    (ROOT / 'START_HERE.md').write_text('\n'.join(start) + '\n', encoding='utf8')
    print(json.dumps({'report': str(path), 'result_channel_rows': len(rows), 'updated_utc': updated}, ensure_ascii=False))

if __name__ == '__main__':
    main()
