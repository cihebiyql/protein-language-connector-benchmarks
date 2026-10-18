"""Descriptive fixed-seed paired gains with original values retained."""
# CONFIGURATION
from pathlib import Path
INPUT_FILES = [Path(__file__).resolve().parents[1] / 'reports/independent_math_review_v1/summary.json']
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'analysis/multiseed-stability-figure'
PROJECT_TITLE = 'M3 compared with M0 on the fixed three-task benchmark'
CN_SERIF_FALLBACK = ['SimSun', 'Songti SC', 'Source Han Serif SC', 'Noto Serif CJK SC', 'serif']
EN_SERIF_FALLBACK = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']
OUTPUT_FORMATS = ['pdf', 'png', 'jpg']
DPI = 300
ALPHA = None
SIGNIFICANCE_MARKS = {}
NS_LABEL = ''
ERROR_BAR = 'SD'
NORMALITY_ALPHA = None
DEFAULT_DECIMALS = 5
SCIENTIFIC_LOW = 1e-3
SCIENTIFIC_HIGH = 1e5
THOUSANDS_SEP = ','
FIG_SINGLE_COL = (3.5, 2.6)
FIG_DOUBLE_COL = (10.6, 3.9)
# END CONFIGURATION

import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd

def main():
    assert OUTPUT_DIR.resolve() != INPUT_FILES[0].resolve().parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    for name in (*('figures/' + format for format in OUTPUT_FORMATS), 'process_data'):
        (OUTPUT_DIR / name).mkdir(parents=True)
    summary = json.loads(INPUT_FILES[0].read_text(encoding='utf-8'))
    records = [{key: row[key] for key in ('task', 'backbone')} | pair
        for row in summary['M3_contrasts'] if row['baseline'] == 'M0' for pair in row['pairs']]
    frame = pd.DataFrame(records)
    assert frame.shape == (18, 6)
    assert not frame.isna().any().any()
    assert not frame.duplicated(['task', 'backbone', 'seed']).any()
    assert all(set(values['seed']) == {17, 42, 2026} for _, values in frame.groupby(['task', 'backbone']))
    inventory = [{'column': name, 'dtype': str(frame[name].dtype), 'missing': int(frame[name].isna().sum()),
        'unique': int(frame[name].nunique())} for name in frame.columns]
    describe = frame.groupby(['task', 'backbone'])['gain'].describe()
    lines = ['# 配对差值图数据核验', '', f'数据形状：{len(frame)}行、{len(frame.columns)}列；内存约{frame.memory_usage(deep=True).sum()}字节。', '',
        '|列|类型|缺失|唯一值|', '|---|---|---:|---:|']
    lines += [f"|{row['column']}|{row['dtype']}|{row['missing']}|{row['unique']}|" for row in inventory]
    lines += ['', '复合键task/backbone/seed无重复，18个预定配对全部保留。数据未清洗、剔除或补值。', '',
        '数值描述按任务和底座分别保存于过程表，避免混合不同量纲。每层只有三个固定训练种子，当前图为描述性散点，不开展正态性检验、显著性检验或p值筛选。', '',
        '1.5B seed42在回归任务的退化属于本轮观测结果，完整保留；不以离群规则删除。seed是类别，baseline/M3/gain是关联的测量和确定性差值，不是多变量回归预测变量，VIF与跨任务相关矩阵不适用。', '',
        '误差线表示三个同种子配对增益的样本标准差，既非SEM也非置信区间。三个种子是训练重复，不是独立生物重复。', '']
    (OUTPUT_DIR / '00_data_health.md').write_text('\n'.join(lines), encoding='utf-8')
    frame.to_excel(OUTPUT_DIR / 'process_data/01_paired_gains.xlsx', index=False, sheet_name='paired_gains')
    frame.to_excel(OUTPUT_DIR / 'process_data/fig_01_data.xlsx', index=False, sheet_name='exact_plot_data')
    with pd.ExcelWriter(OUTPUT_DIR / 'process_data/99_summary_stats.xlsx', engine='openpyxl') as writer:
        describe.to_excel(writer, sheet_name='within_stratum_describe')
        pd.DataFrame(inventory).to_excel(writer, sheet_name='column_inventory', index=False)
    available = {font.name for font in font_manager.fontManager.ttflist}
    def resolve(candidates): return next((name for name in candidates if name in available or name == 'serif'), 'serif')
    english, chinese = resolve(EN_SERIF_FALLBACK), resolve(CN_SERIF_FALLBACK)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': [english, chinese, 'serif'],
        'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 10,
        'axes.unicode_minus': False, 'axes.grid': True, 'axes.axisbelow': True,
        'grid.color': '#e7e7e7', 'grid.linewidth': .5, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE_COL, sharey=True)
    task_names = [('meltome', 'A  Thermal stability', 'Spearman gain (unitless)'),
                  ('elm', 'B  Function retrieval', 'Class-macro MRR gain (unitless)'),
                  ('rt', 'C  Retention time', 'MAE reduction (min)')]
    backbones = ['qwen25_1p5b', 'qwen3_4b_2507']
    markers, colors = ['o', 's', '^'], ['#111111', '#666666', '#aaaaaa']
    for ax, (task, label, xlabel) in zip(axes, task_names):
        subset = frame[frame['task'] == task]
        spans = [0.0]
        for y, backbone in enumerate(backbones):
            group = subset[subset['backbone'] == backbone].set_index('seed')
            values = group['gain']
            avg, sd = float(values.mean()), float(values.std(ddof=1))
            spans.extend([float(values.min()), float(values.max()), avg-sd, avg+sd])
            for index, seed in enumerate((17, 42, 2026)):
                ax.scatter(group.loc[seed, 'gain'], y + (index-1)*.12, marker=markers[index], s=40,
                    color=colors[index], edgecolors='black', linewidths=.55, zorder=4)
            ax.errorbar(avg, y+.31, xerr=sd, fmt='D', markersize=4.5, color='black',
                ecolor='black', elinewidth=1, capsize=3, zorder=5)
        width = max(spans)-min(spans)
        ax.set_xlim(min(spans)-.1*width, max(spans)+.1*width)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=.8, zorder=2)
        ax.set_ylim(1.6, -.4)
        ax.set_yticks([0, 1], ['Qwen2.5 1.5B', 'Qwen3 4B'])
        ax.set_xlabel(xlabel)
        ax.text(0, 1.035, label, transform=ax.transAxes, ha='left', va='bottom', fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        precision = 4 if task == 'elm' else 2 if task == 'meltome' else 1
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position, decimals=precision: f'{value:.{decimals}f}'))
        ax.grid(axis='y', visible=False)
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_linewidth(.8)
    handles = [Line2D([], [], marker=marker, linestyle='', color=color, markeredgecolor='black',
        label='Seed ' + str(seed), markersize=6) for marker, color, seed in zip(markers, colors, (17, 42, 2026))]
    handles.append(Line2D([], [], marker='D', color='black', label='Mean ± sample SD', markersize=5))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.54, 1.02))
    fig.text(.5, .015, 'Positive gain favors M3. All three fixed seeds are retained; n = 3 per task and backbone.',
        ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .91), w_pad=2)
    basename = 'fig_01_paired_m3_m0_gains'
    for format in OUTPUT_FORMATS:
        fig.savefig(OUTPUT_DIR / 'figures' / format / (basename + '.' + format), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    shutil.copyfile(OUTPUT_DIR / 'figures/pdf' / (basename + '.pdf'), OUTPUT_DIR / 'report.pdf')
    caption = ('图1 同一随机种子下M3相对M0的主指标增益。热稳定性和功能检索为M3−M0，保留时间为M0−M3，'
        '正值统一表示M3更好；点表示种子17、42、2026，黑色菱形和误差线表示三个配对增益的均值与样本标准差，'
        '未进行显著性检验。六个任务和底座组合均未在三个种子上全部获得正增益，保留时间两底座的三个增益均为负。')
    (OUTPUT_DIR / 'caption.md').write_text(caption + '\n', encoding='utf-8')
    outputs = [{'path': path.relative_to(OUTPUT_DIR).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(OUTPUT_DIR.rglob('*')) if path.is_file()]
    (OUTPUT_DIR / 'manifest.json').write_text(json.dumps({'input_sha256': hashlib.sha256(INPUT_FILES[0].read_bytes()).hexdigest(),
        'n_paired_values': 18, 'n_pairs_per_stratum': 3, 'error_bar': 'sample SD', 'p_values_computed': False,
        'font_english': english, 'font_chinese': chinese, 'files': outputs}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output_dir': str(OUTPUT_DIR), 'plot_png': str(OUTPUT_DIR / 'figures/png' / (basename + '.png')),
        'n_rows': len(frame), 'missing': int(frame.isna().sum().sum()), 'duplicate_keys': 0, 'fonts': [english, chinese]}))

if __name__ == '__main__': main()
