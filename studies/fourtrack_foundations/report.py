"""Build the human-readable comparison from the sealed execution, not paper percentages."""
from .common import E,HERE,OFF,P,jread

NAMES={'joint':'完整 JSTP','act':'ACT','diffusion':'Diffusion Policy','vjepa21':'V-JEPA 2.1','smol':'SmolVLA','dino':'DINO-WM'}
ROLE={'joint':'原版从头训练','act':'ALOHA 策略：末端投影和读出','diffusion':'Push-T：四步去噪后的投影和读出','vjepa21':'冻结视频编码器，学习任务读出','smol':'原生流动作专家局部微调；复用旧训练','dino':'原作者预测器适配；复用旧训练'}
OUT=HERE/'outputs'/'fourtrack-20261120'
def write():
    report=jread(E/'summary.json');before=jread(E/'before.json')
    rows={(r['arm'],r['budget']):r for r in report['rows']}
    rank=sorted(P['arms'],key=lambda a:-rows[a,128]['pair']['value'])
    rivals=[a for a in rank if a!='joint']
    name=lambda a:NAMES[a] if a=='joint' else f"[{NAMES[a]}]({P[a]['source']})"
    pct=lambda v:f"{100*v:.2f}%"
    pos=lambda v:f"{100*v:+.2f}"
    ci=lambda d:f"{pos(d['lo'])} 至 {pos(d['hi'])}"
    interval=lambda d:f"{pct(d['value'])} [{pct(d['lo'])}, {pct(d['hi'])}]"
    out=['# 四轨装置上的同数据对照：完整 JSTP 与五种外部方法','',
        '**怎样才算成功？** 装置先做一次公开试探。把相同经历分成两份，分别交给模型一张从未出现在训练里的目标卡，两张卡方向相反。模型分别选把手；模拟器实际拉动，两次都达到各自目标，一对才得分。下面按 128 条示范时的成对成功率排序。','',
        '| 模型 / 接口 | 128 条示范 | 1,024 条示范 | 16,384 条示范 | 四轨训练方式 |',
        '|---|---:|---:|---:|---|']
    for a in rank:
        vals=[pct(rows[a,b]['pair']['value']) for b in P['budgets']]
        out.append('| '+' | '.join([name(a)]+vals+[ROLE[a]])+' |')
    joint=rows['joint',128]['pair'];best=rows[rivals[0],128]['pair']
    rival=rivals[0];diff=next(c for c in report['contrasts'] if c['budget']==128 and c['other']==rival)
    out+=['',f"128 档最强外部对照是 **{NAMES[rival]}**。完整 JSTP 为 **{pct(joint['value'])}**，该对照为 **{pct(best['value'])}**，JSTP 减对照 **{pos(joint['value']-best['value'])} 个百分点**（校正区间 {ci(diff['primary'])}）。",
          '', '## 128 档：有多大差异？','',
          '| 模型 | 成对成功数 / 总对数（三种子） | 成对成功率 [按布局 95% 区间] | 单题成功率 | 相反目标仍选同一把手 |',
          '|---|---:|---:|---:|---:|']
    for a in rank:
        t=rows[a,128];out.append(f"| {NAMES[a]} | {t['won']:,} / {t['pairs']:,} | {interval(t['pair'])} | {pct(t['single']['value'])} | {pct(t['same']['value'])} |")
    out+=['','| JSTP 减外部对照 | 差值（百分点） | 99% 配对区间¹（百分点） | 三个种子的差值（百分点） |','|---|---:|---:|---|']
    for a in rivals:
        c=next(v for v in report['contrasts'] if v['budget']==128 and v['other']==a)
        out.append(f"| {NAMES[a]} | {pos(c['primary']['value'])} | {ci(c['primary'])} | {', '.join(pos(v) for v in c['seed_deltas'])} |")
    out+=['','¹ 同一批 192 个布局按整布局配对重采样 3,000 次，种子固定。五项主要对比做 Bonferroni 校正，每项显示 99% 百分位区间；不单独重采样每张卡，也不把三个种子当作三批布局。区间覆盖零时，这个差值不足以区分两者。','',
          '## 训练条件与边界','',
          '| 方法 | 各份权重的可更新参数 | 下游动作标签 | 局部光学监督 | 本轮 |','|---|---:|---|---|---|']
    for a in P['arms']:
        rec=jread((OFF if a in P['archived'] else E/a)/'models'/f'{a}-128-9301'/'freeze.json')
        n=rec['adapted'] if a in P['archived'] else (rec['registered_parameters'] if a=='joint' else rec['trainable'])
        act='沿用同协议九次旧训练；重新推理、执行' if a in P['archived'] else '重新训练九次，冻结再执行'
        out.append(f"| {NAMES[a]} | {n:,} | 相同嵌套标签池 | {'未用于训练' if a=='smol' else '同批每条 4 / 12 个读数'} | {act} |")
    out+=['','- 所有方法对应同一批 16,384 条物理经历。三档变化的是有正确把手标注的经历数，不是观察总数。各用 768 次更新，每次 128 个动作标注；训练和考试的目标卡互不相同。使用原论文的 192 个留出布局，并非另出一批全新盲题。',
    '- 完整 JSTP 从头训练全部参数。ACT 和 Diffusion Policy 使用官方预训练的动作路径，但骨干冻结；去噪步骤、动作接口与原场景并非完全相同。V-JEPA 2.1 是编码器读出，不是原生规划器。DINO-WM 是作者预测器的四轨适配，不是原任务的完整发布系统。SmolVLA 使用原生流动作目标，未加局部光学损失。',
    '- JSTP 和三个新外部适配的任务接口都用四个坐标与目标做内积。因此目标取反时换把手几乎是结构保证；若只盲猜第一张卡、第二张固定反选，期望成对成功率是 50%，而两张各自独立盲猜是 25%。换手本身不是学会物理的证据。这个成绩不能单独证明涌现分割、对任意语言指令的理解、真机优势，或是哪一条内部路径导致差异。',
    f"- 每个模型每档 3 个训练种子，每份权重 12,288 对 / 24,576 次实际模拟决策。54 份冻结策略共执行 {report['actual_commands']:,} 条命令。相同起点和命令的确定性相机图像可复用，逐项与原考场指纹比对。",
    f"- SmolVLA 和 DINO-WM 均从冻结权重重新推理，没有把旧成绩录入新表。18 份权重共 442,368 条命令，本轮与原封卷动作不同的有 {sum(before['recomputed_vs_archived_sign_differences'].values()):,} 条。",
    '', '[核对逐种子数据](../../evidence/per-seed.json) · [统计与区间](../../evidence/summary.json) · [事前封存的动作记录](../../evidence/before.json) · [实际执行记录](../../evidence/after.json) · [架构与完整训练口径](../../README.md)']
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'四轨模型对比.md').write_text('\n'.join(out)+'\n',encoding='utf8')
if __name__=='__main__':write()
