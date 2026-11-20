"""Generate the paper's Markdown tables and PDF from archived observations, without training or scoring."""
import json
from pathlib import Path

D=Path(__file__).resolve().parent
E=D.parent/'evidence'
EXTERNAL=D.parents[1]/'fourtrack_foundations'/'evidence'
PDF='联合状态转移预测导论-岳轲.pdf'
K=(128,1024,16384)
SEEDS=(9301,9419,9547)
ARMS=('joint','world','aux','masked','dino','direct','flow','smol')
NAMES={'joint':'JSTP 共享转移','world':'一步世界模型','aux':'同结构，断开后果回输','masked':'掩码轨迹 Transformer',
       'dino':'DINO-WM 任务适配','direct':'仅直接动作','flow':'条件流动作','smol':'官方 SmolVLA 局部微调',
       'world-isolated':'世界模型，隔离动作梯度','act':'ACT 适配','diffusion':'Diffusion Policy 适配',
       'vjepa21':'V-JEPA 2.1 读出'}

# Each pair expresses the same equation in portable Markdown math and PDF typesetting.
EQUATIONS={
    'eq_partial':(r'y^Q_{t+1}=\pi_Q(S_{t+1}),\qquad \hat y^Q_{t+1}=F_\theta(H_t,c,Q).',
        'y<super>Q</super><sub>t+1</sub> = π<sub>Q</sub>(S<sub>t+1</sub>),　　　 ŷ<super>Q</super><sub>t+1</sub> = F<sub>θ</sub>(H<sub>t</sub>, c, Q)'),
    'eq_transition':(r'h=E(x),\qquad T(h,e)=h+f(h+e),\qquad z_a=T(h,e_a).',
        'h = E(x),　　　 T(h, e) = h + f(h + e),　　　 z<sub>a</sub> = T(h, e<sub>a</sub>)'),
    'eq_motor':(r's(h,g)=g^\top \big[ W T(h+Rz_A,e_m)+b \big].',
        's(h, g) = g<super>T</super> [W T(h + Rz<sub>A</sub>, e<sub>m</sub>) + b]'),
    'eq_loss':(r'\mathcal L=\mathbb E_{\mathrm{label}}\!\left[\log(1+e^{-ys})\right]+3\,\mathbb E_{\mathrm{phys}}\!\left[\tfrac14\sum_{j\in M}(\hat d_j-d_j)^2\right].',
        'L = E<sub>label</sub> [log(1 + exp(−ys))] + 3 E<sub>phys</sub> [(1/4) Σ<sub>j ∈ M</sub> (d<super>pred</super><sub>j</sub> − d<sub>j</sub>)<super>2</super>]')
}

def j(p):
    return json.loads(p.read_text(encoding='utf8'))

def pc(x):
    return f'{100*x:.2f}'

def table(head, rows):
    line=lambda r: '| '+' | '.join(r)+' |'
    return '\n'.join([line(head),line(['---']+['---:']*(len(head)-1)),*[line(r) for r in rows]])

def external_fields():
    """Same layouts, a separate family of five contrasts. Never pool it with the original study."""
    report=j(EXTERNAL/'summary.json')
    rates={(r['arm'],r['budget']):r['pair']['value'] for r in report['rows']}
    comparisons={(r['other'],r['budget']):r for r in report['contrasts']}
    order=sorted((a for a,k in rates if k==128),key=lambda a:-rates[a,128])
    if set(order)!={'joint','act','diffusion','vjepa21','dino','smol'}:
        raise ValueError('Unexpected four-track methods')
    best={k:max(rates[a,k] for a in order) for k in K}
    signed=lambda x:f'{100*x:+.2f}'.replace('-','−')
    limits=lambda low,high:f'[{signed(low)}, {signed(high)}]'
    rows=[]
    for a in order:
        cells=[NAMES[a]]+[f'**{pc(rates[a,k])}**' if rates[a,k]==best[k] else pc(rates[a,k]) for k in K]
        if a=='joint':
            cells+=['—']*3
        else:
            c=comparisons[a,128];v=c['primary']
            cells+=[signed(v['value']),limits(v['lo'],v['hi']),f"{sum(x>0 for x in c['seed_deltas'])}/3"]
        rows.append(cells)
    low=comparisons['act',128]['primary'];mid=comparisons['act',1024]['delta95']
    fields={f'ext_{a}_{k}':pc(rates[a,k]) for a in order for k in K}
    fields.update(external_table=table(['方法','128 条','1,024 条','16,384 条','Δ128','99% 区间','胜/3'],rows),
                  ext_act_lead=pc(-low['value']),ext_act_limits=f"[{pc(-low['hi'])}, {pc(-low['lo'])}]",
                  ext_mid_delta=signed(mid['value']),ext_mid_limits=limits(mid['lo'],mid['hi']),
                  ext_diffusion_gap=pc(comparisons['diffusion',128]['primary']['value']),
                  ext_strokes=f"{report['actual_commands']:,}")
    return fields

def study_fields():
    primary=j(E/'analysis/summary.json')
    official=j(E/'official/analysis/summary.json')
    second=j(E/'challenge/results.json')
    p={(r['arm'],r['budget']):r['metrics'] for r in primary['rows']}
    o={(r['arm'],r['budget']):r['metrics'] for r in official['rows']}
    mainseeds={(r['arm'],r['budget'],r['seed']):r['metrics'] for r in j(E/'analysis/per-seed.json')}
    offseeds={(r['arm'],r['budget'],r['seed']):r['metrics'] for r in j(E/'official/analysis/per-seed.json')}
    # JSTP's original execution is the sole reference. The official tables only add metrics or new adapters.
    allscores={**o,**p}
    sdata={**offseeds,**mainseeds}
    dif={r['other']:r['family'] for r in primary['paired']+official['paired'] if r['budget']==128}
    s={(r['arm'],r['budget']):r for r in second['rows']}
    supcmp={(r['a'],r['b'],r['budget']):r['delta'] for r in second['contrasts']}
    def get(arm,key,k=128):
        return allscores[arm,k][key]['value']
    def pp(arm,key,k=128):
        return pc(get(arm,key,k))
    def intr(m):
        return f"[{pc(m['lo'])}, {pc(m['hi'])}]"
    best={k:max(get(a,'pair',k) for a in ARMS) for k in K}
    ranking=[];detail=[]
    for a in ARMS:
        row=[NAMES[a]]
        for k in K:
            v=get(a,'pair',k)
            row.append(f'**{pc(v)}**' if v==best[k] else pc(v))
        if a=='joint':
            row.extend(['—']*3)
        else:
            d=dif[a]
            n=sum(sdata['joint',128,seed]['pair']>sdata[a,128,seed]['pair'] for seed in SEEDS)
            row.extend(['+'+pc(d['value']),intr(d),f'{n}/3'])
        ranking.append(row)
        same=o[a,128]['same_action']['value'] if (a,128) in o else 2*(get(a,'single')-get(a,'pair'))
        detail.append([NAMES[a],*[pc(sdata[a,128,seed]['pair']) for seed in SEEDS],pp(a,'single'),pc(same),
                       *([pp(a,'group'),pp(a,'blind_group')] if (a,128) in p else ['未测','未测'])])
    base=[a for a in ARMS if (a,128) in p]
    groups=[[NAMES[a]]+[pp(a,'group',k) for k in K] for a in base]
    challenge=[[NAMES[a]]+[pc(s[a,k]['pair']['value']) for k in K] for a in [*base,'world-isolated']]
    m=p['joint',128]; sd=supcmp['world-isolated','world',128]
    jgroup=[sdata['joint',128,seed]['group'] for seed in SEEDS]
    references='\n\n'.join(f"[{i}] {a.rstrip('.')}. [{title}]({url}). {venue}." for i,(a,title,venue,url) in enumerate(j(D/'references.json'),1))
    results={
        'abstract_findings':f"在 128 条标注下，JSTP 的成对成功率为 {pp('joint','pair')}%，一步世界模型为 {pp('world','pair')}%，仅动作策略为 {pp('direct','pair')}%。同结构模型只断开后果至动作的通路后为 {pp('aux','pair')}%（差 {pc(dif['aux']['value'])} 个百分点，99% 布局配对区间 {intr(dif['aux'])}）；该对照在两个较大预算下反而略高。",
        'main_table':table(['方法','128 条','1,024 条','16,384 条','Δ128','校正区间','胜/3'],ranking),
        'main_findings':f"128 条标注时，JSTP 相对仅动作策略、一步世界模型和同结构断回输模型，成对成功率分别高 {pc(dif['direct']['value'])}、{pc(dif['world']['value'])} 和 {pc(dif['aux']['value'])} 个百分点。它在表 1 的原八组中最高，但在表 2 的扩展比较中低于 ACT。SmolVLA 和 DINO-WM 的原始适配分别为 {pp('smol','pair')}% 和 {pp('dino','pair')}%。增至 1,024 或 16,384 条后，同结构断回输在这两档均略高于 JSTP。",
        'detail_table':table(['方法','9301','9419','9547','单题','同手','共动','遮试探'],detail),
        'error_analysis':f"种子 9301 的一步世界模型为 {pc(sdata['world',128,9301]['pair'])}%，超过同种子的 JSTP（{pc(sdata['joint',128,9301]['pair'])}%）；对两种官方适配则三个种子均领先。SmolVLA 单题成功率为 {pp('smol','single')}%，但在 {pc(o['smol',128]['same_action']['value'])}% 的相反目标对中仍选同一把手，这些题对必然一对一错。DINO-WM 的同手率仅 {pc(o['dino',128]['same_action']['value'])}%，而成对成功率为 {pp('dino','pair')}%。其候选效用与目标作内积，目标反号通常足以触发换手；换手本身不能证明模型预期了正确的物理方向。对 JSTP 的加性目标读出也适用同一限制。",
        'group_findings':f"128 条标注时，共动关系最容易从断回输模型读出（{pp('aux','group')}%），其次为掩码轨迹模型（{pp('masked','group')}%），JSTP 为 {pp('joint','group')}%。JSTP 的三个分组成绩从 {pc(min(jgroup))}% 到 {pc(max(jgroup))}%；遮掉试探引起的亮度变化和横向位移后，平均读出降至 {pp('joint','blind_group')}%。这些差异说明可线性读出的关系对历史敏感，同时不能替代成对任务成绩。三档完整结果见表 A1；未对五种外部适配进行此探针。",
        'intervention':f"128 条时，在供体与原情境的最优动作相反的题中，原先正确且向供体动作切换的比例为 {pc(m['directed']['value'])}%，等范数对照为 {pc(m['null']['value'])}%，净差 {pc(m['over_null']['value'])} 个百分点（95% 布局配对区间 {intr(m['over_null'])}）。当最优动作本不应改变时，物理移植造成的原本正确后错换比例为 {pc(m['false']['value'])}%。原预期更接近真实读数且干预后更接近供体读数的比例为 {pc(m['donor_match']['value'])}%。1,024 和 16,384 条时，物理移植相对等范数对照的定向换手净差分别为 {pp('joint','over_null',1024)} 和 {pp('joint','over_null',16384)} 个百分点。",
        'challenge_table':table(['方法','128 条','1,024 条','16,384 条'],challenge),
        'challenge_result':f"128 条时，隔离梯度的一步世界模型为 {pc(s['world-isolated',128]['pair']['value'])}%，原世界模型为 {pc(s['world',128]['pair']['value'])}%，相差 {pc(sd['value'])} 个百分点（95% 布局配对区间 {intr(sd)}）。全量标注时，两者的光学均方根误差分别为 {s['world-isolated',16384]['optical_rmse']['value']:.3f} 和 {s['world',16384]['optical_rmse']['value']:.3f}（以四像素行程归一化）。单独阻断梯度未恢复任务成绩；该对照不足以确定原性能下降的唯一原因。",
        'group_table':table(['方法','128 条','1,024 条','16,384 条'],groups),'references':references
    }
    results.update(external_fields())
    results.update({k:'$$\n'+latex+r'\tag{'+str(n)+'}\n$$' for n,(k,(latex,_)) in enumerate(EQUATIONS.items(),1)})
    return results

def fill_template(source,results):
    text=source.read_text(encoding='utf8')
    for key,val in results.items():
        text=text.replace('[['+key+']]',val)
    if '[[' in text:
        raise ValueError('unfilled manuscript field')
    return text

def compile_text():
    text=fill_template(D/'source.md',study_fields())
    (D/'manuscript.md').write_text(text,encoding='utf8',newline='\n')
    return text

if __name__=='__main__':
    from layout import typeset
    typeset(D,PDF,compile_text(),EQUATIONS)
