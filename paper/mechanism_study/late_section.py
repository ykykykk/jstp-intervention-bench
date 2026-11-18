"""从晚到目标归档绘制首题证据与填充论文。无需 PyTorch。"""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from lategoal.rig import Rig,goal_wins,lane_travel

def material(rel,fig,font):
    summary=json.loads((rel/"analysis/summary.json").read_text(encoding="utf8"))
    replay=json.loads((rel/"frozen-replay.json").read_text(encoding="utf8"));a=summary["arms"];j=a["joint"]
    def pct(k,arm="joint"):return f"{100*a[arm][k]['value']:.2f}"
    def ci(k):
        r=j[k];return f"{100*r['lo']:.2f}–{100*r['hi']:.2f}"
    stats=summary["per_seed"]["joint"];names={"joint":"联合预测＋直接通路","aux":"后果学习、断开支路","direct":"仅直接动作"}
    table=["**表 4　新目标与隐状态移植（%，每组固定三个种子）**","",
           "| 训练条件 | 两目标均成功 | 应换·定向换手 | 等幅零空间 | 不应换·误换 | 物理读数转向 |",
           "| --- | --- | --- | --- | --- | --- |"]
    for arm,label in names.items():
        table.append("| "+" | ".join([label,*[pct(k,arm) for k in ("pair","directed","null_directed","false_flip")],
                                      pct("physical_turn",arm) if arm!="direct" else "—"])+" |")
    decided=summary["gates"];overall="均通过" if decided["task_primary"] and decided["mechanism_primary"] else "未全部通过"
    lseg="、".join(f"{100*stats[str(i)]['directed']:.2f}%" for i in summary["seeds"])
    diff=summary['paired_contrasts']['direct']['pair']
    report=f"重读冻结权重，在空结果目录重新标定物理方向、试探、封卷并执行；比较 {replay['decisions']:,} 项驱动及逐项光学、相机和统计记录。差异字段数为 {len(replay['changed_array_elements'])}，统计汇总{'一致' if replay['identical_summary'] else '不同'}。这是同机冻结复演，不是独立重训。"
    z=np.load(rel/"sealed/joint-6043.npz");o=np.load(rel/"observed/joint-6043.npz");cq=np.load(rel/"sealed/goals.npz")
    goals=cq["goals"];rng_id=int(cq["ids"][0]);r=Rig(rng_id)
    opt=np.where((o["left"][0,:,None]*goals).sum(-1)>0,-1,1);changed=opt[0]!=opt[1]
    ja=int(np.flatnonzero(changed)[0]);jb=int(np.flatnonzero(~changed)[0]);p=int(z["probe"][0])
    im=Image.new("RGB",(1530,680),"white");d=ImageDraw.Draw(im);ink="#193634";green="#14664f";orange="#a54e29";mut="#566962"
    text=lambda x,y,t,size=27,col=ink,b=False,anchor="lm":d.text((x,y),t,font=font(size,b),fill=col,anchor=anchor)
    text(28,35,f"首个布局 {rng_id}   /   试探把手 {'A' if p<0 else 'B'}   /   现实机构不变",32,green,True)
    for x,t in [(28,"操作条件"),(411,"目标卡  1 · 2 · 4"),(688,"内部读出的纵向后果"),(1090,"把手"),(1214,"原机构实测")]:text(x,97,t,25,green,True)
    for row,(label,g,k) in enumerate([("原预测",ja,0),("只换相反目标",ja^1,0),("只换隐式预期",ja,2),("同一移植 · 不应换手题",jb,2)]):
        y=137+row*129
        d.rounded_rectangle((16,y,1515,y+115),radius=10,fill="#F2F6F3" if row%2==0 else "#F9FAF7")
        text(29,y+55,label,28,ink,True)
        direction="    ".join("↓" if s>0 else "↑" for s in goals[g] if s)
        text(430,y+55,direction,34,ink,True)
        v=z["eye"][0,0,k];base=z["eye"][0,0,0];axis=y+57
        d.line((694,axis,1011,axis),fill="#91A39A",width=2)
        for col,vals,dy in (("#B4C9BC",base,-2),(green if k==0 else orange,v,3)):
            pts=[(702+27*i+dy,axis+float(np.clip(t,-1.3,1.3))*38) for i,t in enumerate(vals)]
            d.line(pts,fill=col,width=5)
        u=int(z["u"][0,0,g,k]);good=bool(o["wins"][0,0,g,k]);w=r.world(0);peak=w.probe(p);w.set_clutch(0);frame=w.stroke(u)
        if hashlib.sha256(frame.tobytes()).hexdigest().encode()!=o["frame_sha"][0,0,int(u>0)]:raise ValueError("单题相机与封存实测不同")
        if bool(goal_wins(lane_travel(r.base,frame),goals[g]))!=good:raise ValueError("单题成败不同")
        text(1111,axis,"A" if u<0 else "B",40,green if k==0 else orange,True)
        im.paste(Image.fromarray(frame).resize((130,87),Image.Resampling.NEAREST),(1194,y+14))
        text(1354,axis,"达到" if good else "未达到",25,green if good else orange,True)
    im.save(fig/"late-case.png",optimize=True)
    return dict(
        LATE_ABSTRACT=f"同经历的一对相反新目标均完成的比例为 **{pct('pair')}%**。在应该改换把手的题上，仅移植隐式后果的物理方向就有 **{pct('directed')}%** 定向换手，等幅零空间改动为 **{pct('null_directed')}%**；同一移植在不应换手的题上误换 **{pct('false_flip')}%**。直接动作对照的任务对成功率为 **{pct('pair','direct')}%**。",
        LATE_TASK=f"两项事前主判据{overall}。主要模型对单张新卡完成 **{pct('task')}%**，相反目标要求在同一真实前缀上两次都达成，成功 **{pct('pair')}%**（95% 布局区间 {ci('pair')}%）。对有一项与其余两项冲突的题目，正确率为 {pct('contested')}%。三种条件都能把旧经历用到新卡上；仅直接动作对照的目标对正确率为 {pct('pair','direct')}%。",
        LATE_TABLE="\n".join(table),
        LATE_MECHANISM=f"固定真实世界，只换隐状态的物理分量，有 **{pct('physical_turn')}%** 的预期读数从更像本世界变为更像供体。物理上应改换把手的题占一半：三种子定向换手为 {lseg}，汇总 **{pct('directed')}%**（95% 布局区间 {ci('directed')}%）。等幅零空间对照为 {pct('null_directed')}%，差 **{pct('vs_null')} 个百分点**（{ci('vs_null')}）。当同一移植对应的目标不应换手时，误换为 {pct('false_flip')}%；两个目标条件相差 {pct('vs_stay')} 点（{ci('vs_stay')}）。因此，观测到的不是普遍扰乱运动输出。",
        LATE_CONTROLS=f"整段自然供体预期被移入时，应换手题有 {pct('full_directed')}% 定向换手；它同时带走更多信息，不作为主判据。换成经历不同、实际后果相同的整段预期，动作只有 {pct('equal_future_changed')}% 发生变化。断开后果支路的辅助组也能把物理读数移向供体，但这条被切断的支路不能再改变它的动作。联合组相对直接组的目标对成功率差为 **{100*diff['value']:.2f} 个百分点**（{100*diff['lo']:.2f} 至 {100*diff['hi']:.2f}）。没有任务性能优势。",
        LATE_REPLAY=report)
