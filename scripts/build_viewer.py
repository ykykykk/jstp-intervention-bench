"""把预定第一个布局的真实提交与执行结果做成自包含网页，不重新推理、不修改记录。"""
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/"evidence/paper"

def png(rgb,mask=None):
    im=Image.fromarray(rgb).convert("RGB")
    if mask is not None:
        draw=ImageDraw.Draw(im)
        for lane in range(3):
            x=32*lane
            if mask[0,x]:
                draw.rectangle((x,0,x+31,len(mask)-1),outline="#81efdb")
    f=io.BytesIO();im.save(f,format="PNG",optimize=True)
    return "data:image/png;base64,"+base64.b64encode(f.getvalue()).decode("ascii")

def build():
    out={"scenes":[],"curves":[]}
    for split in ("test_id","test_link"):
        with np.load(E/"exam"/split/"jstp-1729.npz") as p,np.load(E/"observed"/split/"example.npz") as x,np.load(E/"observed"/split/"outcomes.npz") as wins:
            out["scenes"].append(dict(current=png(x["current"]),peaks=[png(v) for v in x["peaks"]],
                goals=[[png(np.concatenate((g,x["current"][43:])),x["masks"][t]) for g in row] for t,row in enumerate(x["goals"])],
                pred=[[png(v) for v in row] for row in p["example"]],actual=[[png(v) for v in row] for row in x["actual"]],
                costs=p["cost"][0].tolist(),choice=p["choice"][0].tolist(),wins=wins["jstp-1729"][0].tolist()))
    for seed in (1729,2718,31415):
        records=[json.loads(l) for l in (E/"models"/f"jstp-{seed}"/"curve.jsonl").read_text(encoding="utf8").splitlines()]
        out["curves"].append([[r["step"],r["val_task"]] for r in records if r["phase"]==2])
    (ROOT/"docs/paired-demo.html").write_text(PAGE.replace("__DATA__",json.dumps(out,separators=(",",":"))),
                                              encoding="utf8",newline="\n")

PAGE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>同一画面，不同后果 · JSTP 公开证据</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f1;color:#162d2c;font:16px/1.65 system-ui,sans-serif}
main{max-width:1010px;padding:36px 22px 70px;margin:auto}header{border-top:4px solid #09665c}h1{font:700 clamp(32px,5vw,54px)/1.25 serif;margin:10px 0}h2{margin:18px 0 9px;font-size:22px}
small,.note{color:#50625f}.tag{margin:16px 0 5px;font:700 12px system-ui;letter-spacing:2px;color:#12685c}
.panel{margin:20px 0;padding:18px 22px 22px;background:white;border:1px solid #d1ddda;border-radius:8px}
.controls,.grid{display:grid;gap:10px}.controls{grid-template-columns:repeat(auto-fit,minmax(139px,1fr));margin:14px 0}.grid{grid-template-columns:repeat(3,1fr)}
select{width:100%;padding:7px;color:#153a34;background:#f2f7f5;border:1px solid #a9bdb6;border-radius:4px}
label,figcaption{font-size:13px;color:#415954}figure{margin:0}figure img{width:100%;image-rendering:pixelated;border-radius:3px;background:#111722;display:block}
.verdict{font-size:20px;font-weight:700;color:#085f52;margin-bottom:3px}.bad{color:#974425}svg{width:100%;max-height:235px}.legend{font-size:14px}
@media(max-width:620px){.grid{grid-template-columns:repeat(2,1fr)}main{padding:20px 12px}.panel{padding:12px}}</style>
<main><header><p class="tag">JSTP  /  INTERVENTION STUDIES</p><h1>同一画面，不同后果</h1>
<p>固定一次经历，只换目标；再固定目标，只换模型看到的试探。看预测、选择和真实结果，三者是否一致。</p></header>
<section class="panel"><h2>冻结测试：一题，两种接线</h2><small>两套试卷均展示协议中编号最小的布局；固定种子 1729。这里读取的是先封存、再执行的存档，不是实时训练或新推理。</small>
<div class="controls"><label>布局<select id="s"><option>熟悉连接</option><option>新连接</option></select></label>
<label>真实挂接<select id="j"><option>世界 Ⅰ</option><option>世界 Ⅱ</option></select></label>
<label>给模型的经历<select id="m"><option>真实试探</option><option>交换视觉证据</option></select></label>
<label>任务范围<select id="t"><option>单区</option><option>双区（训练未见）</option></select></label>
<label>任务目标<select id="g"><option>目标一</option><option>目标二</option></select></label></div>
<div class="grid"><figure><img id="peak" alt="输入模型的试探峰值"><figcaption>模型看到的试探峰值</figcaption></figure>
<figure><img id="now" alt="复位相机"><figcaption>复位后的真实画面</figcaption></figure>
<figure><img id="goal" alt="目标"><figcaption>目标：青框内须逐像素匹配</figcaption></figure>
<figure><img id="a" alt="预测 A"><figcaption id="ca"></figcaption></figure>
<figure><img id="b" alt="预测 B"><figcaption id="cb"></figcaption></figure>
<figure><img id="actual" alt="选定把手的真实执行"><figcaption>选定把手的实际后果</figcaption></figure></div>
<div id="verdict" class="verdict" aria-live="polite"></div><small>换目标不会改变固定动作的预测图，因为目标只进入比较器。交换证据时现实接线保持不变。</small></section>
<section class="panel"><h2>训练过程：三次初始化</h2><small>独立同分布验证集的单区任务成功率；横轴是第二阶段更新数，不含考试数据。</small>
<svg id="chart" viewBox="0 0 800 205" role="img" aria-label="三个种子的验证集学习曲线"></svg>
<div class="legend"><span style="color:#00645d">● 1729</span>　<span style="color:#a74823">■ 2718</span>　<span style="color:#3b5390">▲ 31415</span></div></section>
<small>岳轲 · 数据与论文表格共用同一归档。单个例子说明接口，不替代整份盲测结果。</small></main>
<script>
const data=__DATA__, el=id=>document.getElementById(id),idx=id=>el(id).selectedIndex;
function render(){let s=data.scenes[idx('s')],j=idx('j'),mode=idx('m')?2:0,h=mode?1-j:j,t=idx('t'),g=idx('g'),k=s.choice[t][mode][j][g],v=s.costs[t][mode][j][g],w=s.wins[t][mode][j][g];
for(const [id,src] of Object.entries({peak:s.peaks[h],now:s.current,goal:s.goals[t][g],a:s.pred[h][0],b:s.pred[h][1],actual:s.actual[j][k]})){el(id).src=src;}
el('ca').textContent='如果拉 A · 预测代价 '+v[0].toFixed(4);el('cb').textContent='如果拉 B · 预测代价 '+v[1].toFixed(4);
el('verdict').textContent='执行 '+(k?'B':'A')+' → '+(w?'达成目标':'未达目标');el('verdict').className='verdict'+(w?'':' bad');}
for(const x of document.querySelectorAll('select'))x.onchange=render;render();
let svg=el('chart'),ink=['#00645d','#a74823','#3b5390'],X=x=>56+(x-150)/1350*708,Y=v=>166-v*138;
function make(tag,attrs,body){let e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(body)e.textContent=body;svg.appendChild(e);}
for(let y of [0,.5,1]){make('line',{x1:52,x2:768,y1:Y(y),y2:Y(y),stroke:'#c8d7d3'});make('text',{x:6,y:Y(y)+4,fill:'#435956','font-size':13},(100*y)+'%');}
for(let x of [150,450,750,1050,1500])make('text',{x:X(x),y:192,'text-anchor':'middle','font-size':13,fill:'#435956'},String(x));
data.curves.forEach((series,i)=>{make('polyline',{points:series.map(p=>X(p[0])+','+Y(p[1])).join(' '),fill:'none',stroke:ink[i],'stroke-width':2.4});for(const p of series){let x=X(p[0]),y=Y(p[1]);if(i===0)make('circle',{cx:x,cy:y,r:3.6,fill:ink[i]});else if(i===1)make('rect',{x:x-3,y:y-3,width:6,height:6,fill:ink[i]});else make('polygon',{points:x+','+(y-4)+' '+(x-4)+','+(y+3)+' '+(x+4)+','+(y+3),fill:ink[i]});}});
</script></html>
'''
if __name__=="__main__":
    build()
