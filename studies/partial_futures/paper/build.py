"""Typeset an editable Chinese paper from sealed results. No model inference during typesetting."""
import csv,json,re,os
from pathlib import Path
from xml.sax.saxutils import escape,quoteattr
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle,Spacer,Frame,Image as Figure
from reportlab.pdfgen.canvas import Canvas

D=Path(__file__).resolve().parent;E=D.parent/"evidence"
LABEL={"partial":"JSTP-P","detached":"断回输","direct":"仅动作","world":"一步世界模型","token":"动作词元 Transformer","flow":"条件流动作"}
METHOD={"partial":"同一残差单元续写局部后果与自身动作，有历史直通",
        "detached":"与 JSTP-P 同骨架，光学损失保留，动作不接后果",
        "direct":"同骨架仅动作监督，动作不接后果",
        "world":"预测两候选的光学场，比较学习到的目标评分",
        "token":"传感历史、四个指令词元与动作查询经自注意力输出 A/B",
        "flow":"目标条件向量场，从噪声到有符号驱动；推理用八步积分"}
def get(p):return json.loads(p.read_text(encoding="utf8"))
def f(v):return f"{100*v:.2f}"
def ci(m):return f(m["value"])+" ["+f(m["lo"])+", "+f(m["hi"])+"]"
def tab(header,rows):
    return "\n".join(["| "+" | ".join(header)+" |","|"+"---|"*len(header)]+["| "+" | ".join(map(str,r))+" |" for r in rows])
def manuscript():
    s=get(E/"analysis/summary.json");a=s["fresh"]["arms"];b=s["v3"]["arms"]
    seeds=get(E/"analysis/per_seed.json");refs=get(D/"references.json")
    repl={}
    repl["models"]=tab(("方法","参数","学习或执行的区别"),[(LABEL[k],f"{get(E/'models'/f'{k}-6043'/'freeze.json')['parameters']:,}",METHOD[k]) for k in LABEL])
    repl["results"]=tab(("方法","旧面板：成对 %","新面板：成对 % [95% CI]","四轨 / 抹去 %"),
        [(LABEL[k],f(b[k]["pair"]["value"]),ci(a[k]["pair"]),f(a[k]["group"]["value"])+" / "+f(a[k]["blind"]["value"])) for k in LABEL])
    repl["mechanism"]=tab(("条件","旧面板 %","新面板 % [95% CI]"),[(label,f(b["partial"][k]["value"]),ci(a["partial"][k])) for k,label in (
        ("directed","改物理方向，应换时定向换"),("null","等幅零空间，应换时定向换"),("false","同一物理修改，不应换时错换"),
        ("over_null","定向换手减零空间（百分点）"),("over_false","定向换手减不应换时错换（百分点）"))])
    repl["seeds"]="、".join(f(x["metrics"]["directed"])+"%" for x in seeds if x["panel"]=="fresh" and x["arm"]=="partial")
    rows=[];scores={}
    labels={"smol":"官方 SmolVLA 局部微调","dino":"DINOv2 + DINO-WM 适配","matched-partial":"JSTP-P（单轮桥接）","matched-direct":"仅动作（单轮桥接）","matched-world":"一步世界模型（单轮桥接）"}
    for k,label in labels.items():
        p=E/"transfer"/k;fr=get(p/"freeze.json");v=get(p/"v3-result.json");u=get(p/"fresh-result.json")
        size=fr.get("adapted",fr.get("params"));scores[label]=u["pair"]["value"];rows.append((label,f"{size:,}",f(v["pair"]["value"]),ci(u["pair"])))
    repl["transfer"]=tab(("同样的 16,384 条记录","更新参数¹","旧面板成对 %","新面板成对 % [95% CI]"),rows)
    cd=get(E/"transfer/paired-contrasts.json")["contrasts"]["fresh"]
    repl["bridge"]="配对比较同一批新布局：在单轮预算下，JSTP-P 减仅动作是 "+ci(cd["matched-direct"])+" 个百分点，减一步世界模型是 "+ci(cd["matched-world"])+" 个百分点。此预算下成对任务成绩最高的是："+max(scores,key=scores.get)+"。"
    repl["references"]="### 参考文献与原始实现\n\n"+"\n\n".join(f"[{i}] {author}（{year}）。[{title}]({url})。" for i,(author,title,year,url) in enumerate(refs,1))
    src=(D/"source.md").read_text(encoding="utf8")
    for k,v in repl.items():src=src.replace("[["+k+"]]",v)
    if "[[" in src:raise ValueError("论文有未填的数据字段")
    (D/"manuscript.md").write_text(src,encoding="utf8",newline="\n");return src
INK="#193039";ACC="#146859";MUT="#566b6b";W=A4[0]-96
def inline(s):
    a=escape(s);a=re.sub(r"\*\*(.*?)\*\*",r"<b>\1</b>",a)
    a=re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)",lambda m:'<a href='+quoteattr(m[2])+' color="'+ACC+'">'+m[1]+'</a>',a)
    return a
def ps(font,size,leading,after=0,color=INK):
    return ParagraphStyle("s",fontName=font,fontSize=size,leading=leading,textColor=colors.HexColor(color),wordWrap="CJK",
                          splitLongWords=1,spaceAfter=after,allowWidows=0,allowOrphans=0)
def objects(page,fac):
    flow=[]
    for group in re.split(r"\n\s*\n",page.strip()):
        if not group:continue
        if group.startswith("!["):
            m=re.fullmatch(r"!\[(.*?)\]\((.*?)\)",group);flow.extend([Figure(str(D/m[2]),width=W,height=W/4),Spacer(1,4),Paragraph(inline(m[1]),ps("JR",8.4*fac,12*fac,6,MUT))]);continue
        if group.startswith("|"):
            lines=[[s.strip() for s in ln.strip("|").split("|")] for ln in group.splitlines() if not re.fullmatch(r"[|\-: ]+",ln)]
            cols=len(lines[0]); parts=([.24,.14,.62] if "学习或执行" in lines[0][-1] else
                     [.48,.19,.33] if cols==3 else [.33,.19,.18,.30] if "更新参数" in group else [.25,.18,.34,.23])
            rr=[[Paragraph(inline(s),ps("JB" if i==0 else "JR",8.5*fac,12*fac,color="#FFFFFF" if i==0 else INK)) for s in r] for i,r in enumerate(lines)]
            t=Table(rr,colWidths=[W*f for f in parts],hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(ACC)),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#EFF5F1"),colors.white]),
              ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),5),
              ("TOPPADDING",(0,0),(-1,-1),4*fac),("BOTTOMPADDING",(0,0),(-1,-1),4*fac),("LINEBELOW",(0,-1),(-1,-1),.7,colors.HexColor("#99B9AC"))]))
            flow.extend([t,Spacer(1,11*fac)]);continue
        if group.startswith("# "):text=group[2:];font,size,lead,space,col="JB",24,34,12,INK
        elif group.startswith("## "):text=group[3:];font,size,lead,space,col="JB",17,25,11,ACC
        elif group.startswith("### "):text=group[4:];font,size,lead,space,col="JB",12,18,6,ACC
        elif group.startswith("[" ) and re.match(r"\[\d+\]",group):text=group;font,size,lead,space,col="JR",8.3,12.5,4,MUT
        else:text=group;font,size,lead,space,col="JR",11.5,19.2,10,INK
        flow.append(Paragraph(inline(text.replace("\n"," ")),ps(font,size*fac,lead*fac,space*fac,col)))
    return flow
def pdf(raw):
    os.environ.setdefault("SOURCE_DATE_EPOCH","1795046400")
    for name,style in [("JR","Regular"),("JB","Bold")]:
        pdfmetrics.registerFont(TTFont(name,str(D/"fonts"/f"JSTPSerif-{style}.ttf")))
    pdfmetrics.registerFontFamily("JR",normal="JR",bold="JB")
    out=D/"JSTP-局部未来与同数据比较-岳轲.pdf";c=Canvas(str(out),pagesize=A4,pageCompression=1,invariant=1)
    c.setAuthor("岳轲");c.setTitle("不必预演整幅未来：局部预测、直接动作与世界模型的同数据比较")
    c.setSubject("同源训练、冻结闭环、功能性分组与后果干预");pages=raw.split("<!-- page -->")
    names=["摘要","问题与装置","同数据架构","任务与分组","干预结果","官方权重与预算匹配","讨论与文献"]
    for i,page in enumerate(pages):
        w,h=A4;c.bookmarkPage(str(i));c.addOutlineEntry(names[i],str(i));c.setFont("Helvetica-Bold",8);c.setFillColor(colors.HexColor(ACC));c.drawString(48,h-32,"JSTP   /   CONTROL STUDIES     04")
        c.setFont("JR",8.5);c.drawRightString(w-48,h-32,"同数据比较   /   "+names[i]);c.setStrokeColor(colors.HexColor("#B3C8BF"));c.line(48,h-42,w-48,h-42);c.line(48,37,w-48,37)
        c.setFillColor(colors.HexColor(MUT));c.drawString(48,23,"岳轲   /   研究手稿");c.setFont("Helvetica",8);c.drawRightString(w-48,23,f"{i+1:02d}  /  {len(pages):02d}")
        for fac in [1,.98,.95,.92,.89,.86,.83,.80]:
            oo=objects(page,fac);height=h-115
            if sum(x.wrap(W,height)[1]+x.getSpaceAfter()+x.getSpaceBefore() for x in oo)<height-3:break
        else:raise ValueError("页面超限："+names[i])
        frame=Frame(48,49,W,height,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0);frame.addFromList(oo,c)
        if oo:raise ValueError("分页遗漏："+names[i])
        c.showPage()
    c.save()
if __name__=="__main__":pdf(manuscript())

