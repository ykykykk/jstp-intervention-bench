"""由封存的原始样例和逐种子统计生成图、表、可编辑文稿和中文 PDF。"""
from __future__ import annotations
import csv
import json
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Image as RLImage, Paragraph, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E = ROOT/"evidence/paper"
FIG = HERE/"figures"
AUTHOR = "岳轲"
TITLE = "同一画面，不同后果"
SUB = "联合状态转移预测中功能性分组的形成与迁移边界"
PAPER = HERE/"JSTP-岳轲.pdf"
INK, GRAY, TEAL, PALE = "#192D30", "#536363", "#00665F", "#E6F3EF"
CHAPTER = ["研究问题", "相关工作与命题", "实验构造", "模型与训练", "任务选择", "物理后果", "讨论与结论", "参考文献"]
NAMES = {
    "jstp":"JSTP：搬运＋成对", "transport_factual":"搬运，仅单分支损失",
    "rgb_paired":"残差 RGB，成对", "rgb_factual":"残差 RGB，单分支",
    "history_policy":"完整历史直接策略", "dino_wm":"DINO-WM 适配",
    "one_bit":"显式规则：一位", "visible_rule":"显式规则：三位"}
CELLS = [(s,t) for s in ("test_id","test_link") for t in ("single","compound")]
HDS = ["熟悉连接·单区","熟悉连接·双区","新连接·单区","新连接·双区"]

def rows(filename):
    with (E/"analysis"/filename).open(encoding="utf8",newline="") as f:
        return list(csv.DictReader(f))

def typeface(n, bold=False):
    return ImageFont.truetype(str(HERE/"fonts"/("JSTPSerif-Bold.ttf" if bold else "JSTPSerif-Regular.ttf")),n)

def figures():
    FIG.mkdir(exist_ok=True)
    with np.load(E/"observed/test_id/example.npz") as observed, np.load(E/"exam/test_id/jstp-1729.npz") as sealed:
        grid=[[observed["peaks"][j],observed["current"],sealed["example"][j,0],observed["actual"][j,0]] for j in range(2)]
    im=Image.new("RGB",(1500,542),"white"); d=ImageDraw.Draw(im)
    for i,label in enumerate(("横向试探峰值","复位后（相同）","固定拉 A：预测","固定拉 A：实际")):
        d.text((228+365*i,30),label,font=typeface(30,True),fill=TEAL if i==2 else INK,anchor="mm")
    for j,row in enumerate(grid):
        y=69+234*j
        d.text((24,y+100),"Ⅰ" if j==0 else "Ⅱ",font=typeface(34,True),fill=GRAY,anchor="mm")
        for i,frame in enumerate(row):
            x=67+365*i
            tile=Image.fromarray(frame).resize((318,212),Image.Resampling.NEAREST)
            im.paste(tile,(x,y))
            d.rectangle((x-1,y-1,x+318,y+212),outline=TEAL if i==2 else "#C5CDCA",width=2)
    im.save(FIG/"paired-example.png",optimize=True)
    data={(r["link"],r["goals"],r["run"]):100*int(r["completed"])/int(r["total"]) for r in rows("per_seed.csv")}
    im=Image.new("RGB",(1500,405),"white");d=ImageDraw.Draw(im)
    xs=[266,610,958,1299]; top,bottom=70,327
    yof=lambda v:bottom-(bottom-top)*v/100
    d.text((52,15),"任务成功率（%）",font=typeface(25,True),fill=INK)
    palette=[TEAL,"#A44926","#3B5390"]
    def dot(x,y,idx):
        r=8
        if idx==0:d.ellipse((x-r,y-r,x+r,y+r),fill=palette[idx])
        elif idx==1:d.rectangle((x-r,y-r,x+r,y+r),fill=palette[idx])
        else:d.polygon([(x,y-r-2),(x-r-1,y+r),(x+r+1,y+r)],fill=palette[idx])
    for idx,seed in enumerate((1729,2718,31415)):
        x=878+190*idx;dot(x,30,idx)
        d.text((x+19,30),"种子 "+str(seed),font=typeface(23),fill=INK,anchor="lm")
    for v in (0,25,50,75,100):
        y=yof(v)
        d.text((90,y),str(v),font=typeface(22),fill=GRAY,anchor="rm")
        if v==50:
            for x in range(110,1450,24):d.line((x,y,x+13,y),fill="#859590",width=2)
        else:d.line((110,y,1455,y),fill="#DCE5E2",width=2)
    for i,(s,t) in enumerate(CELLS):
        d.text((xs[i],365),HDS[i],fill=INK,font=typeface(25,True),anchor="mm")
        for j,seed in enumerate((1729,2718,31415)):
            dot(xs[i]+(j-1)*28,yof(data[s,t,f"jstp-{seed}"]),j)
    im.save(FIG/"seed-transfer.png",optimize=True)

def md_table(title,head,body):
    row=lambda a:"| "+" | ".join(map(str,a))+" |"
    return f"**{title}**\n\n"+"\n".join([row(head),row(["---"]*len(head)),*(row(a) for a in body)])

def content():
    stats={(r["method"],r["link"],r["goals"]):r for r in rows("metrics.csv")}
    desc={"jstp":"平移搬运，实测像素差＋目标排序","transport_factual":"平移搬运，仅观测分支重建",
          "rgb_paired":"局部残差，相同成对目标","rgb_factual":"局部残差，仅观测分支重建",
          "history_policy":"按真实后果监督动作选择","dino_wm":"冻结 DINOv2，回归未来特征",
          "one_bit":"仅读右带是否动，平均相容未来","visible_rule":"三带响应＋已知互补机构"}
    params={k:f'{json.loads((E/"models"/(k+"-1729")/"freeze.json").read_text(encoding="utf8"))["trainable"]:,}' for k in list(NAMES)[:6]}
    a=[[name,params.get(k,"不训练"),desc[k]] for k,name in NAMES.items()]
    b=[[name,*[f'{100*float(stats[k,s,t]["task"]):.1f}' for s,t in CELLS]] for k,name in NAMES.items()]
    frows=[]
    for k,name in NAMES.items():
        cells=[]
        for s in ("test_id","test_link"):
            rr=stats[k,s,"single"]
            cells+=([f'{100*float(rr["physics_pair"]):.1f}',f'{float(rr["pixel_mae"]):.2f}'] if rr["physics_pair"] else ["—","—"])
        frows.append([name,*cells])
    refs=json.loads((HERE/"references.json").read_text(encoding="utf8"))
    reftext="\n\n".join(f'[{i}] {r["short"]}. [{r["title"]}]({r["url"]}). {r["venue"]}，{r["year"]}。' for i,r in enumerate(refs,1))
    bib="\n\n".join("@misc{"+r["key"]+",\n"+",\n".join("  "+k+" = {"+v+"}" for k,v in [
            ("author",r["author"]),("title",r["title"]),("year",r["year"]),("howpublished",r["venue"]),("url",r["url"])])+"\n}" for r in refs)
    (HERE/"references.bib").write_text(bib+"\n",encoding="utf8",newline="\n")
    text=(HERE/"source.md").read_text(encoding="utf8")
    for key,value in (
        ("LITERATURE",md_table("比较口径　谁预测什么，本文做了哪一级比较",["路线","控制接口","本研究的对照"],
             [["动作型 VLA [1]","按观察和目标产生动作","文献；同场测小型完整历史策略"],
              ["视觉子目标 VLA [2,3]","以期望画面指导动作","文献；未直接测通用 VLA"],
              ["动作条件世界模型 [5,6,14]","以候选后果选动作","DINO-WM 适配和 RGB 对照同场实测"],
              ["联合世界—动作模型 [7,8,13]","建模未来状态与行动","文献；不比通用操作能力"]])),
        ("METHODS",md_table("表 1　同场方法及可训练参数",["方法","参数","训练目标或已知先验"],a)),
        ("MAIN",md_table("表 2　实际执行成功率（%）",["方法",*HDS],b)),
        ("PHYS",md_table("表 3　严格物理配对正确率与平均绝对位移误差",["方法","熟悉·配对 %","熟悉·误差 px","新连接·配对 %","新连接·误差 px"],frows)),
        ("REFS",reftext)):
        text=text.replace("{{"+key+"}}",value)
    if "{{" in text:raise ValueError("存在未填的出版字段")
    (HERE/"manuscript.md").write_text(text,encoding="utf8",newline="\n")
    return text

def rich(t):
    t=escape(t)
    t=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",t)
    return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",r'<link href="\2" color="#00665F">\1</link>',t)

def style(name,size,lead,after,scale,**kw):
    base=dict(fontName="Study",fontSize=size*scale,leading=lead*scale,spaceAfter=after*scale,
              textColor=colors.HexColor(INK),wordWrap="CJK",splitLongWords=0)
    base.update(kw)
    return ParagraphStyle(name,**base)

def elements(page,number,scale):
    W=A4[0]-96
    normal=style("body",10.35,16.7,6.5,scale)
    ref=style("ref",9.5,14.5,7,scale,wordWrap=None,leftIndent=22,firstLineIndent=-22)
    caption=style("caption",8.7,13,8,scale,textColor=colors.HexColor(GRAY))
    h3=style("h3",16.4,22,10,scale,fontName="StudyBold",textColor=colors.HexColor(TEAL))
    blocks=re.split(r"\n\s*\n",page.strip())
    out=[]
    for block in blocks:
        block=block.strip()
        if block.startswith("# "):
            ls=block.splitlines()
            out.extend([Paragraph(escape(ls[0][2:]),style("title",32,41,6,scale,fontName="StudyBold")),
                        Paragraph(escape(ls[1][3:]),style("sub",16.2,24,11,scale,textColor=colors.HexColor(TEAL))),
                        Paragraph(rich(ls[2]),style("author",12.5,18,20,scale))])
        elif block.startswith("### "):
            lines=block.split("\n",1)
            out.append(Paragraph(rich(lines[0][4:]),h3))
            if len(lines)>1:
                out.append(Paragraph(rich(lines[1]),ref if number==8 else normal))
        elif block.startswith("!["):
            ma=re.fullmatch(r"!\[(.*?)\]\((.*?)\)",block)
            img=HERE/ma[2]; wi,hi=Image.open(img).size
            imagew=W*.965*scale
            out.extend([Spacer(1,2),RLImage(str(img),width=imagew,height=imagew*hi/wi),Spacer(1,5),
                        Paragraph(rich(ma[1]),caption)])
        elif block.startswith("|"):
            rs=[[v.strip() for v in ln.strip().strip("|").split("|")] for ln in block.splitlines()]
            data=[rs[0],*rs[2:]]
            three=len(data[0])==3
            lit=data[0][1]=="控制接口"
            widths=([W*.32,W*.31,W*.37] if lit else [W*.31,W*.18,W*.51]) if three else [W*.31,*([W*.1725]*4)]
            cells=[]
            for i,row in enumerate(data):
                cells.append([Paragraph(rich(t),style("cell",8.8 if three else 8.25,12,0,scale,
                         fontName="StudyBold" if i==0 or (i==1 and j==0) else "Study",
                         textColor=colors.white if i==0 else colors.HexColor(INK),
                         alignment=TA_LEFT if j==0 or (three and j==2) or lit else TA_CENTER))
                              for j,t in enumerate(row)])
            tb=Table(cells,colWidths=widths,hAlign="LEFT")
            tb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(TEAL)),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#EFF5F3"),colors.white]),
                ("LINEBELOW",(0,-1),(-1,-1),.6,colors.HexColor("#9AB5B0")),
                ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),4),
                ("TOPPADDING",(0,0),(-1,-1),2.8*scale),("BOTTOMPADDING",(0,0),(-1,-1),2.8*scale),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            out.extend([tb,Spacer(1,8*scale)])
        elif block.startswith("> "):
            t=Table([[Paragraph(rich(block[2:]),style("quote",10.1,16,0,scale,fontName="StudyBold"))]],colWidths=[W])
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor(PALE)),
                ("LINEBEFORE",(0,0),(0,0),2.5,colors.HexColor(TEAL)),("LEFTPADDING",(0,0),(-1,-1),10),
                ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
            out.extend([t,Spacer(1,8)])
        else:
            st=ref if number==8 else (style("tablelabel",10,14,3,scale,fontName="StudyBold") if block.startswith("**表 ") else normal)
            out.append(Paragraph(rich(block),st))
    return out

def pdf(text):
    # Document metadata uses the manuscript date, not the current wall clock.
    os.environ.setdefault("SOURCE_DATE_EPOCH","1794960000")
    for name,file in (("Study","Regular"),("StudyBold","Bold")):
        pdfmetrics.registerFont(TTFont(name,str(HERE/"fonts"/f"JSTPSerif-{file}.ttf")))
    pdfmetrics.registerFontFamily("Study",normal="Study",bold="StudyBold")
    cv=canvas.Canvas(str(PAPER),pagesize=A4,invariant=1,pageCompression=1)
    cv.setTitle(TITLE+"："+SUB);cv.setAuthor(AUTHOR);cv.setSubject("JSTP 配对干预实验与跨机构迁移")
    cv.setCreator("JSTP Intervention Bench · paper/build.py")
    w,h=A4; margin=48; top=h-60; bottom=49; usable=w-2*margin
    for i,page in enumerate(text.split("<!-- page -->"),1):
        cv.setFillColor(colors.HexColor(TEAL));cv.setFont("Helvetica-Bold",8.2)
        cv.drawString(margin,h-33,"JSTP   /   INTERVENTION STUDIES")
        cv.setFillColor(colors.HexColor(GRAY));cv.setFont("Study",8.5)
        cv.drawRightString(w-margin,h-32,"联合状态转移预测　·　研究手稿")
        cv.setStrokeColor(colors.HexColor("#B3C7C1"));cv.line(margin,h-42,w-margin,h-42)
        cv.line(margin,37,w-margin,37);cv.setFont("Study",8.2)
        cv.drawString(margin,23,AUTHOR+"　/　"+CHAPTER[i-1])
        cv.setFont("Helvetica",8.2);cv.drawRightString(w-margin,23,f"{i:02}   /   08")
        cv.bookmarkPage(f"p{i}");cv.addOutlineEntry(CHAPTER[i-1],f"p{i}",0,False)
        for s in (1,.98,.96,.94,.92,.90,.88,.86):
            items=elements(page,i,s)
            total=sum(v.wrap(usable,top-bottom)[1]+v.getSpaceAfter()+v.getSpaceBefore() for v in items)
            if total<=top-bottom-2:break
        else:raise ValueError(f"第 {i} 页放不下：{total} pt")
        frame=Frame(margin,bottom,usable,top-bottom,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        frame.addFromList(items,cv)
        if items:raise RuntimeError(f"第 {i} 页排版截断")
        print(f"page {i} | scale {s:.2f} | content {total:.0f} pt")
        cv.showPage()
    cv.save()

if __name__ == "__main__":
    figures();pdf(content());print(PAPER.name)
