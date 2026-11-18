"""研究稿 A4 排版；每章有显式分页，溢出直接报错。"""
from pathlib import Path
import os,re
from xml.sax.saxutils import escape
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT,TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Frame,Paragraph,Spacer,Table,TableStyle,Image as Figure

D=Path(__file__).resolve().parent
INK="#192C31";ACC="#116755";MUT="#536765";W=A4[0]-96
CHAPTER=["研究问题","理论与比较","实验装置","控制与判据","熟悉与新接线","隐状态移植","符号与回输","首轮解释边界","晚到目标","干预与判据","同经历换目标","机制与成绩","复现与文献"]
def rich(x):
    x=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",escape(x))
    return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",r'<link href="\2" color="#116755">\1</link>',x)
def style(size,leading,after,scale,bold=False,**kw):
    base=dict(fontName="StudyBold" if bold else "Study",fontSize=size*scale,leading=leading*scale,
              spaceAfter=after*scale,textColor=colors.HexColor(INK),wordWrap="CJK")
    base.update(kw);return ParagraphStyle("item",**base)
def flow(page,scale):
    out=[]
    for group in re.split(r"\n\s*\n",page.strip()):
        group=group.strip()
        if group.startswith("# "):
            a,b,c=group.splitlines()
            out.extend([Paragraph(rich(a[2:]),style(32,41,6,scale,True)),
                        Paragraph(rich(b[3:]),style(16.8,24,10,scale,textColor=colors.HexColor(ACC))),
                        Paragraph(rich(c),style(12.5,20,18,scale))])
        elif group.startswith("### "):
            a,*b=group.split("\n",1)
            out.append(Paragraph(rich(a[4:]),style(16,22,10,scale,True,textColor=colors.HexColor(ACC))))
            if b:out.append(Paragraph(rich(b[0]),style(10.5,17,7,scale)))
        elif group.startswith("|"):
            lines=[[t.strip() for t in l.strip("|").split("|")] for l in group.splitlines()]
            rr=[lines[0],*lines[2:]];n=len(rr[0]); widths=[W*.29]+[W*.71/(n-1)]*(n-1)
            cells=[[Paragraph(rich(t),style(8.65,12,0,scale,i==0 or (i==1 and j==0),alignment=TA_LEFT if j==0 else TA_CENTER,
                         textColor=colors.white if i==0 else colors.HexColor(INK))) for j,t in enumerate(row)] for i,row in enumerate(rr)]
            tab=Table(cells,colWidths=widths,hAlign="LEFT")
            tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(ACC)),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#EBF3EF"),colors.white]),
                ("LINEBELOW",(0,-1),(-1,-1),.6,colors.HexColor("#A4BBB1")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
            out.extend([tab,Spacer(1,7)])
        elif group.startswith("!["):
            m=re.fullmatch(r"!\[(.*)\]\((.*)\)",group);path=D/m[2]
            with Image.open(path) as im:wi,he=im.size
            width=W*.96*scale
            out.extend([Spacer(1,2),Figure(str(path),width=width,height=width*he/wi),Spacer(1,4),
                        Paragraph(rich(m[1]),style(8.7,13,8,scale,textColor=colors.HexColor(MUT)))])
        elif group.startswith("> "):
            text=Paragraph(rich(group[2:]),style(10.6,16.5,0,scale,True))
            t=Table([[text]],colWidths=[W]);t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#E8F3EC")),
              ("LINEBEFORE",(0,0),(0,-1),2,colors.HexColor(ACC)),("LEFTPADDING",(0,0),(-1,-1),11),
              ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
            out.extend([t,Spacer(1,6)])
        else:
            if re.match(r"^\[\d+\]",group):sty=style(9.1,13.4,5,scale,wordWrap=None,leftIndent=20,firstLineIndent=-20)
            elif group.startswith("**表 "):sty=style(10,15,3,scale,True)
            else:sty=style(10.5,17,7,scale)
            out.append(Paragraph(rich(group),sty))
    return out
def pdf(text,target):
    os.environ.setdefault("SOURCE_DATE_EPOCH","1794960000")
    for n,f in (("Study","Regular"),("StudyBold","Bold")):pdfmetrics.registerFont(TTFont(n,str(D/"fonts"/f"JSTPSerif-{f}.ttf")))
    pdfmetrics.registerFontFamily("Study",normal="Study",bold="StudyBold")
    c=Canvas(str(target),pagesize=A4,pageCompression=1,invariant=1);c.setAuthor("岳轲")
    c.setTitle("预期能否驱动行动？联合状态转移预测的运动、符号回输与晚到目标实验");c.setSubject("冻结模型、晚到目标与对隐式后果的因果干预")
    c.setCreator("JSTP Intervention Bench / Mechanism Study");w,h=A4
    pages=text.split("<!-- page -->");assert len(pages)==len(CHAPTER)
    for i,p in enumerate(pages):
        title=CHAPTER[i];c.bookmarkPage("s"+str(i));c.addOutlineEntry(title,"s"+str(i))
        c.setFont("Helvetica-Bold",8.4);c.setFillColor(colors.HexColor(ACC));c.drawString(48,h-31,"JSTP   /   MECHANISM STUDIES")
        c.setFillColor(colors.HexColor(MUT));c.setFont("Study",8.3);c.drawRightString(w-48,h-31,"联合状态转移预测　·　晚到目标增订版")
        c.setStrokeColor(colors.HexColor("#A5BEB2"));c.line(48,h-42,w-48,h-42);c.line(48,37,w-48,37)
        c.drawString(48,23,"岳轲　/　"+title);c.setFont("Helvetica",8.2);c.drawRightString(w-48,23,f"{i+1:02}   /   {len(pages):02}")
        height=h-108
        for s in (1,.98,.96,.94,.92,.90,.88,.86):
            items=flow(p,s)
            total=sum(v.wrap(W,height)[1]+v.getSpaceBefore()+v.getSpaceAfter() for v in items)
            if total<height-2:break
        else:raise ValueError(f"排版超出第 {i+1} 页：{total}")
        frame=Frame(48,48,W,height,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        frame.addFromList(items,c)
        if items:raise ValueError("未排出的内容："+title)
        print("page",i+1,"scale",s,"body",round(total));c.showPage()
    c.save()
