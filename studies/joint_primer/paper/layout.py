"""Single-column research-paper layout; Markdown remains the canonical editable article."""
import re
from urllib.parse import quote
from xml.sax.saxutils import escape,quoteattr

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_LEFT,TA_RIGHT,TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate,PageTemplate,Frame,Paragraph,Table,TableStyle,
                                KeepTogether,Image,Spacer,CondPageBreak)

INK='#1C282E'; MUT='#43545B'; RULE='#52656C'; LINK='#205268'
W,H=A4
LEFT=50; RIGHT=50; SW=W-LEFT-RIGHT

def styles(english=False):
    def p(name,font='JR',size=10.5,lead=16.7,after=6,before=0,color=INK,align=TA_JUSTIFY,**kw):
        if english and name!='math':
            font={'JR':'Times-Roman','JB':'Times-Bold'}.get(font,font)
        return ParagraphStyle(name,fontName=font,fontSize=size,leading=lead,spaceBefore=before,spaceAfter=after,
                              textColor=colors.HexColor(color),alignment=align,wordWrap=None if english else 'CJK',splitLongWords=True,
                              allowWidows=0,allowOrphans=0,**kw)
    return {
        'title':p('title','JB',22,32,3,color=INK,align=TA_LEFT,keepWithNext=True),
        'english':p('english','Helvetica',10.8,15,8,color=MUT,align=TA_LEFT,keepWithNext=True),
        'byline':p('byline','JR',10.1,15,11,color=MUT,align=TA_LEFT,keepWithNext=True),
        'resource':p('resource','JR',10.1,14,3,color=INK,align=TA_LEFT,keepWithNext=True),
        'section':p('section','JB',15.3,21,5,10,align=TA_LEFT,keepWithNext=True),
        'sub':p('sub','JB',11.5,17,3,5,align=TA_LEFT,keepWithNext=True),
        'body':p('body'),
        'abstract':p('abstract','JR',10,16,6),
        'keywords':p('keywords','JR',9.4,14,5,color=MUT,align=TA_LEFT),
        'cap':p('cap','JR',9.0,13.5,5,4,align=TA_LEFT),
        'fig':p('fig','JR',8.8,13,6,color=MUT,align=TA_LEFT),
        'ref':p('ref','JR',9.2,13.3,4,align=TA_LEFT,leftIndent=20,firstLineIndent=-20),
        'math':p('math','JR',10.3,17,0,align=TA_CENTER),
        'num':p('num','JR',9.4,13,0,align=TA_RIGHT)
    }

def inline(s, paper):
    out=[]
    for part in re.split(r'(\[[^\]]+\]\([^)]+\))',s):
        link=re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',part)
        if link:
            label,url=link.groups()
            if not url.startswith(('https://','http://')):
                rel=(paper/url).resolve().relative_to(paper.parents[2]).as_posix()
                url='https://github.com/ykykykk/jstp-intervention-bench/blob/codex/jstp-primer/'+quote(rel)
            out.append('<link color="'+LINK+'" href='+quoteattr(url)+'>'+escape(label)+'</link>')
        else:
            part=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',escape(part))
            # English Base-14 faces delegate these to an unembedded Symbol font.
            # A missing minus reverses the apparent result in the ACT comparison.
            part=re.sub('[Δπ−]',lambda m:'<font name="JR">'+m[0]+'</font>',part)
            def ref(m):
                return '['+', '.join('<link color="'+LINK+'" href="#ref-'+x+'">'+x+'</link>' for x in m[1].split(', '))+']'
            out.append(re.sub(r'\[(\d+(?:, \d+)*)\]',ref,part))
    return ''.join(out)

def numeric_table(raw,paper,english=False):
    rows=[[c.strip() for c in line.strip('|').split('|')] for line in raw.splitlines()
          if not re.fullmatch(r'[| :\-]+',line)]
    n=len(rows[0])
    widths={4:(210,96,96,96),7:(137,48,51,55,51,100,55),8:(128,50,50,50,50,50,60,60)}[n]
    if english and n==7:widths=(143,47,48,52,47,98,48)
    if english and n==8:widths=(143,48,48,48,47,47,50,50)
    widths=[SW*x/sum(widths) for x in widths]
    cells=[]
    for i,row in enumerate(rows):
        cells.append([])
        for c,v in enumerate(row):
            face=('Helvetica-Bold' if i==0 else 'Helvetica') if english else ('JB' if i==0 else 'JR')
            sty=ParagraphStyle('cell',fontName=face,fontSize=8.8 if n<8 else 8.5,
                               leading=13,wordWrap=None if english else 'CJK',alignment=TA_LEFT if c==0 else TA_RIGHT,
                               textColor=colors.HexColor(INK))
            cells[-1].append(Paragraph(inline(v,paper),sty))
    t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.95,colors.HexColor(INK)),
                           ('LINEBELOW',(0,0),(-1,0),.48,colors.HexColor(RULE)),
                           ('LINEBELOW',(0,-1),(-1,-1),.9,colors.HexColor(INK)),
                           ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),4),
                           ('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),3),
                           ('BOTTOMPADDING',(0,0),(-1,-1),3)]))
    return t

class Manuscript(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable,'outline_name'):
            label,key=flowable.outline_name
            self.canv.bookmarkHorizontalAbsolute(key,self.frame._y+flowable.height+3)
            self.canv.addOutlineEntry(label,key,level=0,closed=False)
        if hasattr(flowable,'table_label'):
            self.table_pages[flowable.table_label]=self.page

def typeset(paper,filename,text,equations,english=False):
    for name,weight in [('JR','Regular'),('JB','Bold')]:
        pdfmetrics.registerFont(TTFont(name,str(paper/'fonts'/('JSTPSerif-'+weight+'.ttf'))))
    pdfmetrics.registerFontFamily('JR',normal='JR',bold='JB',italic='JR',boldItalic='JB')
    document_title=text.splitlines()[0][2:]
    doc=Manuscript(str(paper/filename),pagesize=A4,leftMargin=LEFT,rightMargin=RIGHT,topMargin=48,bottomMargin=48,
                   title=document_title,author='Ke Yue' if english else '岳轲',
                   subject='Goal-conditioned control with sparse action labels' if english else '联合状态转移预测、功能性分组与动作标注效率',
                   invariant=1,pageCompression=1,allowSplitting=1)
    doc.table_pages={}
    def page_info(c,d):
        c.saveState(); c.setStrokeColor(colors.HexColor('#BAC4C7')); c.setLineWidth(.35)
        if d.page>1:
            c.setFillColor(colors.HexColor(MUT));c.setFont('Helvetica' if english else 'JR',8.4)
            c.drawString(LEFT,H-31,'Ke Yue' if english else '联合状态转移预测导论')
            c.setFont('Helvetica',8);c.drawRightString(W-RIGHT,H-31,'JOINT STATE TRANSITION PREDICTION')
            c.line(LEFT,H-37,W-RIGHT,H-37)
        c.setFillColor(colors.HexColor(MUT));c.setFont('Helvetica',8.5);c.drawCentredString(W/2,27,str(d.page));c.restoreState()
    frame=Frame(LEFT,46,SW,H-94,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id='main')
    doc.addPageTemplates([PageTemplate(id='paper',frames=[frame],onPage=page_info)])
    S=styles(english); story=[];cap=None;sec='';nsec=0
    eqs={f'{latex}\\tag{{{i}}}':(i,rich) for i,(_, (latex,rich)) in enumerate(equations.items(),1)}
    for block in re.split(r'\n\s*\n',text.strip()):
        block=block.strip()
        if cap and not block.startswith('|'):
            raise ValueError('missing table after '+cap[:24])
        if block.startswith('# '):
            story.append(Paragraph(inline(block[2:],paper),S['title']))
        elif block.startswith('Joint State Transition Prediction:'):
            story.append(Paragraph(escape(block),S['english']))
        elif block.startswith(('岳轲　','Ke Yue ·')):
            story.append(Paragraph(escape(block),S['byline']))
        elif block.startswith(('**Project repository:', '**Full JSTP theory:', '**Preprint DOI (v1.2):', '**项目仓库：')):
            story.append(Paragraph(inline(block,paper),S['resource']))
        elif block.startswith('## '):
            sec=block[3:];nsec+=1
            p=Paragraph(inline(sec,paper),S['section']);p.outline_name=(sec,'sec'+str(nsec))
            story.extend([CondPageBreak(55),p])
        elif block.startswith('### '):
            story.extend([CondPageBreak(46),Paragraph(inline(block[4:],paper),S['sub'])])
        elif block.startswith(('**表 ','**Table ')):
            cap=block
        elif block.startswith('|'):
            if not cap:raise ValueError('a table needs a caption')
            p=Paragraph(inline(cap,paper),S['cap']);p.table_label=re.split('[　.]',cap,1)[0].strip('*')
            group=[p,numeric_table(block,paper,english)]
            # A section beginning with a table travels with it, rather than leaving a heading on the preceding page.
            while story and isinstance(story[-1],Paragraph) and story[-1].style.name in ('section','sub'):
                group.insert(0,story.pop())
                if story and isinstance(story[-1],CondPageBreak):story.pop()
            story.extend([KeepTogether(group),Spacer(1,6)])
            cap=None
        elif block.startswith('!['):
            m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',block)
            im=Image(str(paper/m[2])); ratio=im.imageHeight/im.imageWidth
            im.drawWidth=SW;im.drawHeight=SW*ratio
            story.append(KeepTogether([Spacer(1,3),im,Spacer(1,4),Paragraph(inline(m[1],paper),S['fig'])]))
        elif block.startswith('$$'):
            line=block.splitlines()[1];i,rich=eqs[line]
            eq=Table([[Paragraph(rich,S['math']),Paragraph('('+str(i)+')',S['num'])]],colWidths=[SW-36,36])
            eq.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),2),
                                    ('BOTTOMPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),2)]))
            story.append(eq)
        elif re.match(r'^\[\d+\] ',block):
            num,body=block.split(' ',1);i=num.strip('[]')
            story.append(Paragraph('<a name="ref-'+i+'"/>'+escape(num)+(' ' if english else '　')+inline(body,paper),S['ref']))
        else:
            sty='keywords' if block.startswith(('**关键词','**Keywords:')) else 'abstract' if sec in ('摘要','Abstract') else 'body'
            story.append(Paragraph(inline(block.replace('\n',' '),paper),S[sty]))
    if cap:raise ValueError('unplaced table')
    doc.build(story)
    print(filename,doc.page,'pages;',', '.join(f'{k}: p{v}' for k,v in doc.table_pages.items()))
