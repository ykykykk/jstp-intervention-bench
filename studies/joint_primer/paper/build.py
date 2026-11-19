"""Typeset the programme paper from archived estimates, without running a controller."""
import json,re,os
from pathlib import Path
from xml.sax.saxutils import escape,quoteattr
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph,Table,TableStyle,Image,Spacer,Frame

D=Path(__file__).resolve().parent;E=D.parent/'evidence'
PDF='联合状态转移预测导论-岳轲.pdf'
NAMES={'joint':'JSTP 共享转移','aux':'同结构断回输','direct':'仅直接动作','world':'一步世界模型','masked':'掩码轨迹 Transformer','flow':'条件流动作','world-isolated':'隔离梯度世界模型'}
K=[128,1024,16384]
def j(p):return json.loads(p.read_text(encoding='utf8'))
def pct(m):return f"{100*m['value']:.2f}"
def ci(m):return f"{100*m['value']:.2f} [{100*m['lo']:.2f}, {100*m['hi']:.2f}]"
def table(header,body):
    row=lambda a:'| '+' | '.join(a)+' |'
    return '\n'.join([row(header),row(['---']*len(header)),*[row(v) for v in body]])
def compile_text():
    main=j(E/'analysis/summary.json');supp=j(E/'challenge/results.json')
    M={(r['arm'],r['budget']):r['metrics'] for r in main['rows']}
    S={(r['arm'],r['budget']):r for r in supp['rows']}
    cmp={(r['other'],r['budget']):r for r in main['paired']}
    supcmp={(r['a'],r['b'],r['budget']):r['delta'] for r in supp['contrasts']}
    def t(source,metric,arms):
        return table(['模型','128 条','1,024 条','16,384 条'],
             [[NAMES[a]]+[pct(source[a,k][metric]) for k in K] for a in arms])
    names=list(NAMES);m=M['joint',128]
    results={
      'abstract_result':f"在 192 个新布局上，只给 128 条动作示范时，JSTP 在同一次试探后把相反目标两题均做对的比例为 {pct(m['pair'])}%，仅动作策略为 {pct(M['direct',128]['pair'])}%，一步世界模型为 {pct(M['world',128]['pair'])}%。保留物理学习、只断开预期通往动作的路径，成绩为 {pct(M['aux',128]['pair'])}%；示范充足时，这个对照反而略高。保持现实和目标不变，仅替换与物理后果相关的隐分量，一部分动作会按照供体世界的方向改变。这支持的是特定装置上复用物理经验的优势，不是各类智能都需要这种动作通路。",
      'action_table':t(M,'pair',names[:-1]),'group_table':t(M,'group',names[:-1]),'challenge_table':t(S,'pair',names),
      'primary':f"稀缺档里，JSTP 相对仅动作高 {ci(cmp['direct',128]['family'])} 个百分点，相对一步世界模型高 {ci(cmp['world',128]['family'])} 点，相对同结构断回输高 {ci(cmp['aux',128]['family'])} 点。方括号是五项事先比较各用 99% 的布局配对区间，作 Bonferroni 家族校正。三个初始化平均后 JSTP 在这个预算下领先五组；其中一个初始化的一步世界模型仍以 74.4% 高于对应 JSTP 的 72.6%。这些区间不含重新抽取示范或换一批模型初始化的方差。",
      'intervention':f"128 条示范时，在供体与真实最优动作相反的题中，原本正确且向供体换手的比例为 {pct(m['directed'])}%；等幅无关扰动为 {pct(m['null'])}%，两者净差 {ci(m['over_null'])} 个百分点（95% 布局区间）。在最优动作相同的题中，物理移植使原本正确的动作错换了 {pct(m['false'])}%。原预期更贴近真实物理且移植后更贴近供体的比例为 {pct(m['donor_match'])}%。随着示范增至 1,024 和 16,384 条，物理移植与等幅对照的定向换手净差分别为 {pct(M['joint',1024]['over_null'])} 和 {pct(M['joint',16384]['over_null'])} 点。各比率的分母是相应条件下全部题，不是事后只挑会换手的题。",
      'challenge_result':f"隔离梯度没有救回这个实现。128 条示范时，隔离版相对端到端世界模型的成对差为 {ci(supcmp['world-isolated','world',128])} 点；全部标注时亦未反超。相同光学读数上的均方根误差在全部标注时分别为 {S['world-isolated',16384]['optical_rmse']['value']:.3f} 与 {S['world',16384]['optical_rmse']['value']:.3f}（相机位移已除以四像素行程）。因而不能把原来的反常表现简单归咎于动作梯度破坏物理预测。这里没有诊断出唯一原因，也没有在补考后再调对手。"
    }
    results['references']='\n\n'.join(f"[{i}] {a} ({v}). [{title}]({url})." for i,(a,title,v,url) in enumerate(j(D/'references.json'),1))
    text=(D/'source.md').read_text(encoding='utf8')
    for key,val in results.items():text=text.replace('[['+key+']]',val)
    if '[[' in text:raise ValueError('unfilled research field')
    (D/'manuscript.md').write_text(text,encoding='utf8',newline='\n')
    return text

INK='#182E32';ACC='#096B60';MUT='#526966';GOLD='#AE7139';SW=A4[0]-100
def inline(s):
    s=escape(s);s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
    def anchor(m):
        if m[2].startswith('https://'):return '<a color="'+ACC+'" href='+quoteattr(m[2])+'>'+m[1]+'</a>'
        return '<font color="'+ACC+'">'+m[1]+'</font>'
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',anchor,s)
def style(font,size,line,tail,color):
    return ParagraphStyle('j',fontName=font,fontSize=size,leading=line,spaceAfter=tail,
        textColor=colors.HexColor(color),wordWrap='CJK',splitLongWords=1,allowWidows=0,allowOrphans=0)
def pageflows(text,f):
    P=lambda s,font,sz,lh,sp=0,col=INK:Paragraph(inline(s),style(font,sz*f,lh*f,sp*f,col))
    out=[]
    for s in re.split(r'\n\s*\n',text.strip()):
        if s.startswith('|'):
            rows=[[cell.strip() for cell in line.strip('|').split('|')] for line in s.splitlines() if not re.fullmatch(r'[| :\-]+',line)]
            n=len(rows[0]);widths=([.43,.19,.19,.19] if n==4 else [.26,.32,.42])
            cells=[[P(cell,'JB' if i==0 else 'JR',9.1,13,0,'#FFFFFF' if i==0 else INK) for cell in row] for i,row in enumerate(rows)]
            tbl=Table(cells,colWidths=[SW*x for x in widths],hAlign='LEFT')
            tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(ACC)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EDF4F1'),colors.white]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4*f),('TOPPADDING',(0,0),(-1,-1),4*f)]))
            out.extend([tbl,Spacer(1,9*f)]);continue
        if s.startswith('!['):
            m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',s);im=Image(str(D/m[2]));ratio=im.imageHeight/im.imageWidth
            im.drawWidth=SW;im.drawHeight=SW*ratio
            out.extend([im,Spacer(1,6*f),P(m[1],'JR',8.7,13,9,MUT)]);continue
        s=s.replace('\n',' ')
        if s.startswith('# '):out.append(P(s[2:],'JB',26,36,10))
        elif s.startswith('## '):out.append(P(s[3:],'JB',18,27,12,ACC))
        elif s.startswith('### '):out.append(P(s[4:],'JB',12.8,19,6,ACC))
        elif re.match(r'^\[\d+\]',s):out.append(P(s,'JR',9.1,13.8,5,MUT))
        elif s.startswith('Joint State Transition Prediction:'):out.append(P(s,'Helvetica-Oblique',11,16,12,MUT))
        else:out.append(P(s,'JR',11.2,18.6,11))
    return out
def pdf(raw):
    for nm,sty in [('JR','Regular'),('JB','Bold')]:
        pdfmetrics.registerFont(TTFont(nm,str(D/'fonts'/('JSTPSerif-'+sty+'.ttf'))))
    pdfmetrics.registerFontFamily('JR',normal='JR',bold='JB')
    can=Canvas(str(D/PDF),pagesize=A4,invariant=1,pageCompression=1)
    can.setTitle('联合状态转移预测导论：从共同变化到对象、行动与语言');can.setAuthor('岳轲')
    pages=raw.split('<!-- page -->');names=['摘要','局部未来','对象、动作和语言','与已有研究的关系','网络与装置','同数据协议','任务与示范','分组与干预','世界模型补考','理论能被怎样否定','文献与复现']
    if len(pages)!=len(names):raise ValueError('page names do not match manuscript')
    for i,(s,name) in enumerate(zip(pages,names)):
        W,H=A4;can.bookmarkPage(str(i));can.addOutlineEntry(name,str(i))
        can.setFillColor(colors.HexColor(ACC));can.setFont('Helvetica-Bold',8);can.drawString(50,H-34,'JSTP     /     FOUNDATIONS')
        can.setFont('JR',8.5);can.drawRightString(W-50,H-34,name)
        can.setStrokeColor(colors.HexColor('#B9CAC4'));can.line(50,H-43,W-50,H-43);can.line(50,38,W-50,38)
        can.setFillColor(colors.HexColor(MUT));can.setFont('JR',8);can.drawString(50,24,'岳轲   /   联合状态转移预测导论')
        can.setFont('Helvetica',8);can.drawRightString(W-50,24,f'{i+1:02d}  /  {len(pages):02d}')
        avail=H-115
        for fac in [1,.98,.96,.94,.92,.90,.88,.86,.84,.82,.80,.78]:
            flow=pageflows(s,fac)
            if sum(v.wrap(SW,avail)[1]+v.getSpaceAfter()+v.getSpaceBefore() for v in flow)<avail-3:break
        else:raise ValueError('page too long: '+name)
        Frame(50,53,SW,avail,leftPadding=0,bottomPadding=0,topPadding=0,rightPadding=0).addFromList(flow,can)
        if flow:raise ValueError('missing page content: '+name)
        can.showPage()
    can.save();print(PDF,len(pages),'pages')
if __name__=='__main__':pdf(compile_text())
