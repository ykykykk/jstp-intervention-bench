"""Typeset the complete theory in both languages; the two Markdown texts are authoritative."""
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from xml.sax.saxutils import escape, quoteattr

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents

D = Path(__file__).resolve().parent
ROOT = D.parent
REPO = 'https://github.com/ykykykk/jstp-intervention-bench'
TAG = 'v5.2.0'
DOI = 'https://doi.org/10.5281/zenodo.24366435'
EN = 'JSTP-Theory.en.md'
ZH = 'JSTP-详细理论稿-2026-11-19.md'
W, H = A4
MARGIN = 52
WIDTH = W - 2 * MARGIN
INK, TEAL, MUTE = '#203039', '#175D62', '#445961'
c = colors.HexColor


def external(target):
    parts = urlsplit(target)
    if parts.scheme:
        return target
    path = (D / unquote(parts.path)).resolve().relative_to(ROOT).as_posix()
    return f'{REPO}/blob/{TAG}/' + quote(path, safe='/') + (
        '#' + quote(unquote(parts.fragment), safe='-') if parts.fragment else '')


def link(label, target):
    return '<link color="' + TEAL + '" href=' + quoteattr(target) + '>' + escape(label) + '</link>'


def markup(items):
    out = []
    for item in items:
        t, v = item['t'], item.get('c')
        if t == 'Str': out.append(escape(v))
        elif t in ('Space', 'SoftBreak'): out.append(' ')
        elif t == 'LineBreak': out.append('<br/>')
        elif t in ('Strong', 'Emph'):
            tag = 'b' if t == 'Strong' else 'i'
            out.append(f'<{tag}>{markup(v)}</{tag}>')
        elif t == 'Code': out.append('<font name="TheoryRegular">' + escape(v[1]) + '</font>')
        elif t == 'Link':
            dest = v[2][0]
            out.append('<link color="' + TEAL + '" href=' + quoteattr(dest if dest.startswith('#') else external(dest))
                       + '>' + markup(v[1]) + '</link>')
        else: raise ValueError(f'Unsupported theory inline: {t}')
    return ''.join(out)


def formatted_table(spec, style, section):
    head = spec[3][1]
    body = [row for part in spec[4] for row in part[2] + part[3]]
    rows = []
    for i, (_, cells) in enumerate(head + body):
        row = []
        for cell in cells:
            row.append(Paragraph('<br/>'.join(markup(block['c']) for block in cell[4]),
                                 style['th' if i < len(head) else 'td']))
        rows.append(row)
    n = len(rows[0])
    if n == 2:
        frac = (.44, .56) if section == 'B' else (.24, .76)
    elif section == '15': frac = (.20, .53, .27)
    elif section == '17': frac = (.08, .64, .28)
    else: frac = (.25, .37, .38)
    t = Table(rows, colWidths=[WIDTH*x for x in frac], repeatRows=len(head), hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),c('#e9f0ef')),
        ('LINEABOVE',(0,0),(-1,0),.7,c(TEAL)), ('LINEBELOW',(0,0),(-1,0),.4,c(TEAL)),
        ('LINEBELOW',(0,-1),(-1,-1),.6,c(TEAL)), ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,c('#f6f8f7')]),
        ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    return t


class TheoryDoc(BaseDocTemplate):
    def afterFlowable(self, flow):
        if hasattr(flow, 'entry'):
            level, key, label, for_contents = flow.entry
            self.canv.bookmarkHorizontalAbsolute(key, self.frame._y + flow.height + 4)
            self.canv.addOutlineEntry(label, key, level, closed=level == 0)
            if for_contents: self.notify('TOCEntry', (0, escape(label), self.page, key))


def build(language, source):
    en = language == 'en'
    blocks = json.loads(subprocess.check_output(
        ['pandoc', '-f', 'gfm', '-t', 'json', str(D/source)]
    ))['blocks']
    body_face, bold = ('Times-Roman', 'Times-Bold') if en else ('TheoryRegular', 'TheoryBold')
    def s(name, font=body_face, size=11, lead=15.7, before=0, after=5, color=INK, **opts):
        return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead, spaceBefore=before,
                              spaceAfter=after, textColor=c(color), wordWrap=None if en else 'CJK',
                              allowWidows=0, allowOrphans=0, **opts)
    styles = {
        'body': s('body'), 'title': s('title', bold, 26, 31, after=6, keepWithNext=True),
        'subtitle': s('subtitle', body_face, 14, 19, after=12, color=TEAL, keepWithNext=True),
        'meta': s('meta', size=10, lead=14.5, after=6, color=MUTE),
        'h2': s('chapter', bold, 17, 23, 13, 6, keepWithNext=True),
        'h3': s('subsection', bold, 12, 17.5, 7, 3, keepWithNext=True),
        'quote': s('quotation', size=11.1, lead=16.5, before=6, after=10,
                   leftIndent=13, rightIndent=11, borderPadding=8, backColor=c('#edf3f2')),
        'td': s('cell', size=9.1, lead=13.4, after=0),
        'th': s('table header', bold, 9.2, 13.5, after=0),
        'bullet': s('bullet', leftIndent=15, firstLineIndent=-10, after=3),
    }
    title = Paragraph(markup(blocks[0]['c'][2]), styles['title']).getPlainText()
    out = D/f'JSTP-Theory-Ke-Yue.{language}.pdf'
    doc = TheoryDoc(str(out), pagesize=A4, title=title, author='Ke Yue' if en else '岳轲',
                    subject='JSTP — full theoretical manuscript', invariant=1, pageCompression=1)
    def page(canvas, d):
        canvas.saveState()
        if d.page > 1:
            canvas.setFillColor(c(MUTE)); canvas.setFont('Helvetica', 8)
            canvas.drawString(MARGIN, H-30, 'JOINT STATE TRANSITION PREDICTION')
            canvas.drawRightString(W-MARGIN, H-30, 'FULL THEORY  /  ' + language.upper())
            canvas.setStrokeColor(c('#c3ced0')); canvas.setLineWidth(.4); canvas.line(MARGIN,H-37,W-MARGIN,H-37)
        canvas.setFillColor(c(MUTE)); canvas.setFont('Helvetica',8.5)
        canvas.drawCentredString(W/2, 26, str(d.page)); canvas.restoreState()
    doc.addPageTemplates([PageTemplate('theory',[Frame(MARGIN,47,WIDTH,H-99,leftPadding=0,rightPadding=0,
                            topPadding=0,bottomPadding=0)],onPage=page)])
    toc = TableOfContents()
    toc.levelStyles = [s('contents entry',size=11,lead=17,after=3,textTransform=None)]
    toc.dotsMinLevel = 0
    story = [Paragraph('JSTP &nbsp; / &nbsp; THEORETICAL MANUSCRIPT',s('kicker','Helvetica-Bold',8.4,13,after=6,color=TEAL))]
    in_toc = False
    chapter = ''
    section_count = 0
    for i, b in enumerate(blocks):
        kind, v = b['t'], b.get('c')
        if kind == 'Header':
            level, key, words = v[0], v[1][0], markup(v[2])
            if i < 2:
                story.append(Paragraph(words, styles['title' if i==0 else 'subtitle']))
                continue
            h = Paragraph(words, styles['h2' if level==2 else 'h3'])
            title_text = h.getPlainText()
            if level == 2:
                if in_toc: story.append(PageBreak())
                in_toc = title_text in ('Contents','目录')
                if in_toc:
                    story.extend([PageBreak(), h, Spacer(1,4), toc])
                    h.entry = (0, key, title_text, False)
                    continue
                if re.match(r'^(?:\d+\. |[一二三四五六七八九十]+、)', title_text):
                    section_count += 1; chapter = str(section_count)
                elif title_text.startswith(('Appendix A', '附录 A')): chapter = 'A'
                elif title_text.startswith(('Appendix B', '附录 B')): chapter = 'B'
            h.entry = (level-2, key, title_text, bool(chapter) and level==2)
            story.extend([CondPageBreak(65 if level==2 else 48), h])
        elif in_toc:
            continue  # Print an equivalent linked contents list with actual PDF page numbers.
        elif kind in ('Para','Plain'):
            if en and i==3 and any(w['t']=='Link' for w in v): continue  # linked edition strip below
            story.append(Paragraph(markup(v),styles['meta' if i==2 else 'body']))
            if i==2:
                nav = (link('English' if en else '英文全文', external(EN))+' &nbsp; / &nbsp; '+
                       link('Chinese original' if en else '中文原稿', external(ZH))+' &nbsp; / &nbsp; '+
                       link('Empirical preprint' if en else '实证论文', DOI))
                story.append(Paragraph(nav, styles['meta']))
        elif kind == 'BlockQuote':
            story.extend(Paragraph(markup(p['c']),styles['quote']) for p in v)
        elif kind == 'CodeBlock':
            code = v[1].expandtabs(4)
            def runs(line):
                return re.findall(r'[\x00-\x7f]+|[^\x00-\x7f]+',line)
            def face(run):
                return 'Courier' if en and run.isascii() else 'TheoryRegular'
            longest = max(sum(pdfmetrics.stringWidth(x,face(x),1) for x in runs(line))
                          for line in code.splitlines())
            fs = min(9, (WIDTH-22)/max(1,longest))
            rich = '\n'.join(''.join('<font name="'+face(x)+'">'+escape(x)+'</font>' for x in runs(line))
                             for line in code.splitlines())
            story.append(XPreformatted(rich,s('diagram', size=fs, lead=fs*1.53, before=3, after=7,
                                             leftIndent=10, rightIndent=10, backColor=c('#f4f6f6'), borderPadding=5)))
        elif kind == 'Table':
            story.extend([formatted_table(v,styles,chapter),Spacer(1,6)])
        elif kind == 'BulletList':
            for row in v: story.append(Paragraph('• &nbsp;'+markup(row[0]['c']),styles['bullet']))
        elif kind == 'HorizontalRule':
            pass  # Chapter spacing takes the place of the Markdown dividers in print.
        else:
            raise ValueError(f'Unsupported theory block: {kind}')
    doc.multiBuild(story)
    print(out.name, doc.page, 'pages')


if __name__ == '__main__':
    for weight in ('Regular','Bold'):
        pdfmetrics.registerFont(TTFont('Theory'+weight, str(D/'fonts'/f'JSTPSerif-{weight}.ttf')))
    pdfmetrics.registerFontFamily('TheoryRegular',normal='TheoryRegular',bold='TheoryBold',
                                  italic='TheoryRegular',boldItalic='TheoryBold')
    build('en', EN)
    build('zh', ZH)
