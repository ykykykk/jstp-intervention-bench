"""Build the English edition and its self-contained pdfLaTeX submission from archived results."""
import csv
import gzip
import io
import re
import tarfile
import numpy as np

from build import D, E, EXTERNAL, EQUATIONS, NAMES, fill_template, j, study_fields

PDF = 'JSTP-Goal-Generalization-Ke-Yue.pdf'
VERSION_DOI = '10.5281/zenodo.24392467'
RECORD = VERSION_DOI.rsplit('.', 1)[1]
NAMES_EN = {
    'joint': 'JSTP (shared transition)', 'world': 'One-step world model',
    'aux': 'No consequence feedback', 'masked': 'Masked trajectory model',
    'dino': 'DINO-WM adaptation', 'direct': 'Action-only policy',
    'flow': 'Conditional flow policy', 'smol': 'SmolVLA fine-tuning',
    'world-isolated': 'Gradient-isolated world model', 'act': 'ACT adaptation',
    'diffusion': 'Diffusion Policy adaptation', 'vjepa21': 'V-JEPA 2.1 readout',
}
HEADERS = {
    'main_table': ['Method', '128', '1,024', '16,384', 'Δ128', 'Adjusted interval', 'Wins'],
    'external_table': ['Method', '128', '1,024', '16,384', 'Δ128', '99% interval', 'Wins'],
    'detail_table': ['Method', '9301', '9419', '9547', 'Single', 'Same', 'Group', 'Blind'],
    'challenge_table': ['Method', '128', '1,024', '16,384'],
    'group_table': ['Method', '128', '1,024', '16,384'],
}


def english_text():
    fields = study_fields()
    fields.update(version_doi=VERSION_DOI, record=RECORD)
    translations = {NAMES[k]: v for k, v in NAMES_EN.items()}
    translations['未测'] = 'n.m.'
    for key, header in HEADERS.items():
        rows = fields[key].splitlines()
        translated = ['| ' + ' | '.join(header) + ' |', rows[1]]
        for row in rows[2:]:
            cells = [v.strip() for v in row.strip('|').split('|')]
            translated.append('| ' + ' | '.join(translations.get(v, v) for v in cells) + ' |')
        fields[key] = '\n'.join(translated)
    return fill_template(D / 'source-en.md', fields)


def tex_plain(value):
    replacements = {
        '&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#', '$': r'\$',
        '{': r'\{', '}': r'\}', 'π': r'\ensuremath{\pi}', 'Δ': r'\ensuremath{\Delta}',
        '−': r'\ensuremath{-}', '–': '--', '—': '---', '“': chr(96) * 2,
        '”': "''", '’': "'", '·': r'\textperiodcentered{}',
    }
    return ''.join(replacements.get(c, c) for c in value)


def tex_inline(value):
    tokens = r'(\*\*[^*]+\*\*|\[[^]]+\]\([^)]+\)|\[\d+(?:, \d+)*\])'
    out = []
    for part in re.split(tokens, value):
        link = re.fullmatch(r'\[([^]]+)\]\(([^)]+)\)', part)
        cite = re.fullmatch(r'\[(\d+(?:, \d+)*)\]', part)
        if link:
            label, url = link.groups()
            out.append(r'\href{' + url.replace('%', r'\%').replace('#', r'\#') + '}{' + tex_plain(label) + '}')
        elif cite:
            out.append(r'\cite{' + ','.join('r' + n for n in cite[1].split(', ')) + '}')
        elif part.startswith('**') and part.endswith('**'):
            out.append(r'\textbf{' + tex_inline(part[2:-2]) + '}')
        else:
            out.append(tex_plain(part))
    return ''.join(out)


PREAMBLE = r"""% English manuscript of Ke Yue. Compile with pdflatex main.tex twice.
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,graphicx,caption,placeins}
\usepackage{xcolor}
\usepackage[colorlinks=true,allcolors=blue!40!black]{hyperref}
\graphicspath{{./}{../figures/}}
\hypersetup{pdfauthor={Ke Yue},pdftitle={Joint State Transition Prediction: Goal Generalization with Sparse Action Labels}}
\setlength{\emergencystretch}{1.7em}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}
\setcounter{topnumber}{3}\renewcommand{\topfraction}{.9}\renewcommand{\textfraction}{.08}
\title{Joint State Transition Prediction:\\Goal Generalization with Sparse Action Labels}
\author{Ke Yue\\\small Independent Researcher}
\date{v1.2 \quad | \quad Protocol clarified 21 November 2026}
\begin{document}
\maketitle
"""


def tex_table(caption, text):
    header, _, *body = text.splitlines()
    parts = lambda line: [tex_inline(v.strip()) for v in line.strip('|').split('|')]
    header = parts(header)
    cols = r'@{}>{\raggedright\arraybackslash}X' + 'r' * (len(header) - 1) + '@{}'
    label = re.fullmatch(r'\*\*Table (A?\d+)\. (.*?)\*\*(.*)', caption)
    if caption.startswith('**Worked example.'):
        out = [r'\par\medskip\noindent\begin{minipage}{\linewidth}\small',
               tex_inline(caption) + r'\par\smallskip']
        end = r'\end{minipage}\par\medskip'
    else:
        out = [r'\begin{table}[htbp]', r'\centering\small\setlength{\tabcolsep}{3.5pt}',
               r'\caption{' + tex_inline(label[2] + label[3]) + '}', r'\label{tab:' + label[1] + '}']
        end = r'\end{table}'
    out += [r'\begin{tabularx}{\linewidth}{' + cols + '}', r'\toprule',
           ' & '.join(r'\textbf{' + v + '}' for v in header) + r' \\', r'\midrule']
    out += [' & '.join(parts(row)) + r' \\' for row in body]
    out += [r'\bottomrule', r'\end{tabularx}', end]
    return '\n'.join(out)


def tex_document(text):
    blocks = re.split(r'\n\s*\n', text.strip())
    out = [PREAMBLE]
    abstract = False
    cap = None
    for b in blocks[2:]:  # title and author are typeset by \maketitle
        if b.startswith(('**Project repository:', '**Full JSTP theory:', '**Preprint DOI (v1.2):')):
            out.append(r'\noindent{\small ' + tex_inline(b) + r'}\par')
        elif b == '## Abstract':
            out.append(r'\begin{abstract}')
            abstract = True
        elif b.startswith('**Keywords:'):
            out += [r'\end{abstract}', r'\noindent\small ' + tex_inline(b) + r'\normalsize']
            abstract = False
        elif b.startswith('## Appendix A. '):
            out += [r'\FloatBarrier\appendix\setcounter{table}{0}\renewcommand{\thetable}{A\arabic{table}}',
                    r'\section{' + tex_inline(b.split('. ', 1)[1]) + '}']
        elif b == '## References':
            out += [r'\FloatBarrier\begin{thebibliography}{99}']
        elif b.startswith('## ') or b.startswith('### '):
            s = b.startswith('### ')
            title = re.sub(r'^#{2,3} \d+(?:\.\d+)?\.? ', '', b)
            out.append((r'\subsection{' if s else r'\FloatBarrier\section{') + tex_inline(title) + '}')
        elif b.startswith(('**Table ', '**Worked example.')):
            cap = b
        elif b.startswith('|'):
            if not cap:
                raise ValueError('table without caption')
            out.append(tex_table(cap, b))
            cap = None
        elif b.startswith('!['):
            fig = re.fullmatch(r'!\[Figure 1\. (.*?)\]\(figures/one-history\.png\)', b)
            out.append(r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{one-history.png}' +
                       '\n' + r'\caption{' + tex_inline(fig[1]) + r'}\label{fig:history}\end{figure}')
        elif b.startswith('$$'):
            out.append('\\begin{equation*}\n' + b.splitlines()[1] + '\n\\end{equation*}')
        elif re.match(r'^\[\d+\] ', b):
            n, ref = b.split(' ', 1)
            out.append(r'\bibitem{r' + n[1:-1] + '} ' + tex_inline(ref))
        else:
            out.append(tex_inline(b))
    if abstract or cap:
        raise ValueError('incomplete manuscript block')
    out += [r'\end{thebibliography}', r'\end{document}', '']
    return '\n\n'.join(out)


def extension_files():
    """Package recorded layout counts so Table 2 can be reconstructed without a GPU or a future repository link."""
    with np.load(E/'sealed/items.npz',allow_pickle=False) as items:
        ids=items['ids']
    counts=io.StringIO(newline='');writer=csv.writer(counts,lineterminator='\n')
    writer.writerow(('method','labels','seed','layout_id','pairs_won','pairs'))
    for r in j(EXTERNAL/'per-seed.json'):
        a,k,s=r['arm'],r['budget'],r['seed']
        with np.load(EXTERNAL/'observed'/f'{a}-{k}-{s}.npz',allow_pickle=False) as rec:
            w=rec['wins'];pairs=(w[...,::2]&w[...,1::2]).reshape(len(ids),-1)
        if int(pairs.sum())!=r['won'] or pairs.size!=r['pairs']:
            raise ValueError(f'Published pair counts disagree with executed outcomes: {a}-{k}-{s}')
        writer.writerows((a,k,s,int(i),int(n),pairs.shape[1]) for i,n in zip(ids,pairs.sum(1),strict=True))
    return {'anc/fourtrack-layout-counts.csv':counts.getvalue().encode('utf8'),
            'anc/fourtrack-summary.json':(EXTERNAL/'summary.json').read_bytes(),
            'anc/fourtrack-per-seed.json':(EXTERNAL/'per-seed.json').read_bytes()}


def submission_files(text):
    name = 'JSTP-arXiv-source.tar.gz'
    manuscript = tex_document(text)
    src = D / 'arxiv'
    src.mkdir(exist_ok=True)
    (src / 'main.tex').write_text(manuscript, encoding='utf8', newline='\n')
    # One TeX root and its figure. Ancillary data are not required to compile it.
    files = {'main.tex': manuscript.encode('utf8'), 'one-history.png': (D / 'figures/one-history.png').read_bytes(),
             **extension_files()}
    with (D / name).open('wb') as stream, gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tar:
            for key, value in files.items():
                entry = tarfile.TarInfo(key)
                entry.size = len(value)
                entry.mode = 0o644
                tar.addfile(entry, io.BytesIO(value))
    title = text.splitlines()[0][2:]
    abstract = text.split('## Abstract\n\n', 1)[1].split('\n\n', 1)[0]
    keywords = next(line for line in text.splitlines() if line.startswith('**Keywords:'))
    byline = text.split('\n\n', 2)[1]
    short = (f'# {title}\n\n{byline}\n\n'
             f'## Abstract\n\n{abstract}\n\n{keywords}\n\n'
             f'[Preprint DOI (v1.2)](https://doi.org/{VERSION_DOI}) · [Revised English PDF]({PDF}) · '
             '[Full English text](manuscript-en.md) · [Chinese edition](manuscript.md)\n\n'
             'Full theory: [English](../../../theory/JSTP-Theory.en.md) · '
             '[Chinese original](../../../theory/JSTP-详细理论稿-2026-11-19.md)\n')
    (D / 'abstract-en.md').write_text(short, encoding='utf8', newline='\n')
    ntables=len(re.findall(r'^\*\*Table ',text,re.M))
    meta = (f'Title: {title}\nAuthors: Ke Yue\nPrimary category: cs.LG (Machine Learning)\n'
            f'Comments: Revised manuscript v1.2; 1 figure, {ntables} numbered tables and a worked scoring example. Per-layout counts supplied as ancillary data.\n'
            f'Preprint DOI (v1.2): https://doi.org/{VERSION_DOI}\n\nAbstract:\n{abstract}\n')
    (src / 'metadata.txt').write_text(meta, encoding='utf8', newline='\n')


if __name__ == '__main__':
    from layout import typeset
    manuscript = english_text()
    (D / 'manuscript-en.md').write_text(manuscript, encoding='utf8', newline='\n')
    submission_files(manuscript)
    typeset(D, PDF, manuscript, EQUATIONS, english=True)
