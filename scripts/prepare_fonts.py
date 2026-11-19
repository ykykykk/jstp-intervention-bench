"""从锁定的 OFL 字体生成论文排版子集；正常重建论文无需下载或运行本脚本。"""
import hashlib
import argparse
from pathlib import Path
import shutil
import urllib.request
from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/google/fonts/8b0a1d0f5983c89bc2b93f1b5fb55f9e252744b5/ofl/notoserifsc/"
DIGEST = "050080d9255a86808f2945bffac582b31ef32bc36411ce29563b4961670c66f9"

def prepare(paper=None, text_files=None):
    paper=Path(paper) if paper is not None else ROOT/"paper"
    cache, out = ROOT/"work/fonts", paper/"fonts"
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    source = cache/"NotoSerifSC.ttf"
    for filename, remote in ((source,"NotoSerifSC%5Bwght%5D.ttf"),(cache/"OFL.txt","OFL.txt")):
        if not filename.exists():
            urllib.request.urlretrieve(BASE+remote, filename)
    if hashlib.sha256(source.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("字体来源校验失败")
    texts = "".join(chr(i) for i in range(32,127))+"　℃±−–—→×π²·≤≥Δ％（）"
    for name in text_files or ("source.md","references.json","build.py"):
        texts += (paper/name).read_text(encoding="utf8")
    if (paper/"layout.py").exists():texts+=(paper/"layout.py").read_text(encoding="utf8")
    if (paper/"late_section.py").exists():texts+=(paper/"late_section.py").read_text(encoding="utf8")
    chars = set(map(ord,texts))
    for style,weight in (("Regular",450),("Bold",700)):
        font = TTFont(source)
        opt = subset.Options()
        opt.layout_features = []
        sub = subset.Subsetter(options=opt)
        sub.populate(unicodes=chars)
        sub.subset(font)
        font = instantiateVariableFont(font, {"wght":weight}, inplace=True)
        family = "JSTP Study Serif"
        names = {1:family,2:style,4:family+" "+style,6:"JSTPStudySerif-"+style,16:family,17:style}
        for rec in font["name"].names:
            if rec.nameID in names:
                rec.string = names[rec.nameID].encode(rec.getEncoding())
        font["head"].created = font["head"].modified = 3800000000
        font.recalcTimestamp = False
        font.save(out/f"JSTPSerif-{style}.ttf")
        print(style, len(font.getBestCmap()), "glyph mappings")
    shutil.copyfile(cache/"OFL.txt",out/"OFL.txt")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--paper-dir",type=Path)
    ap.add_argument("--text-file",action="append",help="relative typesetting source; repeat for each input")
    args=ap.parse_args()
    prepare(args.paper_dir,args.text_file)
