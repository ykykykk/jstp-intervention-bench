"""Plot the actual fixed-batch validation records, not the held-out test curves."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent;E=D.parent/"evidence/models"
def draw():
    pic=Image.new("RGB",(1056,264),"#f2f6f2");d=ImageDraw.Draw(pic);f=D/"fonts/JSTPSerif-Regular.ttf";font=ImageFont.truetype(str(f),18)
    xx=lambda u:55+(u-256)*920/3840; yy=lambda v:207-(v-.45)*283
    for v in (.5,.75,1):
        y=yy(v);d.line((52,y,985,y),fill="#cddbd2");d.text((1,y-14),str(round(v*100))+"%",font=font,fill="#385458")
    colors={"partial":("#145c4b","JSTP-P"),"direct":("#4288bc","仅动作"),"world":("#9f5722","世界模型"),"token":("#714b88","动作词元"),"flow":("#767c27","条件流"),"detached":("#a85661","断回输")}
    for j,(arm,(color,label)) in enumerate(colors.items()):
        lines=[[json.loads(x) for x in (E/f"{arm}-{s}"/"curve.jsonl").read_text().splitlines()] for s in (6043,6221,6553)]
        val=np.mean([[r["validation"] for r in line] for line in lines],0)
        d.line([(xx(r["update"]),yy(v)) for r,v in zip(lines[0],val)],fill=color,width=3)
        x=60+j*156;d.line((x,20,x+20,20),fill=color,width=4);d.text((x+24,6),label,font=font,fill="#193239")
    for u in (256,1024,2048,3072,4096):d.text((xx(u)-14,219),str(u),font=font,fill="#365457")
    pic.save(D/"figures/training.png")
if __name__=="__main__":draw()

