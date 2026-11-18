"""用逐题归档重建论文表格、插图、Markdown 与 PDF；无需 GPU。"""
import csv,json,sys,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from layout import pdf
D=Path(__file__).resolve().parent;ROOT=D.parents[1];sys.path.insert(0,str(ROOT))
from mechanism.data import CFG,R,images,act,scene
E=ROOT/"evidence/mechanism";FIG=D/"figures";FILE=D/"JSTP-运动与符号回输-岳轲.pdf"
NAMES={"joint_symbol":"共同预测＋符号","aux_symbol":"预测后果但不回输","direct_symbol":"直接运动对照","joint_holistic":"共同预测＋整体码","joint_latent":"共同预测＋自学码"}
def readcsv(p):
    with p.open(encoding="utf8",newline="") as f:return list(csv.DictReader(f))
def font(size,bold=False):return ImageFont.truetype(str(D/"fonts"/("JSTPSerif-Bold.ttf" if bold else "JSTPSerif-Regular.ttf")),size)
def data():
    return {(r["split"],r["arm"],r["metric"]):r for r in readcsv(E/"analysis/metrics.csv")}
def tables():
    ms=data()
    val=lambda s,a,k:float(ms[s,a,k]["value"])*100
    fmt=lambda x:f"{x:.2f}"
    def tab(title,head,rows):
        line=lambda r:"| "+" | ".join(r)+" |"
        return "**"+title+"**\n\n"+"\n".join([line(head),line(["---"]*len(head)),*(line(r) for r in rows)])
    m=[]
    for a,label in NAMES.items():
        g=lambda s,k:fmt(val(s,a,k)) if (s,a,k) in ms else "—"
        m.append([label,g("test_familiar","task"),g("test_familiar","strict"),g("test_familiar","specific"),g("test_novel","task"),g("test_novel","strict")])
    ht={(r["arm"],r["metric"]):r for r in readcsv(E/"latent_followup/metrics.csv") if r["split"]=="familiar" and r["seed"]=="pooled"}
    ha=[]
    for a in list(NAMES)[:3]:
        ha.append([NAMES[a],*[fmt(float(ht[a,k]["value"])*100) if (a,k) in ht else "—" for k in
                              ("relevant_motor","relevant_physical","relevant_both","nuisance_both","random_both")]])
    ll=[]
    for a in ("joint_symbol","joint_holistic","joint_latent"):
        ll.append([NAMES[a],*[fmt(val("depth"+str(d),a,"accuracy")) for d in (4,6,8)],
                   fmt(val("depth8",a,"cleared")),fmt(val("depth8",a,"donor_target"))])
    return dict(MOTOR=tab("表 1　主动成功率、严格共动配对（%）与感觉回输的定向差（百分点）",
        ["训练条件","熟悉·成功","熟悉·共动","熟悉·定向差","新连接·成功","新连接·共动"],m),
        LATENT=tab("表 2　新抽熟悉布局的两维隐状态移植（%，分母是全部主动题）",["模型","定向换手","物理转向","两者均变","无关·均变","随机·均变"],ha),
        LANG=tab("表 3　电路的非语言操作正确率（%）；供体列按供体正确答案核对",["同一模型的回输","四步","六步","八步","八步·清空","八步·供体"],ll))
def figures():
    FIG.mkdir(exist_ok=True);ink="#1B3737";green="#116755";orange="#A7512B"
    f=E/"sealed/test_familiar";r=np.load(f/"input.npz")["rows"];z=np.load(f/"joint_symbol-1729.npz")
    phys=np.load(E/"observed/test_familiar/apparatus.npz");s=scene(CFG["test_familiar"][0]);im=Image.new("RGB",(1460,500),"white");d=ImageDraw.Draw(im)
    for i,t in enumerate(("先前试探","自提运动的预期","另一挂接的预期","正式驱动实测")):d.text((200+353*i,30),t,font=font(29,True),fill=green,anchor="mm")
    heat=lambda v:(np.stack([v*.38,v,v*.78],-1)*17).clip(0,255).astype("uint8").reshape(8,12,3)
    for j in range(2):
        ix=np.where((r[:,0]==0)&(r[:,1]==j)&(r[:,2]==0)&(r[:,3]==1)&(r[:,4]==1)&(r[:,5]==0))[0][0]
        u=int(z["u"][ix,0]);b,peak=images(s,j);actual=act(s,j,u)
        assert hashlib.sha256(actual.tobytes()).hexdigest().encode()==phys["frame_sha"][0,j,1,u+1]
        d.text((19,164+j*216),str(j+1),font=font(26,True),fill=ink,anchor="mm")
        for k,v in enumerate((peak,heat(z["prospect"][ix,:R].astype(float)),heat(z["donor"][ix].astype(float)),actual)):
            x=45+k*353;y=64+j*216
            im.paste(Image.fromarray(v).resize((307,202),Image.Resampling.NEAREST),(x,y))
    im.save(FIG/"twins.png",optimize=True)
    stats=data();im=Image.new("RGB",(1480,400),"white");d=ImageDraw.Draw(im)
    X=lambda v:358+v*9.3
    for v in (0,25,50,75,100):
        x=X(v);d.line((x,56,x,335),fill="#D9E2DE",width=2);d.text((x,363),str(v),font=font(22),fill=ink,anchor="mm")
    for i,(k,t) in enumerate((("task","完成任务"),("strict","局部共动均正确"),("specific","定向运动效应差"))):
        y=99+102*i;d.text((300,y),t,font=font(28,True),fill=ink,anchor="rm")
        for j,(sp,col) in enumerate((("test_familiar",green),("test_novel",orange))):
            rr=stats[sp,"joint_symbol",k];a,b,v=[float(rr[u])*100 for u in ("lo","hi","value")]
            yy=y+(j*2-1)*11;d.line((X(a),yy,X(b),yy),fill=col,width=4);x=X(v)
            if j==0:d.ellipse((x-8,yy-8,x+8,yy+8),fill=col)
            else:d.rectangle((x-7,yy-7,x+7,yy+7),fill=col)
    d.text((395,12),"● 熟悉机构",font=font(24,True),fill=green);d.text((655,12),"■ 新机构",font=font(24,True),fill=orange)
    im.save(FIG/"dissociation.png",optimize=True)
def build():
    figures();fields=tables();refs=json.loads((D/"references.json").read_text(encoding="utf8"))
    from late_section import material
    fields.update(material(ROOT/"evidence/late-goals",FIG,font))
    fields["REFS"]="\n\n".join(f'[{i}] {r["short"]}. [{r["title"]}]({r["url"]}). {r["venue"]}，{r["year"]}。' for i,r in enumerate(refs,1))
    re=json.loads((E/"independent-reproduction.json").read_text())
    n=sum(v["decisions"] for v in re["ledgers"].values());diff=sum(v["changed_decisions"] for v in re["ledgers"].values())
    fields["REPRO"]=f"同一设备在独立的 GitHub 克隆与新环境中，从种子重建训练资料，重新训练全部 15 份模型。两次的 {n:,} 个主试验离散提交逐项比较，不同数为 {diff:,}。这些提交含对照和交换，分母不是独立场景数。同机复现不代表跨设备或第三方复现。"
    text=(D/"source.md").read_text(encoding="utf8")
    for k,v in fields.items():text=text.replace("{{"+k+"}}",v)
    if "{{" in text:raise ValueError("未填字段")
    (D/"manuscript.md").write_text(text,encoding="utf8",newline="\n")
    bib="\n\n".join("@misc{"+r["key"]+",\n"+",\n".join("  "+k+" = {"+r[k]+"}" for k in ("author","title","year","url"))+"\n}" for r in refs)
    (D/"references.bib").write_text(bib+"\n",encoding="utf8",newline="\n")
    pdf(text,FILE)
if __name__=="__main__":build()
