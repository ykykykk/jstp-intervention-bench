"""Optional chart room. No camera, inference settings or diagnostic plots on the play deck."""
import tkinter as tk

from .game_board import Switchboard, HANDLE, NIGHT, PANEL, SEA, GOLD, WHITE, MUTED, FACE


class HarborLog:
    def __init__(self,station):
        self.app=station
        self.window=self.chart=self.photo=self.guide=None
        self.shown_frame=self.shown_text=self.shown_eye=None

    def text(self,parent,value,size=14,color=WHITE):
        return tk.Label(parent,text=value,fg=color,bg=PANEL,font=(FACE,-size),
                        anchor="w",justify="left",wraplength=362)

    def open(self):
        if self.window is not None:
            self.window.lift(); return
        self.window=win=tk.Toplevel(self.app.root)
        win.title("潮汐港 · 实验仪表"); win.configure(bg=PANEL)
        win.protocol("WM_DELETE_WINDOW",self.close)
        self.text(win,"原始相机",15,SEA).pack(anchor="w",padx=15,pady=(12,3))
        self.cam=tk.Label(win,bg=NIGHT); self.cam.pack(anchor="w",padx=15)
        self.reading=self.text(win,""); self.reading.pack(anchor="w",padx=15,pady=(6,0))
        self.eye_title=self.text(win,"",13,SEA); self.eye_title.pack(anchor="w",padx=15,pady=(7,0))
        self.chart=tk.Canvas(win,width=315,height=40,bg=PANEL,highlightthickness=0)
        self.chart.pack(anchor="w",padx=15,pady=(0,10))
        self.shown_frame=self.shown_text=self.shown_eye=None
        self.update()

    def close(self):
        self.window.destroy()
        self.window=self.photo=self.shown_frame=None

    def update(self):
        if self.window is None: return
        a=self.app
        # Never manufacture a camera frame at an interpolated animation position.
        frame=a.trip.rig.base
        if a.axis=="x" and a.p>=.999:
            frame=a.trip.peak if a.paper else a.shot.peak
        elif a.axis=="y" and a.p>=.999 and a.voyage is not None:
            frame=a.voyage.frame
        if self.shown_frame is not frame:
            from PIL import Image,ImageTk
            self.photo=ImageTk.PhotoImage(Image.fromarray(frame).resize((288,192),Image.Resampling.NEAREST),master=self.window)
            self.cam.configure(image=self.photo)
            self.shown_frame=frame
        probes="   ".join(f"{HANDLE[s]}  "+''.join("←" if x<0 else "→" for x in a.soundings[s].trace)
                          for s in (-1,1) if s in a.soundings)
        marks="、".join(str(i+1) for i,v in enumerate(a.trip.rig.mark) if v)
        gear=(f"换档：{marks} 轨反向" if a.trip.deal.gear else "原档") if a.show_gear() else "档位未发"
        words=[f"布局 {a.trip.deal.layout}    ·    {gear}",
               "已试："+(probes or "无"),f"代驾：{a.arm.get()}  /  {a.budget.get():,} 条  /  种子 {a.seed.get()}",
               "模型读相机特征、公开仪表及目标卡，选 A/B；鼠标仅供人操作。",
               "模拟器负责拉绞盘。测试不提供隐藏接线、真实物体坐标或正确把手。"]
        if a.mode=="watch" and a.last_side is not None:
            words.append(f"模型输入：最后一次完整横试 {HANDLE[a.last_side]}。")
        if a.phase=="result":
            rival=a.spec() if a.mode=="human" else ("human",)
            other=a.attempts.get((a.route,a.trip.deal,rival))
            if other: words.append("同题"+("模型" if a.mode=="human" else "人工")+"："+other)
            w,n=a.tally.get((a.route,a.driver()),(0,0))
            words.append(f"本次该模式：两单都通过 {w}/{n} 局（含重玩）。")
        val="\n".join(words)
        if val!=self.shown_text:
            self.reading.configure(text=val); self.shown_text=val
        pred=a.prediction
        key=(a.mode,id(pred),a.phase=="fault")
        if key==self.shown_eye: return
        self.shown_eye=key; self.chart.delete("all")
        if a.mode=="human": title="人工驾驶，没有模型预期"
        elif a.error: title="代驾没有执行"
        elif pred is None: title="等待模型"
        elif pred.optics is None: title="本适配只输出动作，没有光学读出"
        else:
            title="若竖拉 A：模型的 12 点预期（不是实况）"
            self.chart.create_line(0,20,258,20,fill=MUTED)
            for i,v in enumerate(pred.optics):
                z=max(-1.4,min(1.4,v))*13
                self.chart.create_line(10+i*21,20,10+i*21,20+z,fill=SEA if v<0 else GOLD,width=9)
        self.eye_title.configure(text=title)

    def legend(self):
        if self.guide is not None:
            self.guide.lift(); return
        win=self.guide=tk.Toplevel(self.app.root)
        win.title("码头操作图"); win.configure(bg=PANEL)
        def close():
            win.destroy(); self.guide=None
        win.protocol("WM_DELETE_WINDOW",close)
        art=Switchboard(win)
        art.configure(width=568,height=173); art.pack(fill="both",expand=True)
        def paint(event):
            art.delete("all"); art.s=min(event.width/568,event.height/173)
            art.ox=(event.width-568*art.s)/2; art.oy=(event.height-173*art.s)/2
            for i,title in enumerate(("横拖试绳","下拉送舱","点亮两盏灯")):
                x=11+i*186
                art.rect(x,10,x+175,161,PANEL,"#345363")
                art.txt(x+13,29,f"0{i+1}",13,MUTED,True,"w")
                art.txt(x+88,139,title,16,WHITE,True)
            art.rect(36,71,90,96,"#655037",GOLD,2); art.txt(63,84,"A / B",14,GOLD,True)
            art.arrow(100,83,44,0,GOLD,3); art.line(149,63,149,103,ink=GOLD,dash=(3,3))
            art.dock(283,100,True,False,0); art.cabin(1,283,64); art.arrow(333,57,0,39,SEA,2)
            art.verdict(421,81,True,15); art.verdict(460,81,True,15); art.oval(499,81,15,15,"",MUTED,2)
        art.bind("<Configure>",paint)
        self.text(win,"A / B 选绞盘　 → / ↓ 拉　 ← / ↑ 收　 空格松手　 P 暂停",13,MUTED).pack(padx=10,pady=(0,10))
