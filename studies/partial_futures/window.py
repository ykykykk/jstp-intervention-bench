"""A live isolated apparatus, not a screen recorder or an OS input controller."""
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
from PIL import Image,ImageTk
from lategoal.rig import Rig,first,history,EXAM,lane_travel,optical_delta,goal_wins
from lategoal.control import T
from mechanism.data import read
from .train import C,DEFAULT,load,SUP

class Window:
    def __init__(self,win):
        self.win=win;win.title("JSTP · 局部未来 / 独立四轨试验台")
        self.arm=tk.StringVar(value="partial"); self.gidx=0;self.sid=90286000;self.altered=tk.BooleanVar()
        self.note=tk.StringVar();self.card=tk.StringVar();self.arm_name=None
        box=ttk.Frame(win,padding=16);box.grid();ttk.Label(box,text="四轨试验台  /  真实操作与未来预期",font=("Microsoft YaHei UI",15,"bold")).grid(row=0,column=0,columnspan=3,sticky="w")
        menu=ttk.Combobox(box,textvariable=self.arm,values=C["arms"],state="readonly",width=12);menu.grid(row=1,column=0,sticky="w");menu.bind("<<ComboboxSelected>>",lambda e:self.new())
        ttk.Label(box,textvariable=self.card,foreground="#08778a").grid(row=1,column=1,columnspan=2,sticky="w")
        self.real=tk.Canvas(box,width=384,height=256,bg="#162534",highlightthickness=0);self.real.grid(row=2,column=0,columnspan=2)
        self.eye=tk.Canvas(box,width=320,height=256,bg="#172a34",highlightthickness=0);self.eye.grid(row=2,column=2,padx=(12,0))
        btn=ttk.Frame(box);btn.grid(row=3,column=0,columnspan=3,sticky="w",pady=9)
        for t,fn in (("新布局",self.new),("① 横向试探",self.probe),("换相反目标",self.flip),("② 正式执行",self.act)):
            ttk.Button(btn,text=t,command=fn).pack(side="left",padx=2)
        self.ck=ttk.Checkbutton(btn,text="只换隐式后果",variable=self.altered,command=self.preview);self.ck.pack(side="left",padx=6)
        ttk.Label(box,textvariable=self.note,wraplength=715).grid(row=4,column=0,columnspan=3,sticky="w")
        self.new()
    def paint(self,im):
        self.im=ImageTk.PhotoImage(Image.fromarray(im).resize((384,256),Image.Resampling.NEAREST))
        self.real.delete("all");self.real.create_image(0,0,anchor="nw",image=self.im)
    def new(self):
        self.sid+=1;self.phase="ready";self.gidx=(self.gidx+2)%len(EXAM);self.altered.set(False)
        a=self.arm.get()
        if a!=self.arm_name:
            self.m=load(DEFAULT,a,C["seeds"][0]);self.arm_name=a
        self.gear=self.sid%2;self.wiring=(self.sid//2)%2;self.r=Rig(self.sid);self.w=self.r.world(self.wiring)
        self.paint(self.r.base);self.eye.delete("all");self.card.set("目标尚未揭示");self.ck.state(["!disabled"] if a=="partial" else ["disabled"])
        self.note.set("左：真实相机。模型先选择把手横拉并复位。右：给目标前的后果预期，不是录像。")
    @torch.inference_mode()
    def probe(self):
        if self.phase!="ready":return
        self.p=1 if self.m.probe(T(first(self.r)[None])).item()>=0 else -1
        peak=self.w.probe(self.p);self.w.set_clutch(self.gear);self.paint(peak);self.phase="probing"
        x=T(history(self.r,peak,self.p,self.gear)[None]);self.h=self.m.observe(x);self.pp=T([self.p])
        self.z=self.m.representation(self.h,self.pp);self.alt=self.z
        if self.arm.get()=="partial":
            hf=self.m.observe(T(history(self.r,peak,self.p,1-self.gear)[None]))
            donor=self.m.future(hf,self.pp);q=T(read(DEFAULT/"calibration"/f"partial-{C['seeds'][0]}.npz")["Q"])
            self.alt=self.z+((donor-self.z)@q)@q.T
        self.card.set("试探中：把手 "+("A" if self.p<0 else "B"))
        current=self.sid
        def reset():
            if current!=self.sid:return
            self.paint(self.r.base);self.phase="goal";self.preview()
        self.win.after(700,reset)
    def flip(self):
        if self.phase!="goal":return
        self.gidx^=1;self.preview()
    @torch.inference_mode()
    def preview(self):
        if self.phase!="goal":return
        g=EXAM[self.gidx];a=self.arm.get();z=self.alt if self.altered.get() and a=="partial" else self.z
        logit=(self.m.issue(self.h,z,T(g[None])) if a=="partial" else self.m.policy(self.h,self.pp,T(g[None])))
        self.cmd=1 if logit.item()>=0 else -1
        self.card.set("目标  "+"   ".join(f"{i+1}{'↑' if x<0 else '↓'}" for i,x in enumerate(g) if x)+"     预测下一动作："+("A" if self.cmd<0 else "B"))
        self.eye.delete("all");self.eye.create_text(16,18,anchor="nw",fill="white",text="若正式拉刚试探的把手",font=("Microsoft YaHei UI",12))
        if a in SUP:
            y=self.m.eye(z).flatten().tolist();self.eye.create_line(12,132,308,132,fill="#778692")
            for i,val in enumerate(y):
                x=18+i*24;self.eye.create_line(x,132,x,132+float(np.clip(val,-1.7,1.7))*55,fill="#63dbbe",width=9)
            self.eye.create_text(16,210,anchor="nw",fill="#c5d8e3",text="12 个光学测点的位移预期；不是分割图")
        else:self.eye.create_text(16,110,anchor="nw",fill="#dce8ee",text="此方法只输出动作，没有物理预测图。")
        self.note.set("现在可换相反目标，试探不会重做。只有 partial 支持单独更换物理预期；实际离合与布局保持原样。")
    def act(self):
        if self.phase!="goal":return
        end=self.w.stroke(self.cmd);self.paint(end);vel=lane_travel(self.r.base,end)
        won=goal_wins(vel,EXAM[self.gidx]);self.phase="ended"
        self.note.set(("成功" if won else "未完成")+"。相机实测四轨方向："+" ".join("↑" if v<0 else "↓" for v in vel)+"。本轮已结束；按“新布局”重新开始。")
if __name__=="__main__":
    torch.set_num_threads(2);root=tk.Tk();Window(root);root.mainloop()

