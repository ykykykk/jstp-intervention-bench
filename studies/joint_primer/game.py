"""Tide Harbor. The arcade and the sealed paper route share a deck, not a scoring protocol."""
import os
from dataclasses import replace
from queue import Empty, Queue
from threading import Thread
from time import monotonic
import traceback
import tkinter as tk
from tkinter import messagebox

import numpy as np

from lategoal.rig import EXAM
from .game_round import Deal, Dispatch, FreeHarbor, PAPER_EXAMPLE, Sounding
from .game_board import Switchboard, HANDLE, THROW_X, THROW_Y, clip, smooth
from .game_board import NIGHT, PANEL, SEA, GOLD, WHITE, MUTED, RED, FACE
from .game_tools import HarborLog


ARMS = {"完整 JSTP": "joint", "ACT": "act", "Diffusion Policy": "diffusion",
        "V-JEPA 2.1": "vjepa21", "DINO-WM": "dino", "SmolVLA": "smol"}
# The first hull on the published example travels down with A. This one-berth lesson is not scored.
LESSON_GOAL = np.array((1,0,0,0),dtype=np.int8)


def pilot_worker(requests,replies):
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    pilot=key=view=seen=None
    while True:
        req=requests.get()
        while not requests.empty():
            req=requests.get()
        if req is None:
            return
        stamp,spec,observation,card=req
        try:
            if key!=spec:
                if pilot is not None:
                    pilot.close()
                pilot=key=view=seen=None
                from .game_pilots import FrozenPilot
                pilot=FrozenPilot(*spec)
                key=spec
            if seen!=stamp:
                view=pilot.observe(observation)
                seen=stamp
            answer=pilot.predict(view,card)
        except Exception as exc:
            traceback.print_exc()
            replies.put((stamp,card,None,f"{type(exc).__name__}: {exc}"))
        else:
            replies.put((stamp,card,answer,None))


def label(parent,text="",size=14,color=WHITE,bg=NIGHT,bold=False):
    return tk.Label(parent,text=text,bg=bg,fg=color,anchor="w",justify="left",
                    font=(FACE,-size,"bold" if bold else "normal"))


def button(parent,text,action,bg=PANEL,fg=WHITE):
    return tk.Button(parent,text=text,command=action,bg=bg,fg=fg,activebackground=GOLD,
                     activeforeground=NIGHT,relief="flat",borderwidth=0,highlightthickness=0,
                     padx=13,pady=6,cursor="hand2",font=(FACE,-14,"bold"))


class Station:
    def __init__(self,root):
        self.root=root
        root.title("潮汐港 · 四轨货运")
        root.configure(bg=NIGHT)
        w,h=root.winfo_screenwidth(),root.winfo_screenheight()
        root.geometry(f"{min(1080,w-30)}x{min(690,h-78)}")
        root.minsize(min(700,w-34),min(490,h-80))
        self.mode="human"
        self.generation=0
        self.tally,self.attempts={},{}
        self.arm=tk.StringVar(value="完整 JSTP")
        self.budget=tk.IntVar(value=128); self.seed=tk.IntVar(value=9301)
        self.continuous=tk.BooleanVar(value=False)
        self.requests,self.replies,self.worker=Queue(),Queue(),None
        self.clock,self.last,self.timer=0.,monotonic(),None
        self.layout()
        self.log=HarborLog(self)
        self.new("lesson",PAPER_EXAMPLE)
        root.bind("<KeyPress>",self.keypress)
        root.protocol("WM_DELETE_WINDOW",self.close)
        self.timer=root.after(32,self.tick)

    @property
    def lesson(self): return self.route=="lesson"

    @property
    def paper(self): return self.route=="paper"

    def layout(self):
        head=tk.Frame(self.root,bg=NIGHT)
        head.pack(fill="x",padx=18,pady=(11,5))
        label(head,"潮汐港",27,GOLD,bold=True).pack(side="left")
        label(head,"  /  四轨货运",19,bold=True).pack(side="left")
        self.badge=label(head,size=13,color=SEA)
        self.badge.pack(side="left",padx=12)
        self.menu_button=button(head,"☰",self.open_menu)
        self.menu_button.pack(side="right",padx=(5,0))
        self.pause_button=button(head,"Ⅱ",self.pause)
        self.pause_button.pack(side="right",padx=3)
        self.pilot_button=button(head,"▶  看模型玩",self.switch_pilot)
        self.pilot_button.pack(side="right",padx=3)
        foot=tk.Frame(self.root,bg=NIGHT,height=49)
        foot.pack(side="bottom",fill="x",padx=18,pady=(4,9)); foot.pack_propagate(False)
        self.next_button=button(foot,"",self.advance,SEA,NIGHT)
        self.hint=label(foot,size=15,color=WHITE)
        self.hint.pack(side="left",fill="both",expand=True)
        self.board=Switchboard(self.root)
        self.board.pack(fill="both",expand=True,padx=12)
        self.board.bind("<Configure>",lambda _:self.draw() if hasattr(self,"trip") else None)
        self.board.bind("<ButtonPress-1>",self.press)
        self.board.bind("<B1-Motion>",self.move)
        self.board.bind("<ButtonRelease-1>",self.release)
        self.board.bind("<Motion>",self.hover)
        self.board.bind("<FocusOut>",self.interrupt_drag)
        self.board.bind("<Unmap>",self.interrupt_drag)

    def menu(self,parent):
        return tk.Menu(parent,tearoff=False,bg=PANEL,fg=WHITE,activebackground=SEA,
                       activeforeground=NIGHT,relief="flat",font=(FACE,-14))

    def open_menu(self,event=None):
        if hasattr(self,"nav"): self.nav.destroy()
        m=self.menu(self.root)
        self.nav=m
        m.add_command(label="练手码头",command=lambda:self.change_route("lesson"))
        m.add_command(label="自由港　·　两根绞盘都能试",command=lambda:self.change_route("free"))
        m.add_command(label="论文赛道　·　只试一次",command=lambda:self.change_route("paper"))
        m.add_separator()
        m.add_command(label="换新布局",command=self.fresh)
        m.add_command(label="重玩这局",command=lambda:self.new(self.route,self.trip.deal))
        m.add_command(label="同一港口换订单",command=self.remix)
        m.add_command(label="论文例题",command=lambda:self.new("paper",PAPER_EXAMPLE))
        m.add_separator()
        for title,var,vals in (("代驾模型",self.arm,tuple(ARMS)),("示范量",self.budget,(128,1024,16384)),
                               ("训练种子",self.seed,(9301,9419,9547))):
            sub=self.menu(m)
            for v in vals:
                sub.add_radiobutton(label=str(v),value=v,variable=var,command=self.pilot_changed)
            m.add_cascade(label=title,menu=sub)
        m.add_checkbutton(label="观战自动换港",variable=self.continuous)
        m.add_command(label="打开实验仪表　F2",command=self.log.open)
        m.add_command(label="操作图例　F1",command=self.legend)
        if len(self.trip.outcomes)==2 and self.phase=="result":
            m.add_separator()
            m.add_command(label="回看去程",command=lambda:self.replay(0))
            m.add_command(label="回看返航",command=lambda:self.replay(1))
        try:
            m.tk_popup(self.menu_button.winfo_rootx(),self.menu_button.winfo_rooty()+self.menu_button.winfo_height())
        finally:
            m.grab_release()

    def spec(self):
        return ARMS[self.arm.get()],self.budget.get(),self.seed.get()

    def driver(self):
        return ("human",) if self.mode=="human" else self.spec()

    def clear_lever(self):
        self.selected,self.axis,self.drag=None,None,None
        self.echo=self.shot=self.voyage=None
        self.p=0.

    def new(self,route,deal,memories=None):
        self.generation+=1
        self.route=route
        if self.lesson:
            self.mode="human"
        self.trip=Dispatch(deal) if self.paper else FreeHarbor(deal)
        self.order,self.revision=0,0
        self.soundings,self.last_side=({},None) if memories is None else memories
        self.paused=self.replaying=False
        self.error=self.prediction=None
        self.auto_sides=[]
        self.touched=False
        self.last=monotonic()
        self.clear_lever()
        self.board.set_camera(self.trip.rig.base)
        self.board.focus_set()
        self.enter("probe" if self.paper else "deck")
        if self.mode=="watch":
            self.start_watch()

    def switch_pilot(self):
        watching=self.mode=="human"
        memory=(dict(self.soundings),self.last_side) if watching and self.route=="free" else None
        self.mode="watch" if watching else "human"
        self.new("free" if self.lesson else self.route,self.trip.deal,memory)

    def pilot_changed(self):
        if self.mode=="watch":
            memory=(dict(self.soundings),self.last_side) if not self.paper else None
            self.new(self.route,self.trip.deal,memory)
        else:
            self.refresh()

    def change_route(self,route):
        self.new(route,PAPER_EXAMPLE if route=="lesson" or self.lesson else self.trip.deal)

    def fresh(self):
        self.new("free" if self.lesson else self.route,Deal.fresh())

    def remix(self):
        d=self.trip.deal
        self.new("free" if self.lesson else self.route,replace(d,card=(d.card+2)%len(EXAM)))

    def enter(self,phase):
        self.phase,self.elapsed=phase,0.
        self.drag=None
        self.refresh()

    def goal(self):
        if self.lesson:
            return LESSON_GOAL
        if self.paper and self.trip.observation is None:
            return None
        return EXAM[self.trip.card_ids[self.order]]

    def show_gear(self):
        return not self.paper or self.trip.observation is not None

    def arrival(self):
        if self.phase in ("landed","result") or self.phase=="reveal" and self.p>=1:
            return self.trip.outcomes[self.order]
        return None

    def deck_title(self):
        if self.phase=="fault": return "代驾未能出港"
        o=self.arrival()
        if o is not None:
            if self.phase=="result" and not self.replaying:
                passed=sum(v.won for v in self.trip.outcomes)
                return ("两单都没送够","还差一单","两单都送到了！")[passed]
            if self.lesson: return "送到了！" if o.won else "泊位在另一边"
            return "交货成功" if o.won else "这单没送够"
        if self.phase=="waiting": return f"{self.arm.get()} 正在选绞盘"
        if self.phase=="reveal": return "出港" if self.order==0 else "返航"
        if self.phase=="auto_probe": return "绞盘试车"
        if self.paper and self.goal() is None: return "先试绞盘，再开订单"
        if self.paper and self.phase in ("select","auto_pull"): return "选一根绞盘"
        return "把小艇送进青框" if self.lesson else "把货送进青框"

    def track_axes(self,side):
        if self.can_retry_lesson(): return True,True
        if self.phase in ("landed","result","reveal","fault"):
            return False,self.axis=="y" and side==self.selected
        if not self.paper:
            return True,True
        if self.trip.observation is None:
            return side==self.trip.deal.probe,False
        return False,True

    def gesture_cue(self):
        if self.mode!="human" or self.drag or self.p or self.paused:
            return None
        if self.lesson and self.phase=="deck" and not self.touched: return -1,"y"
        if self.paper and self.phase=="probe": return self.trip.deal.probe,"x"
        return None

    def refresh(self):
        self.badge.configure(text={"lesson":"练手码头","free":"自由港","paper":"论文赛道"}[self.route])
        self.pilot_button.configure(text="▶  看模型玩" if self.mode=="human" else "◀  我来玩",
                                    bg=PANEL if self.mode=="human" else "#296255")
        self.pause_button.configure(text="▶" if self.paused else "Ⅱ")
        action=None
        if self.phase=="fault": action="查看原因"
        elif self.phase=="result": action="下一港   →"
        elif self.phase=="landed":
            action=("四舱出港   →" if self.trip.outcomes[0].won else "再试一次") if self.lesson else "接返航单   →"
        elif self.lesson: action="跳过练手   ›"
        if action:
            self.next_button.configure(text=action,bg=PANEL if action.startswith("跳过") else SEA,
                                       fg=MUTED if action.startswith("跳过") else NIGHT)
            if not self.next_button.winfo_manager(): self.next_button.pack(side="right",padx=(8,0))
        else:
            self.next_button.pack_forget()
        self.hint.configure(text=self.caption())
        self.draw()

    def caption(self):
        if self.paused: return "Ⅱ   已暂停"
        if self.phase=="fault": return "可以切回「我来玩」，不会替换成别的模型。"
        if self.phase=="waiting": return "首次上场需要读取本机权重。"
        if self.phase=="result":
            a,b=self.trip.outcomes
            return f"去程  {HANDLE[self.trip.choices[0]]}  {a.hits}/3      ·      返航  {HANDLE[self.trip.choices[1]]}  {b.hits}/3"
        if self.phase=="landed":
            if self.lesson: return "同样的青框，到了四轨也一样。" if self.trip.outcomes[0].won else "换另一根绞盘试试。"
            return "返航从原处出发，三个泊位全部换边。"
        if self.phase=="reveal": return "现在看到的是实测靠泊位置。"
        if self.mode=="watch":
            return ("试车台在横拉绞盘。" if self.phase in ("auto_probe","rewind") else
                    f"{self.arm.get()} 掌舵   ·   同一港口，同一订单")
        if self.p>=1 and self.phase in ("deck","probe","select"):
            return "松手回中。" if self.axis=="x" else "松手锁定绞盘。" if self.paper else "松手靠泊。"
        if self.paper and self.goal() is None: return f"握住 {HANDLE[self.trip.deal.probe]} 横拖到槽口。"
        if self.paper: return "先选好两单，再一起开船。" if self.order==0 else "第一单已锁定。选返航绞盘。"
        if self.lesson and not self.touched: return "任选绞盘：横拖试绳，往下拖送小艇。"
        if self.axis=="y": return "到线松手交货；不满意可以往回拖。"
        if self.axis=="x": return "松手即回中，两根绞盘都能再试。"
        return "横拖试绳  →      下拉送货  ↓      A、B 随时换手"

    def draw(self):
        self.board.draw(self)
        self.log.update()

    def advance(self):
        self.board.focus_set()
        if self.paused: self.pause()
        if self.phase=="fault":
            self.show_error()
        elif self.lesson:
            if self.phase=="landed" and not self.trip.outcomes[0].won:
                self.new("lesson",PAPER_EXAMPLE)
            else:
                self.new("free",PAPER_EXAMPLE)
        elif self.phase=="landed":
            self.order=1
            self.clear_lever()
            self.enter("deck")
            if self.mode=="watch": self.request_choice()
        elif self.phase=="result": self.fresh()

    def start_watch(self):
        if not self.paper and self.last_side is not None:
            self.request_choice()
            return
        # The station supplies the probes, not the action policy. Free demonstrations show both.
        side=self.trip.deal.probe
        self.auto_sides=[side] if self.paper else [-side,side]
        self.auto_probe()

    def auto_probe(self):
        self.clear_lever()
        self.selected=self.auto_sides.pop(0)
        self.arm_axis("x")
        self.enter("auto_probe")

    def request_choice(self):
        self.prediction=None
        self.enter("waiting")
        if self.worker is None:
            self.worker=Thread(target=pilot_worker,args=(self.requests,self.replies),daemon=True)
            self.worker.start()
        obs=self.soundings[self.last_side].observation
        self.requests.put(((self.generation,self.revision),self.spec(),obs.copy(),self.trip.card_ids[self.order]))

    def arm_axis(self,axis):
        self.axis=axis
        if axis=="x":
            if self.paper:
                self.trip.begin_probe()
                self.echo=self.trip.trace
            else:
                self.shot=self.trip.sound(self.selected)
                self.echo=self.shot.trace
        elif not self.paper:
            self.voyage=self.trip.voyage(self.selected)

    def remember(self,sounding):
        self.last_side=sounding.side
        self.soundings[sounding.side]=sounding
        self.revision+=1

    def rewind(self,commit):
        self.complete_probe=commit
        self.from_p=self.p
        if commit and not self.paper:
            self.remember(self.shot)
        self.enter("rewind")

    def finish_rewind(self):
        if self.paper and self.complete_probe:
            self.trip.finish_probe()
            self.remember(Sounding(self.selected,self.trip.peak,self.trip.trace,self.trip.observation))
        self.clear_lever()
        if self.mode=="watch":
            if self.auto_sides: self.auto_probe()
            else: self.request_choice()
        else:
            self.enter(("select" if self.trip.observation is not None else "probe") if self.paper else "deck")

    def submit(self):
        if self.phase not in ("deck","probe","select","auto_probe","auto_pull"): return
        if self.axis=="x":
            self.rewind(self.p>=1)
        elif self.axis=="y" and self.p>=1:
            if self.paper:
                self.trip.seal(self.selected)
                if len(self.trip.choices)==1:
                    self.order=1; self.clear_lever()
                    if self.mode=="watch": self.request_choice()
                    else: self.enter("select")
                else: self.reveal(0)
            else:
                self.trip.land(self.selected,self.voyage,self.goal())
                if self.lesson or self.order==0: self.enter("landed")
                else: self.finish_pair()

    def reveal(self,index):
        self.order=index
        self.clear_lever()
        self.axis,self.selected,self.voyage="y",self.trip.choices[index],self.trip.outcomes[index]
        self.enter("reveal")

    def finish_pair(self):
        key=(self.route,self.driver()); wins,plays=self.tally.get(key,(0,0))
        self.tally[key]=wins+int(all(o.won for o in self.trip.outcomes)),plays+1
        self.attempts[self.route,self.trip.deal,self.driver()]="   /   ".join(
            f"{HANDLE[s]} {v.hits}/3" for s,v in zip(self.trip.choices,self.trip.outcomes,strict=True))
        self.enter("result")

    def replay(self,index):
        if self.phase=="result":
            self.replaying=True; self.paused=False
            self.reveal(index)

    def can_retry_lesson(self):
        return self.lesson and self.phase=="landed" and not self.trip.outcomes[0].won

    def interactive(self):
        return self.mode=="human" and not self.paused and (self.phase in ("deck","probe","select","rewind") or self.can_retry_lesson())

    def choose(self,side):
        if not self.interactive(): return False
        if self.can_retry_lesson():
            self.new("lesson",PAPER_EXAMPLE,(dict(self.soundings),self.last_side))
        if self.phase=="rewind": self.finish_rewind()
        if not any(self.track_axes(side)): return False
        if side!=self.selected:
            self.clear_lever()
        self.selected=side
        self.draw()
        return True

    def press(self,event):
        self.board.focus_set()
        if not self.interactive(): return
        point=self.board.point(event)
        side=self.board.pick(point,self.phase=="rewind" or self.can_retry_lesson())
        if side is not None and self.choose(side):
            self.drag=(point,self.p)
            self.board.configure(cursor="fleur")

    def set_pull(self,value):
        old=(self.touched,self.p>=1)
        self.p=1. if value>=.97 else clip(value)
        if self.p>0 and not self.touched:
            self.touched=True
        if old!=(self.touched,self.p>=1):
            self.hint.configure(text=self.caption())
        self.draw()

    def move(self,event):
        if self.drag is None: return
        start,old=self.drag
        x,y=self.board.point(event)
        dx,dy=x-start[0],y-start[1]
        if self.axis is None:
            can_x,can_y=self.track_axes(self.selected)
            vx,vy=dx if can_x else 0.,dy if can_y else 0.
            if max(vx,vy)<5: return
            self.arm_axis("x" if vx>=vy else "y")
            self.refresh()
        self.set_pull(old+(dx/THROW_X if self.axis=="x" else dy/THROW_Y))

    def release(self,event):
        if self.drag is None: return
        self.move(event)
        self.drag=None
        self.board.configure(cursor="")
        if self.axis is not None and self.p==0:
            self.axis=None; self.voyage=self.echo=self.shot=None
        else:
            self.submit()
        self.refresh()

    def interrupt_drag(self,event=None):
        if getattr(self,"drag",None) is None: return
        self.drag=None
        self.board.configure(cursor="")
        if self.axis=="x": self.rewind(False)

    def hover(self,event):
        if self.drag is not None: return
        side=self.board.pick(self.board.point(event),self.phase=="rewind" or self.can_retry_lesson())
        yes=self.interactive() and side is not None and (self.phase=="rewind" or any(self.track_axes(side)))
        self.board.configure(cursor="hand2" if yes else "")

    def keypress(self,event):
        key=event.keysym.lower()
        if key=="f1": self.legend(); return "break"
        if key=="f2": self.log.open(); return "break"
        if key=="p": self.pause(); return "break"
        if self.paused: return
        if key=="return" and (self.phase in ("landed","result","fault") or self.lesson):
            self.advance(); return "break"
        if not self.interactive() or self.drag is not None: return
        if key in ("a","b"):
            self.choose(-1 if key=="a" else 1); return "break"
        if key=="space":
            self.submit(); return "break"
        if key not in ("left","right","up","down"): return
        if self.can_retry_lesson(): self.choose(self.selected)
        if self.phase=="rewind": self.finish_rewind()
        if self.selected is None:
            self.choose(self.trip.deal.probe if self.paper and self.goal() is None else -1)
        axis="x" if key in ("left","right") else "y"
        if not self.track_axes(self.selected)[int(axis=="y")]: return "break"
        if self.axis is None:
            if key in ("left","up"): return "break"
            self.arm_axis(axis)
        if self.axis==axis:
            self.set_pull(self.p+(.25 if key in ("right","down") else -.25))
            if self.p==0: self.axis=None; self.voyage=self.echo=self.shot=None
            self.refresh()
        return "break"

    def poll(self):
        while True:
            try: stamp,card,answer,error=self.replies.get_nowait()
            except Empty: return
            if self.phase!="waiting" or stamp!=(self.generation,self.revision) or card!=self.trip.card_ids[self.order]:
                continue
            self.prediction,self.error=answer,error
            if error:
                self.enter("fault")
            else:
                self.clear_lever()
                self.selected=answer.side
                self.arm_axis("y")
                self.enter("auto_pull")

    def tick(self):
        now=monotonic(); dt=min(.15,now-self.last); self.last=now
        self.poll()
        if not self.paused:
            self.elapsed+=dt; self.clock+=dt
            t=self.elapsed
            if self.phase=="auto_probe":
                self.p=smooth((t-.34)/.83)
                if t>=1.38: self.submit()
            elif self.phase=="rewind":
                self.p=self.from_p*(1-smooth((t-.20)/.47))
                if t>=.73: self.finish_rewind()
            elif self.phase=="auto_pull":
                self.p=smooth((t-.65)/1.08)
                if t>=1.95: self.submit()
            elif self.phase=="reveal":
                self.p=smooth((t-.24)/1.05)
                if t>=2.5:
                    if self.replaying:
                        self.replaying=False; self.enter("result")
                    elif self.order==0: self.reveal(1)
                    else: self.finish_pair()
            elif self.phase=="landed" and self.mode=="watch" and t>=2.8: self.advance()
            elif self.phase=="result" and self.mode=="watch" and self.continuous.get() and t>=4.8: self.fresh()
            # Static docks need only a slow beacon/gesture pulse; grabs and actual runs draw at input speed.
            if self.phase in ("auto_probe","auto_pull","rewind","reveal") or int(self.clock*9)!=int((self.clock-dt)*9):
                self.draw()
        self.timer=self.root.after(32,self.tick)

    def pause(self):
        self.interrupt_drag()
        self.paused=not self.paused; self.last=monotonic()
        self.board.focus_set()
        self.refresh()

    def legend(self):
        self.log.legend()

    def show_error(self):
        messagebox.showerror("代驾没有执行",f"{self.arm.get()}\n\n{self.error}",parent=self.root)

    def close(self):
        if self.timer is not None: self.root.after_cancel(self.timer)
        if self.worker is not None: self.requests.put(None)
        self.root.destroy()


def main():
    root=tk.Tk()
    Station(root)
    root.mainloop()


if __name__=="__main__":
    main()
