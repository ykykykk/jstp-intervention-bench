"""潮汐港 · 双舱救援。独立本地窗口，键盘和鼠标只在窗口内生效。"""
from concurrent.futures import CancelledError
from queue import Empty, SimpleQueue
import secrets
from threading import Event, Thread
from time import monotonic
import tkinter as tk
from tkinter import messagebox

import numpy as np

from .rescue_mission import ORDERS, ROTATION, Sortie
from .rescue_scene import Deck, FACE, GOLD, MUTED, NIGHT, PANEL, RED, SEA, WHITE
from .v7_world import D, TABLE, W


def label(parent, text="", *, size=15, color=WHITE, bg=PANEL, bold=False, wrap=0, var=None):
    return tk.Label(parent, text=text, textvariable=var, fg=color, bg=bg, anchor="w",
                    font=(FACE, -size, "bold" if bold else "normal"), justify="left", wraplength=wrap)


def button(parent, text, action, fill="#254051", ink=WHITE):
    return tk.Button(parent, text=text, command=action, bg=fill, fg=ink,
                     activebackground="#365665", activeforeground=WHITE, disabledforeground="#8198a4",
                     relief="flat", overrelief="ridge", highlightthickness=0, borderwidth=1,
                     cursor="hand2", padx=11, pady=5, font=(FACE, -14, "bold"), takefocus=True)


def movement(y):
    if abs(y) < D / 4:
        return "留在原位"
    if abs(y) < D * .65:
        return "走向不明"
    return "↑ 上行" if y < 0 else "↓ 下行"


class RescueGame:
    def __init__(self, root):
        self.root = root
        root.title("潮汐港 · 双舱救援  |  JSTP")
        root.configure(bg=NIGHT)
        width, height = min(1060, root.winfo_screenwidth()-36), min(715, root.winfo_screenheight()-80)
        root.geometry(f"{width}x{height}"); root.minsize(min(795,width), min(565,height))
        self.mode = "human"
        self.stats = {"human": [0,0,0], "ai": [0,0,0]}  # 成功、出航、连胜，按驾驶者分开
        self.first_seed = 41_000_000_000 + secrets.randbelow(1_000_000_000)
        self.number = 0
        self.timer = self.pending = None
        self.remaining = 0
        self.paused = False
        self.pilot = None; self.loading = False
        self.events = SimpleQueue(); self.closing = Event()
        self.build()
        self.new_order()
        root.bind("<KeyPress>", self.key)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def build(self):
        shell = tk.Frame(self.root, bg=NIGHT, padx=16, pady=12); shell.pack(fill="both",expand=True)
        shell.columnconfigure(0,weight=1); shell.rowconfigure(2,weight=1)
        top=tk.Frame(shell,bg=NIGHT); top.grid(row=0,column=0,sticky="ew",pady=(0,9))
        label(top,"潮汐港  /  双舱救援",size=24,bg=NIGHT,bold=True).pack(side="left")
        self.help_btn=button(top,"玩法 ?",self.help); self.help_btn.pack(side="right",padx=(6,0))
        self.pause_btn=button(top,"暂停  P",self.toggle_pause); self.pause_btn.pack(side="right",padx=4)
        self.modes={}
        for k,title in (("ai","JSTP 代驾"),("human","我来驾驶")):
            b=button(top,title,lambda m=k:self.switch(m)); b.pack(side="right",padx=2); self.modes[k]=b

        order=tk.Frame(shell,bg="#213848",padx=15,pady=8); order.grid(row=1,column=0,sticky="ew",pady=(0,9))
        self.order_no=tk.StringVar(); self.title=tk.StringVar(); self.rule=tk.StringVar()
        label(order,var=self.order_no,color=SEA,bg="#213848",size=12,bold=True).pack(anchor="w")
        label(order,var=self.title,color=GOLD,bg="#213848",size=24,bold=True).pack(anchor="w")
        label(order,var=self.rule,bg="#213848",size=15).pack(anchor="w")

        middle=tk.Frame(shell,bg=NIGHT); middle.grid(row=2,column=0,sticky="nsew")
        middle.columnconfigure(0,weight=1); middle.rowconfigure(0,weight=1)
        self.deck=Deck(middle,self.select_handle,self.manual_pull,self.manual_release)
        self.deck.grid(row=0,column=0,sticky="nsew")
        radio=tk.Frame(middle,bg=PANEL,padx=13,pady=10,width=240)
        radio.grid(row=0,column=1,sticky="ns",padx=(10,0)); radio.grid_propagate(False)
        radio.columnconfigure(0,weight=1)
        self.news=tk.StringVar(); self.story=tk.StringVar(); self.clue=tk.StringVar()
        label(radio,"●  港务电台",color=SEA,size=12,bold=True).grid(row=0,column=0,sticky="w")
        self.headline=label(radio,var=self.news,size=20,bold=True,wrap=212)
        self.headline.grid(row=1,column=0,sticky="w",pady=(5,3))
        label(radio,var=self.story,wrap=211,size=14).grid(row=2,column=0,sticky="w")
        label(radio,var=self.clue,wrap=210,size=13,color=GOLD).grid(row=3,column=0,sticky="w",pady=(9,5))
        tk.Frame(radio,bg="#345061",height=1).grid(row=4,column=0,sticky="ew",pady=(2,6))
        self.notehead=tk.StringVar(); self.forecast=tk.StringVar()
        label(radio,var=self.notehead,size=12,color=SEA,bold=True).grid(row=5,column=0,sticky="w")
        label(radio,var=self.forecast,size=13,color=MUTED,wrap=211).grid(row=6,column=0,sticky="w",pady=3)

        self.bar=tk.Frame(shell,bg=NIGHT); self.bar.grid(row=3,column=0,sticky="ew",pady=(8,4))
        for i in range(2): self.bar.columnconfigure(i,weight=1,uniform="levers")
        self.controls=tk.StringVar()
        self.control_strip=label(self.bar,var=self.controls,bg=NIGHT,size=14,bold=True)
        self.more=button(self.bar,"下一关   →   [Enter]",self.next_if_done,SEA,NIGHT)
        self.remix=button(self.bar,"沿用刚才的试探，换张工单",self.remix_if_done)
        self.tally=tk.StringVar()
        label(shell,var=self.tally,bg=NIGHT,color=MUTED,size=13).grid(row=4,column=0,sticky="w",padx=3)

    def say(self, title, text, color=WHITE):
        self.news.set(title); self.story.set(text); self.headline.configure(fg=color)

    def new_order(self):
        self.cancel_timer(); self.paused=False; self.number+=1; self.remixed=False
        k=ROTATION[(self.number-1)%len(ROTATION)]
        self.r=Sortie(self.first_seed+self.number, secrets.randbelow(2), k, secrets.randbelow(2))
        self.phase="ready"; self.active=None; self.prediction=None
        self.clue.set("试拉机会：还剩 1 次")
        self.prompt(); self.render(); self.arm_pilot()

    def prompt(self):
        if self.phase=="post":
            clue=("这单电池要留。哪根只牵乘员？", "这单电池也要走。哪根能带两舱？",
                  "这次电池走留不限。把乘员送进青框就行。")
            title=f"已试 {'AB'[self.r.scouted]}，现在可选 A 或 B" if self.mode=="human" else "选一根出航"
            self.say(title,clue[self.r.order]+(" 握住选好的绞盘，沿竖轨拖到青线再松手。" if self.mode=="human" else ""))
        elif self.mode=="human":
            self.say("先握住一根绞盘","按住甲板下方的 A 或 B，向旁边拖一点，看绿电池跟不跟。也可以不试，直接沿竖轨拖向交付线。")
        else:
            self.say("JSTP 接管绞盘","这关由代驾决定要不要试、试哪根，再决定拉哪根出航。")

    def render(self):
        r=self.r
        self.order_no.set(f"港务工单  /  第 {self.number:02d} 关"+("  ·  同一现场的追加工单" if self.remixed else ""))
        self.title.set(ORDERS[r.order].title)
        self.rule.set(("↑ 向上救援　" if r.dy<0 else "↓ 向下救援　")+ORDERS[r.order].instruction)
        self.deck.show(r,self.active,self.paused,self.manual_handles(),self.phase)
        self.pause_btn.configure(text="继续  P" if self.paused else "暂停  P")
        for key,b in self.modes.items():
            b.configure(bg=SEA if key==self.mode else "#254051",fg=NIGHT if key==self.mode else WHITE,
                        state="normal" if self.phase in ("ready","result","fault") else "disabled")
        if self.phase=="result":
            self.control_strip.grid_remove()
            self.more.grid(row=0,column=0,columnspan=1 if self.can_remix() else 2,sticky="ew",padx=3,ipady=2)
            self.more.configure(state="disabled" if self.paused else "normal")
            if self.can_remix():
                self.remix.grid(row=0,column=1,sticky="ew",padx=3,ipady=2)
                self.remix.configure(state="disabled" if self.paused else "normal")
            else: self.remix.grid_remove()
        else:
            self.more.grid_remove(); self.remix.grid_remove()
            self.controls.set("鼠标：按住绞盘拖动，松手停住　｜　键盘：A / B 选绞盘，方向键逐格拉，空格松手"
                              if self.mode=="human" else "JSTP 正在驾驶　｜　左边是实际操作，右边是它行动前的预想")
            self.control_strip.grid(row=0,column=0,columnspan=2,sticky="ew",padx=3,pady=5)
        if self.mode=="human":
            self.notehead.set("驾驶手册")
            self.forecast.set("两根都拉得动乘员，只有一根也会带动电池。横拉松手会回位；竖拉中途松手会停住。")
        else:
            self.notehead.set("JSTP 预想  /  不是现场")
            if self.prediction is None:
                self.forecast.set("代驾决策时，这里会显示两根绞盘各自可能造成的后果。")
            else:
                seen,rows=self.prediction
                self.forecast.set(("依据试拉后的经历" if seen else "尚未试拉的估计")+"：\n"+
                    "\n".join(f"拉 {key}：乘员{movement(a)} · 电池{movement(b)}" for key,(a,b) in zip("AB",rows)))
        h,a=self.stats["human"],self.stats["ai"]
        self.tally.set(f"本局送达　你 {h[0]}/{h[1]}　·　JSTP {a[0]}/{a[1]}　　 当前驾驶者连胜 {self.stats[self.mode][2]}　｜　现场新关，不计入论文成绩")

    def manual_handles(self):
        """哪些绞盘现在可以接手；横向试拉的次数由工单另外记录。"""
        if self.mode!="human" or self.paused: return ()
        if self.phase in ("ready","post","hand_probe","rewind"): return (0,1)
        if self.phase in ("hand_haul","moored"):
            return (self.active,) if self.r.offset(self.active)[1] else (0,1)
        return ()

    def select_handle(self, handle):
        if handle not in self.manual_handles(): return
        self.deck.drop_pointer()
        if self.phase in ("hand_probe","rewind"):
            # 新输入接手：先让被试的绞盘真正回位并松开，再让新绞盘接受这次拖动。
            # 不能留下旧的复位回调，否则它稍后会把新选择清掉。
            self.cancel_timer()
            if self.phase=="hand_probe": self.note_probe()
            offset=self.r.offset(self.r.scouted)
            if np.any(offset): self.r.step(-offset)
            self.release_probe_at_berth()
        elif self.phase=="hand_haul" and handle!=self.active:
            self.manual_release(commit=False)  # 此处能换手，说明已经拉回起点
        if self.phase=="moored":
            self.r.press(handle); self.phase="hand_haul"
        self.active=handle
        self.deck.focus_set()
        if self.phase in ("ready","post"):
            arrow="上" if self.r.dy<0 else "下"
            self.say("已选绞盘 "+"AB"[handle],
                     ("左右拖动可以试拉；" if self.r.scouted is None else f"{'AB'[self.r.scouted]} 的试拉记录已保留。")+
                     f"向{arrow}拖动开始送舱。点选本身不会移动。")
        self.render()

    def manual_pull(self, x, y):
        """x/y 是相对泊位的位置（单位：轨道格），不是要播放的动作编号。"""
        handle=self.active
        if handle not in self.manual_handles() or self.phase=="rewind": return
        r=self.r
        direction=1 if r.dy>0 else -1
        if self.phase in ("ready","post"):
            if max(abs(x),abs(y)) < 1: return  # 点选或手抖不消耗试拉，也不锁定出航
            if r.scouted is None and abs(x)>abs(y):
                phase="hand_probe"
            else:
                # 试过之后两个绞盘都只走竖轨；斜着拖也取竖向分量。
                if y*direction<1: return
                phase="hand_haul"
        else:
            phase="hand_probe" if self.phase=="hand_probe" else "hand_haul"
        if phase=="hand_probe":
            target=(max(-D,min(D,round(x))),0)
        else:
            target=(0,direction*max(0,min(abs(r.dy),round(y*direction))))
        delta=np.subtract(target,r.offset(handle))
        if not np.any(delta): return
        if self.phase in ("ready","post","moored"):
            r.press(handle)
        self.phase=phase
        r.step(delta)
        key="AB"[handle]
        if phase=="hand_probe":
            # 记住真正到过的最远位置，即使玩家随后自己把绞盘拉回中心。
            if abs(target[0])>abs(r.past[2]): r.remember_peak(handle)
            self.clue.set("本单的试拉已使用")
            self.say(f"正在横拉 {key}","看绿电池跟不跟。松手收绳，或直接按 A / B 选出航绞盘。",GOLD)
        else:
            if r.scouted is None: self.clue.set("本单跳过了试拉")
            at_line=target[1]==r.dy
            self.say("已到交付线" if at_line else f"牵引 {key}　{abs(target[1])} / {abs(r.dy)} 格",
                     "松手交货；还没松手就可以回拉。" if at_line else "继续拖向青线。中途松手会停在这里，再握同一根就能接着走。",GOLD)
        self.render()

    def manual_release(self, *, commit=True):
        if self.mode!="human" or self.paused or self.closing.is_set(): return
        if self.phase=="hand_probe":
            self.note_probe(); self.rewind()
        elif self.phase=="hand_haul":
            distance=int(self.r.offset(self.active)[1])
            if distance==self.r.dy and commit:
                self.landed()
            else:
                self.r.world.release()
                if distance==0:
                    self.active=None; self.phase="post" if self.r.scouted is not None else "ready"
                    if self.r.scouted is None: self.clue.set("试拉机会：还剩 1 次")
                    self.prompt()
                else:
                    self.phase="moored"
                    if distance==self.r.dy:
                        self.say("停在交付线，尚未交货","重新握住绞盘 "+"AB"[self.active]+" 再松手，确认交货。")
                    else:
                        self.say(f"中途停靠　{abs(distance)} / {abs(self.r.dy)} 格",
                                 "握住绞盘 "+"AB"[self.active]+" 继续拖；想换绞盘，可以先拖回起点。")
                self.render()

    def pilot_action(self, k):
        self.active=k%2; self.r.press(self.active)
        self.phase="probe" if k>=2 else "sailing"
        key="AB"[self.active]
        self.say("正在轻试 "+key if k>=2 else "绞盘 "+key+" 出航",
                 "这是实际横拉。看电池有没有跟过来。" if k>=2 else "看甲板上的两艘舱，松手后按青框验收。",GOLD)
        self.render()
        delta=self.r.actions[k,2:]
        self.later(180,lambda:self.travel(np.sign(delta),int(abs(delta).sum()),
                   self.at_peak if k>=2 else self.landed))

    def travel(self, unit, left, on_arrival):
        self.r.step(unit); self.deck.paint()
        if left==1: on_arrival()
        else: self.later(110,lambda:self.travel(unit,left-1,on_arrival))

    def note_probe(self):
        r=self.r
        moved=not np.array_equal(r.peak[:TABLE,W//2:],r.start[:TABLE,W//2:])
        self.clue.set("试拉记录："+ "AB"[r.scouted]+(" 带动了电池" if moved else " 没带动电池"))
        return moved

    def at_peak(self):
        self.r.remember_peak(self.active)
        moved=self.note_probe()
        self.say("看右侧电池舱","它跟着绞盘一起横移了。" if moved else "它留在原地，只有乘员舱横移了。",GOLD)
        self.render(); self.later(920,self.rewind)

    def rewind(self):
        self.phase="rewind"; self.say("绞盘正在复位","刚才只是试拉，还没有交货。"); self.render()
        delta=self.r.offset(self.r.scouted)
        left=int(abs(delta).sum())
        if left: self.travel(-np.sign(delta),left,self.back_at_berth)
        else: self.back_at_berth()

    def release_probe_at_berth(self):
        self.r.finish_probe(); self.active=None; self.phase="post"
        self.prediction=None

    def back_at_berth(self):
        self.release_probe_at_berth()
        self.prompt(); self.render(); self.arm_pilot()

    def landed(self):
        r=self.r; won=r.finish_delivery(self.active); self.active=None; self.phase="result"
        score=self.stats[self.mode]; score[0]+=int(won); score[1]+=1; score[2]=score[2]+1 if won else 0
        end=r.world.camera()[:TABLE]
        crew=np.array_equal(end[:,:W//2],r.goals[0][:,:W//2])
        battery_moved=not np.array_equal(end[:,W//2:],r.start[:TABLE,W//2:])
        why=("乘员已进救援区；" if crew else "乘员没到救援区；")+("电池也已送达。" if battery_moved else "电池留在原泊位。")
        if not won: why+="本单要求电池"+ORDERS[r.order].battery+"。"
        self.say("救援成功" if won else "工单未完成",why,SEA if won else RED)
        if r.scouted is None: self.clue.set("本单直接出航，没有试拉。")
        self.render()
        if self.mode=="ai": self.later(5200,self.new_order)

    def can_remix(self):
        return self.phase=="result" and self.r.scouted is not None and self.r.order!=2 and not self.remixed

    def remix_if_done(self):
        if self.paused or not self.can_remix(): return
        self.cancel_timer(); self.r=self.r.other_order(); self.remixed=True; self.phase="post"; self.active=None
        self.prediction=None
        self.say("同一条线索，新任务","从刚才试完的同一泊位出发。接线没换，只换了电池要走还是留。")
        self.render(); self.arm_pilot()

    def next_if_done(self):
        if self.phase=="result" and not self.paused: self.new_order()

    def switch(self, mode):
        if self.phase not in ("ready","result","fault") or self.mode==mode: return
        self.cancel_timer(); self.deck.drop_pointer(); self.active=None
        self.mode=mode; self.prediction=None
        if self.phase=="fault":
            self.new_order(); return
        if self.phase=="ready": self.prompt()
        self.render()
        if self.phase=="result" and mode=="ai": self.later(1700,self.new_order)
        else: self.arm_pilot()

    def arm_pilot(self):
        if self.mode!="ai" or self.phase not in ("ready","post"): return
        if self.pilot is not None:
            self.later(950,self.plan); return
        self.say("代驾准备中","首次使用会在本机学习试航经验；之后直接载入。你也可以切回「我来驾驶」。")
        if self.loading: return
        self.loading=True

        def report(text):
            if self.closing.is_set(): raise CancelledError()
            self.events.put(("progress",text))

        def prepare():
            try:
                from .rescue_pilot import Pilot
                report("载入冻结 V7")
                self.events.put(("ready",Pilot.prepare(report)))
            except CancelledError:
                return
            except Exception as exc:
                # 将后台初始化错误交还窗口，不默默换成随机或规则代驾。
                self.events.put(("error",f"{type(exc).__name__}: {exc}"))
        Thread(target=prepare,name="rescue-pilot").start()
        self.root.after(70,self.poll_pilot)

    def poll_pilot(self):
        try:
            while True:
                kind,data=self.events.get_nowait()
                if kind=="progress":
                    if self.mode=="ai": self.story.set(data+"。不修改 V7 或论文数据。")
                elif kind=="ready":
                    self.pilot=data; self.loading=False
                    if self.mode=="ai" and self.phase in ("ready","post"): self.prompt(); self.arm_pilot()
                else:
                    self.loading=False
                    if self.mode=="ai": self.pilot_failed(data)
        except Empty:
            pass
        if self.loading: self.root.after(130,self.poll_pilot)

    def plan(self):
        if self.mode!="ai" or self.phase not in ("ready","post"): return
        r=self.r
        try:
            k,rows=self.pilot.decide(r.start,r.peak,r.past,r.actions,r.order,int(r.hand[0]))
        except (RuntimeError,ValueError) as exc:
            self.pilot_failed(str(exc)); return
        self.prediction=(r.scouted is not None,rows)
        self.say("JSTP 选择"+("轻试 " if k>=2 else "出航 ")+"AB"[k%2],
                 "右栏是两个假设。稍后只在左侧甲板执行这一手。",SEA)
        self.render(); self.later(1400,lambda:self.pilot_action(k))

    def pilot_failed(self, reason):
        self.phase="fault"; self.say("代驾未就绪",reason,RED)
        self.clue.set("这单不计分。点「我来驾驶」可开下一关。"); self.render()

    # 所有动作只占用一个时钟。换驾驶、换工单或暂停不会留下晚到的出航回调。
    def cancel_timer(self):
        if self.timer is not None: self.root.after_cancel(self.timer)
        self.timer=self.pending=None

    def later(self, ms, fn):
        self.cancel_timer(); self.pending=fn; self.remaining=ms
        if not self.paused: self.resume_timer()

    def resume_timer(self):
        if self.pending is None: return
        self.due=monotonic()+self.remaining/1000
        def fire():
            fn=self.pending; self.timer=self.pending=None; fn()
        self.timer=self.root.after(self.remaining,fire)

    def toggle_pause(self):
        if not self.paused:
            self.deck.drop_pointer()
            self.manual_release(commit=False)
        self.paused=not self.paused
        if self.paused and self.timer is not None:
            self.remaining=max(1,round(1000*(self.due-monotonic())))
            self.root.after_cancel(self.timer); self.timer=None
        elif not self.paused: self.resume_timer()
        self.render()

    def help(self):
        was_paused=self.paused
        if not was_paused: self.toggle_pause()
        messagebox.showinfo("港务手册 · 三步出航",
            "① 看工单：乘员总要进青框，电池按本关要求走或留。\n\n"
            "② 试拉：按住 A 或 B，向旁边拖一点，看绿电池跟不跟。每单只试一次。试了 B 仍能马上握 A 出航，不用等 B 收绳；键盘也可直接按 A 换手。\n\n"
            "③ 交货：按住选好的绞盘，沿竖轨拖向青色交付线，到线后松手才结算。中途松手会停住，握同一根可继续；退回起点后可改选另一根。\n\n"
            "键盘：A / B 选绞盘，← → 横拉，↑ ↓ 竖拉，空格松手。Enter 下一关，P 暂停。选「JSTP 代驾」才会自动驾驶。", parent=self.root)
        if not was_paused: self.toggle_pause()

    def key(self,event):
        key=event.keysym.lower()
        if key in ("a","b"):
            if not self.manual_handles(): return
            self.select_handle("ab".index(key))
        elif key in ("left","right","up","down","space"):
            if not self.manual_handles() or self.deck.drag is not None: return
            if self.active is None: return
            if key=="space":
                self.manual_release()
            else:
                dx,dy={"left":(-1,0),"right":(1,0),"up":(0,-1),"down":(0,1)}[key]
                x,y=self.r.offset(self.active)
                self.manual_pull(int(x)+dx,int(y)+dy)
        elif key=="p": self.toggle_pause()
        elif key=="return" and self.phase=="result": self.next_if_done()
        elif key=="f1": self.help()
        else: return
        return "break"

    def close(self):
        self.closing.set(); self.cancel_timer(); self.root.destroy()


def main():
    root=tk.Tk(); RescueGame(root); root.mainloop()


if __name__=="__main__": main()
