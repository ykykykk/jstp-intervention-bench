"""航站甲板的显示皮肤；只画可见位置和公开工单，不改 V7 相机输入。"""
import tkinter as tk

from .v7_world import D

NIGHT = "#0c1825"
PANEL = "#142737"
SEA = "#69e2cf"
GOLD = "#ffcb79"
WHITE = "#eaf4f2"
MUTED = "#a3bac4"
RED = "#ff9d85"
FACE = "Microsoft YaHei UI"
LEVER_STEP = 10  # 每个世界位移格在绞盘轨道上占用的设计像素；绘制和拖动共用。


class Deck(tk.Canvas):
    def __init__(self, master, on_grip, on_pull, on_release):
        super().__init__(master, background=NIGHT, highlightthickness=0, width=625, height=350)
        self.round = None
        self.on_grip, self.on_pull, self.on_release = on_grip, on_pull, on_release
        self.handles = ()
        self.drag = None
        self.hits = []
        self.bind("<Configure>", lambda _: self.paint())
        self.bind("<ButtonPress-1>", self.grip)
        self.bind("<B1-Motion>", self.pull)
        self.bind("<ButtonRelease-1>", self.release)
        self.bind("<Motion>", self.hover)
        self.bind("<FocusOut>", self.interrupt)
        self.bind("<Unmap>", self.interrupt)

    def hit(self, event):
        for i, (x1, y1, x2, y2) in enumerate(self.hits):
            if i in self.handles and x1 <= event.x <= x2 and y1 <= event.y <= y2:
                return i
        return None

    def hover(self, event):
        if self.drag is None:
            self.configure(cursor="hand2" if self.hit(event) is not None else "")

    def grip(self, event):
        handle = self.hit(event)
        if handle is None:
            return
        self.on_grip(handle)
        bx, by = self.round.offset(handle)
        self.drag = event.x, event.y, self.scale * LEVER_STEP, int(bx), int(by)
        self.configure(cursor="fleur")

    def pull(self, event):
        if self.drag is not None:
            x, y, pitch, bx, by = self.drag
            self.on_pull(bx + (event.x-x)/pitch, by + (event.y-y)/pitch)

    def release(self, event):
        if self.drag is not None:
            # 快速拖动时 Tk 可能合并 motion 事件；松手位置也是一次真实输入。
            self.pull(event)
            self.drop_pointer()
            self.on_release()
            self.hover(event)

    def drop_pointer(self):
        self.drag = None
        self.configure(cursor="")

    def interrupt(self, _):
        # 窗口失焦或最小化不能只丢掉鼠标坐标，让模拟器还握着旧绞盘。
        # 这也不是玩家在交付线上主动松手，不计作交货。
        self.drop_pointer()
        self.on_release(commit=False)

    def show(self, sortie, active, paused, handles, phase):
        self.round, self.active, self.paused, self.handles, self.phase = sortie, active, paused, handles, phase
        self.paint()

    def paint(self):
        r = self.round
        w, h = self.winfo_width(), self.winfo_height()
        if r is None or min(w, h) < 10:
            return
        self.delete("all"); self.hits.clear()
        s = self.scale = min(w / 660, h / 370)
        ox, oy = (w - 660*s)/2, (h - 370*s)/2

        def xy(*points):
            return tuple((ox if n % 2 == 0 else oy) + p*s for n, p in enumerate(points))

        def txt(x, y, label, size=13, color=MUTED, bold=False, anchor="center"):
            self.create_text(*xy(x, y), text=label, fill=color, anchor=anchor,
                             font=(FACE, -max(9, round(size*s)), "bold" if bold else "normal"))

        def line(*points, color="#345062", width=1, **kwargs):
            self.create_line(*xy(*points), fill=color, width=max(1, width*s), **kwargs)

        def rect(l, t, rr, b, fill="", outline="", width=1):
            self.create_rectangle(*xy(l,t,rr,b), fill=fill, outline=outline, width=max(1,width*s))

        def poly(*points, fill, outline=""):
            self.create_polygon(*xy(*points), fill=fill, outline=outline, width=max(1,s))

        def circle(x, y, radius, fill):
            self.create_oval(*xy(x-radius,y-radius,x+radius,y+radius), fill=fill, outline="")

        rect(3, 4, 657, 367, "#102232", "#294252")
        rect(4, 4, 656, 35, "#162c3c")
        circle(22, 20, 4, SEA)
        txt(34, 19, "现场  /  救援甲板", 13, WHITE, True, "w")
        txt(635, 19, "青色角框 = 本关目标位", 12, SEA, anchor="e")
        line(330, 47, 330, 240, color="#294252", dash=(3,6))
        home_y = 201 if r.dy < 0 else 119
        away_y = home_y + (85 if r.dy > 0 else -85)
        for i, (name, color) in enumerate((("乘员舱", GOLD), ("电池舱", SEA))):
            bx, by = r.board.pieces[i]
            px, py = r.world.p[i]
            x = (172, 488)[i] + (int(bx) - (25,72)[i])*1.5
            cx, cy = x + (int(px)-int(bx))*9, home_y + (int(py)-int(by))*85/D
            txt(x, 57, ("01   " if i==0 else "02   ") + name, 17, color, True)
            for rail in (-53,53):
                line(x+rail, 90, x+rail, 229, color="#274357", width=2)
                for tick in range(99,229,15):
                    line(x+rail-2, tick, x+rail+2, tick, color="#497080")
            for y in (home_y,away_y):
                line(x-46, y, x+46, y, color="#405966", dash=(2,4))
            if i == 0 or r.order != 2:
                dest = home_y if i == 1 and r.order == 0 else away_y
                for sx in (-1,1):
                    for sy in (-1,1):
                        ex, ey = x+sx*49, dest+sy*21
                        line(ex-sx*12, ey, ex, ey, ex, ey-sy*9, color=SEA, width=2)
                txt(x, dest-33, "留守泊位" if dest==home_y else "救援区", 12, SEA, True)
            else:
                txt(x, away_y-29, "位置不限", 12)
            if (cx,cy)!=(x,home_y):
                line(x,home_y,cx,cy,color="#8aa9af",width=2,dash=(3,4))
            # 不依赖颜色区分舱体：乘员是双舷窗，电池是带端子的闪电方舱。
            if i == 0:
                poly(cx-37,cy+6,cx-33,cy-8,cx-20,cy-15,cx+15,cy-15,cx+37,cy-3,
                     cx+40,cy+4,cx+30,cy+13,cx-25,cy+13,fill="#d99959",outline="#ffdf9c")
                for xx in (-16,6):
                    rect(cx+xx-7,cy-9,cx+xx+8,cy+3,"#172f41","#7a563d")
                    circle(cx+xx,cy-3,3,"#ffde9e")
                line(cx-23,cy+8,cx+26,cy+8,color="#ffe0a1",width=2)
            else:
                rect(cx-36,cy-10,cx-31,cy+9,"#9bd1c5")
                rect(cx+31,cy-10,cx+36,cy+9,"#9bd1c5")
                rect(cx-31,cy-15,cx+31,cy+15,"#286661","#91e8cf",2)
                for xx in (-22,-16,17,23):
                    line(cx+xx,cy-10,cx+xx,cy+10,color="#70c5af",width=2)
                poly(cx+4,cy-11,cx-7,cy+2,cx-1,cy+2,cx-4,cy+11,cx+8,cy-3,cx+2,cy-3,fill="#e5ffb7")
        line(18,248,642,248,color="#365463")
        txt(22,263,"牵引台",12,MUTED,True,"w")
        origin_y = 326 if r.dy < 0 else 266
        stop_y = origin_y + r.dy*LEVER_STEP
        for i, key in enumerate("AB"):
            center = (172,488)[i]
            dx, dy = r.offset(i)
            x, y = center + int(dx)*LEVER_STEP, origin_y + int(dy)*LEVER_STEP
            selected = self.active == i
            if self.phase=="ready" or (self.phase in ("hand_probe","probe","rewind") and selected):
                line(center-D*LEVER_STEP,origin_y,center+D*LEVER_STEP,origin_y,
                     color="#62858c",width=2,arrow="both")
            line(center,origin_y,center,stop_y,color="#5c9a9b",width=3,arrow="last")
            for j in range(1,D):
                yy = origin_y + j*LEVER_STEP*(1 if r.dy>0 else -1)
                line(center-4,yy,center+4,yy,color="#80a9a9")
            line(center-38,stop_y,center+38,stop_y,color=SEA,width=2,dash=(3,3))
            txt(center+45,stop_y,"交付线",12,SEA,anchor="w")
            outline=GOLD if selected else SEA if r.scouted is not None and i in self.handles else "#6d909e"
            rect(x-37,y-13,x+37,y+13,"#624b32" if selected else "#213a4a",outline,2)
            txt(x,y, key + "  绞盘",14,GOLD if selected else WHITE,True)
            self.hits.append(xy(x-41,y-16,x+41,y+16))
            if not self.handles and not self.paused and abs(int(r.hand[0])-int(r.world.g[i,0])) <= 4:
                poly(x-5,y+18,x+5,y+18,x,y+13,fill=GOLD)
        if self.handles and self.phase in ("hand_probe","rewind","post"):
            hint=f"已试 {'AB'[r.scouted]}　｜　A、B 都可握，沿竖轨送舱"
        elif self.handles and self.phase in ("hand_haul","moored"):
            hint="沿竖轨送舱，退回起点可换绞盘　｜　到交付线松手才结算"
        else:
            hint="横轨：试拉，看电池跟不跟　｜　竖轨：把乘员送进青框"
        txt(330,352,hint,13)
        if self.paused:
            rect(244,140,416,192,"#182c3b",GOLD,2)
            txt(330,166,"航站暂停  Ⅱ",19,GOLD,True)
