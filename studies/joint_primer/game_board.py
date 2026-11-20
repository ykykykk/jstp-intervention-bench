"""The harbor skin: solid cabins are physical, illuminated berths are destinations."""
from math import sin
import tkinter as tk

import numpy as np

from lategoal.rig import brightness
from jstp.rescue_scene import NIGHT, PANEL, SEA, GOLD, WHITE, MUTED, RED, FACE


BLUE, RAIL, STEEL, WATER = "#91bdf4", "#34596a", "#73949e", "#163747"
HANDLE = {-1: "A", 1: "B"}
PAINT = {-1: GOLD, 1: BLUE}
THROW_X, THROW_Y = 82, 63
HOME = {-1: (235, 359), 1: (644, 359)}
LANES = (143, 347, 551, 755)
LIFT = 60


def clip(v):
    return max(0., min(1., v))


def smooth(v):
    v = clip(v)
    return v*v*(3-2*v)


class Switchboard(tk.Canvas):
    W, H = 900, 460

    def __init__(self, parent):
        super().__init__(parent, bg=NIGHT, highlightthickness=0, takefocus=True)
        self.s, self.ox, self.oy = 1., 0., 0.
        self.knobs = dict(HOME)
        self.base = None
        self.backing = None
        self.layer = "live"

    def set_camera(self, frame):
        # These small layout variations come from the public starting image, not hidden links.
        light = brightness(frame)[:44].reshape(44, 4, 24)
        mass = light.sum((0, 2))
        self.base = ((light.sum(0) @ np.arange(24))/mass, (np.arange(44) @ light.sum(2))/mass)

    def point(self, event):
        return (event.x-self.ox)/self.s, (event.y-self.oy)/self.s

    def pick(self, point, returning=False):
        x, y = point
        for side, pos in self.knobs.items():
            for kx, ky in ((pos, HOME[side]) if returning else (pos,)):
                if abs(x-kx)<=51 and abs(y-ky)<=25:
                    return side
        return None

    def xy(self, pts):
        return tuple((self.oy if i%2 else self.ox)+v*self.s for i,v in enumerate(pts))

    def line(self, *pts, ink=RAIL, w=1, dash=None):
        self.create_line(*self.xy(pts), fill=ink, width=max(1,w*self.s), dash=dash, tags=self.layer)

    def rect(self, l,t,r,b, fill="", edge="", w=1):
        self.create_rectangle(*self.xy((l,t,r,b)), fill=fill, outline=edge, width=max(1,w*self.s), tags=self.layer)

    def poly(self, *pts, fill, edge="", w=1):
        self.create_polygon(*self.xy(pts), fill=fill, outline=edge, width=max(1,w*self.s), tags=self.layer)

    def oval(self,x,y,rx,ry,fill="",edge="",w=1):
        self.create_oval(*self.xy((x-rx,y-ry,x+rx,y+ry)), fill=fill, outline=edge, width=max(1,w*self.s), tags=self.layer)

    def txt(self,x,y,text,size=15,col=WHITE,bold=False,anchor="center"):
        self.create_text(*self.xy((x,y)),text=text,fill=col,anchor=anchor,
                         font=(FACE,-max(8,round(size*self.s)),"bold" if bold else "normal"),tags=self.layer)

    def arrow(self,x,y,dx,dy,col,w=2,tip=5):
        length=(dx*dx+dy*dy)**.5
        ux,uy=dx/length,dy/length
        tx,ty=x+dx,y+dy
        self.line(x,y,tx,ty,ink=col,w=w)
        self.line(tx-ux*tip-uy*tip,ty-uy*tip+ux*tip,tx,ty,
                  tx-ux*tip+uy*tip,ty-uy*tip-ux*tip,ink=col,w=w)

    def flag(self,x,y,col,reverse=False):
        d=-1 if reverse else 1
        self.line(x,y+8,x,y-10,ink=col,w=2)
        self.poly(x,y-10,x+d*13,y-8,x+d*9,y-2,x,y-3,fill=col)

    def verdict(self,x,y,hit,r=10):
        col=SEA if hit else RED
        self.oval(x,y,r,r,NIGHT,col,2)
        if hit:
            self.line(x-r*.55,y,x-r*.14,y+r*.42,x+r*.58,y-r*.42,ink=col,w=2.5)
        else:
            self.line(x-r*.37,y-r*.37,x+r*.37,y+r*.37,ink=col,w=2)
            self.line(x+r*.37,y-r*.37,x-r*.37,y+r*.37,ink=col,w=2)

    def crane(self,x,bottom,scale,flip=1):
        def points(*p):
            return tuple((x+flip*v*scale if i%2==0 else bottom+v*scale) for i,v in enumerate(p))
        c="#315362"
        self.line(*points(0,0,0,-66,56,-87,75,-80),ink=c,w=3)
        self.line(*points(-9,0,0,-58,15,0,0,-35,-5,-22,7,-12),ink=c,w=2)
        self.line(*points(-13,-60,74,-80,0,-55),ink=c,w=2)
        self.line(*points(53,-74,53,-46,58,-41,63,-46),ink=c)
        self.poly(*points(-13,-61,7,-65,7,-57,-13,-55),fill=c)

    def scenery(self):
        self.rect(1,1,899,459,"#102333","#35505d",2)
        self.rect(3,3,897,46,"#132634")
        self.rect(3,47,897,80,"#193341")
        self.rect(3,80,897,109,"#1c3947")
        self.oval(826,69,12,12,"#e2c999")
        for x,h,width in ((13,14,31),(45,22,19),(68,11,41),(797,13,33),(840,18,24),(867,12,27)):
            self.rect(x,111-h,x+width,112,"#284754")
        self.crane(45,108,.75); self.crane(858,111,.62,-1)
        self.rect(3,110,897,299,WATER)
        for row in range(8):
            for j in range(7):
                x=17+(j*149+row*41)%858; y=120+row*23
                self.line(x,y,x+10+(row+j)%4*3,y,ink="#234554")
        self.rect(3,296,897,457,"#142939")
        self.line(3,296,897,296,ink=STEEL,w=3)
        self.line(3,303,897,303,ink="#294352",w=2)
        for x in range(14,892,52):
            self.oval(x,300,1.5,1.5,"#77929b")

    def dock(self,x,y,lit,hit,clock):
        self.rect(x-43,y+16,x+43,y+21,"#264654")
        if not lit:
            self.line(x-39,y+16,x+39,y+16,ink="#43616a",dash=(3,6))
            return
        self.rect(x-41,y-19,x+41,y+16,"#19444b")
        for sx in (-1,1):
            for sy in (-1,1):
                ex,ey=x+sx*45,y+sy*22
                self.line(ex-sx*13,ey,ex,ey,ex,ey-sy*9,ink=SEA,w=3)
        self.line(x-41,y+19,x+41,y+19,ink=SEA,w=2)
        self.oval(x-53,y,3.5,3.5,SEA if hit or sin(clock*3)>-.45 else "#4d9c92")

    def cabin(self,k,x,y):
        # Distinct silhouettes, not just four differently colored rectangles.
        if k==0:
            self.poly(x-32,y+3,x-24,y-10,x-14,y-14,x+13,y-14,x+23,y-8,x+33,y+3,
                      x+25,y+12,x-22,y+12,fill="#d89b59",edge="#ffda91",w=2)
            for dx in (-11,9):
                self.oval(x+dx,y-3,6,6,"#244353","#f4d69b")
                self.oval(x+dx-1,y-4,2,2,"#d8f0df")
            self.line(x-20,y+7,x+20,y+7,ink="#ffde96",w=2)
        elif k==1:
            self.rect(x-33,y-6,x-28,y+7,"#90cabb")
            self.rect(x+28,y-6,x+33,y+7,"#90cabb")
            self.rect(x-28,y-14,x+28,y+14,"#25685f","#8be2be",2)
            for dx in (-22,-17,17,22):
                self.line(x+dx,y-9,x+dx,y+9,ink="#73bca6",w=2)
            self.poly(x+3,y-10,x-8,y+2,x-1,y+2,x-3,y+10,x+9,y-2,x+2,y-2,fill="#e3f8a9")
        elif k==2:
            self.line(x-10,y-12,x-10,y-18,x+10,y-18,x+10,y-12,ink="#d3ccd8",w=3)
            self.rect(x-27,y-12,x+27,y+15,"#ba7977","#ffd3b7",2)
            self.line(x-18,y-11,x-18,y+14,ink="#885c63",w=3)
            self.line(x+18,y-11,x+18,y+14,ink="#885c63",w=3)
            self.rect(x-3,y-7,x+3,y+9,"#ffedd9"); self.rect(x-8,y-2,x+8,y+4,"#ffedd9")
        else:
            self.rect(x-20,y-13,x+20,y+10,"#4d87a5","#b4e3eb",2)
            self.oval(x-20,y-1,7,12,"#4d87a5","#b4e3eb"); self.oval(x+20,y-1,7,12,"#4d87a5","#b4e3eb")
            self.rect(x-14,y-13,x+14,y+11,"#4d87a5")
            self.poly(x,y-9,x-6,y,x-5,y+5,x,y+7,x+5,y+5,x+6,y,fill="#d2f5fa")
            self.line(x-25,y+15,x+25,y+15,ink="#8faeb7",w=3)

    def draw_lanes(self,app,clock):
        goal=app.goal()
        outcome=app.arrival()
        indices=(0,) if app.lesson else range(4)
        dx=self.base[0]; dy=self.base[1]
        for i in indices:
            x=(450 if app.lesson else LANES[i])+(dx[i]-11.5)*2
            y=176+(dy[i]-22)*1.3
            for off in (-61,61):
                self.line(x+off,y-84,x+off,y+84,ink=RAIL,w=2)
                for tick in (-66,-43,-21,0,21,43,66):
                    self.line(x+off-3,y+tick,x+off+3,y+tick,ink="#517380")
            self.txt(x,74,f"{i+1:02}",13,STEEL,True)
            target=0 if goal is None else int(goal[i])
            for dest in (-1,1):
                self.dock(x,y+dest*LIFT,target==dest,outcome is not None and outcome.motion[i]==dest,clock)
            px,py=x,y
            if app.axis=="x" and app.echo is not None:
                px+=float(app.echo[i])*9*app.p
            if app.axis=="y" and app.voyage is not None:
                py+=int(app.voyage.motion[i])*LIFT*app.p
            if abs(px-x)>2 or abs(py-y)>2:
                self.line(x,y,px,py,ink="#8aa9ae",w=2,dash=(3,4))
            if app.lesson and outcome is None and abs(py-y)<2 and app.axis!="x":
                self.arrow(x+47,y+4,0,25,SEA,2)
            self.cabin(i,px,py)
            if outcome is not None and target:
                hit=int(outcome.motion[i])==target
                self.verdict(px+33,py-14,hit)
                age=app.elapsed-(1.29 if app.phase=="reveal" else 0)
                if hit and 0<=age<.8:
                    u=age/.8
                    for vx,vy in ((-28,-18),(-17,-32),(10,-34),(25,-19)):
                        sx,sy=px+vx*(.7+u),py-8+vy*u+14*u*u
                        self.line(sx,sy,sx+vx*.14*(1-u),sy+vy*.16*(1-u),ink=GOLD,w=2)
            if not app.lesson and app.show_gear() and app.trip.deal.gear and app.trip.rig.mark[i]:
                self.arrow(x+54,76,10,0,GOLD,1.5,3); self.arrow(x+64,82,-10,0,GOLD,1.5,3)

    def controls(self,app,sides):
        for side,(hx,hy) in HOME.items():
            enabled_x,enabled_y=app.track_axes(side)
            selected=(side==app.selected)
            col=PAINT[side]
            if enabled_x:
                self.line(hx,hy,hx+THROW_X+9,hy,ink=RAIL,w=7)
                self.arrow(hx+45,hy,THROW_X-44,0,col,2,6)
                self.line(hx+THROW_X,hy-14,hx+THROW_X,hy+14,ink=col,w=2,dash=(3,3))
            if enabled_y:
                self.line(hx,hy,hx,hy+THROW_Y+3,ink=RAIL,w=7)
                self.arrow(hx,hy+23,0,THROW_Y-23,col,2,6)
                self.line(hx-33,hy+THROW_Y,hx+33,hy+THROW_Y,ink=col,w=2,dash=(3,3))
            x=hx+(THROW_X*app.p if selected and app.axis=="x" else 0)
            y=hy+(THROW_Y*app.p if selected and app.axis=="y" else 0)
            self.knobs[side]=x,y
            hi=(enabled_x or enabled_y or selected)
            edge=col if hi else "#557281"
            self.rect(x-40,y-14,x+40,y+20,"#0c1e2c")
            self.poly(x-40,y-9,x-33,y-16,x+33,y-16,x+40,y-9,x+40,y+11,x+33,y+16,
                      x-33,y+16,x-40,y+10,fill=("#665035" if side<0 else "#354e70") if hi else "#293a47",
                      edge=edge,w=3 if selected else 2)
            for off in (-30,-24,24,30):
                self.line(x+off,y-6,x+off,y+6,ink=edge,w=2)
            self.txt(x,y,HANDLE[side],22,WHITE,True)
            if selected and app.mode=="human" and app.phase in ("deck","probe","select") and app.p>=1:
                self.oval(x-48,y,4,4,SEA)
                self.txt(x,y+26,"松手",13,SEA,True)
            if side in sides:
                trace=sides[side].trace
                for i in range(1 if app.lesson else 4):
                    sign=1 if trace[i]>0 else -1
                    self.arrow(hx-71,hy-19+i*12,sign*10,0,col,1.5,3)
            else:
                self.oval(hx-70,hy,3,3,"",STEEL)

    def hand(self,x,y):
        self.poly(x,y,x+2,y-11,x+5,y-13,x+7,y-11,x+7,y-3,x+10,y-4,
                  x+12,y-2,x+15,y-2,x+18,y+1,x+18,y+9,x+12,y+17,x+4,y+17,
                  x-5,y+6,x-7,y+1,x-5,y-1,fill="#f2d6a5",edge="#5a4134",w=2)

    def draw(self,app):
        w,h=self.winfo_width(),self.winfo_height()
        if min(w,h)<10 or self.base is None:
            return
        self.s=min(w/self.W,h/self.H); self.ox=(w-self.W*self.s)/2; self.oy=(h-self.H*self.s)/2
        if self.backing!=(w,h):
            self.delete("all"); self.layer="port"; self.scenery()
            self.layer="live"; self.backing=(w,h)
        else: self.delete("live")
        clock=app.clock
        goal=app.goal(); outcome=app.arrival()
        self.flag(24,24,SEA,app.order==1)
        name="练手" if app.lesson else "去程" if app.order==0 else "返航"
        self.txt(50,24,name,16,WHITE,True,"w")
        self.txt(450,24,app.deck_title(),21,WHITE,True)
        if goal is not None:
            required=1 if app.lesson else 2
            hits=outcome.hits if outcome is not None else 0
            n=1 if app.lesson else 3
            for j in range(n):
                x=796+j*23
                self.oval(x,22,7,7,SEA if j<hits else "#263e4a",SEA if j<required else STEEL,2)
            self.txt(877,25,f"{required}/{n}",12,MUTED,True,"e")
        self.draw_lanes(app,clock)
        self.controls(app,app.soundings)
        cue=app.gesture_cue()
        if cue and not app.paused:
            side,axis=cue; x,y=HOME[side]
            t=smooth((clock%2.2-.25)/1.1)
            self.hand(x+26+(THROW_X*t if axis=="x" else 0),y+5+(THROW_Y*t if axis=="y" else 0))
        if outcome is not None:
            self.verdict(450,366,bool(outcome.won),21)
            self.txt(450,403,f"{outcome.hits} / {1 if app.lesson else 3}",16,WHITE,True)
        elif app.phase in ("waiting","auto_probe","auto_pull","reveal"):
            self.oval(450,369,17,17,"#223e4b","#699da4",2)
            for off in (-6,6): self.oval(450+off,366,2.5,2.5,SEA)
            self.line(445,376,455,376,ink=SEA,w=2)
        if app.paused:
            self.rect(401,137,499,207,"#183443",GOLD,2)
            self.rect(435,155,445,190,GOLD); self.rect(456,155,466,190,GOLD)
