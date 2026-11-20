# Four-Track Harbor · recordings / 四轨货运实操录像

**[Play all three films in the browser / 在线播放](https://ykykykk.github.io/jstp-intervention-bench/)**. Each silent, bilingual, 32-second film shows all six controllers together. Choose a row to watch or download it.

| Action-label budget / 动作示范量 | Watch / 在线看 | File / MP4 |
| ---: | --- | --- |
| 128 | [Play 128](https://ykykykk.github.io/jstp-intervention-bench/#128) | [Download](https://github.com/ykykykk/jstp-intervention-bench/releases/download/four-track-films-2026-11-21/four-track-128.mp4) |
| 1,024 | [Play 1,024](https://ykykykk.github.io/jstp-intervention-bench/#1024) | [Download](https://github.com/ykykykk/jstp-intervention-bench/releases/download/four-track-films-2026-11-21/four-track-1024.mp4) |
| 16,384 | [Play 16,384](https://ykykykk.github.io/jstp-intervention-bench/#16384) | [Download](https://github.com/ykykykk/jstp-intervention-bench/releases/download/four-track-films-2026-11-21/four-track-16384.mp4) |

**What goes in and out.** The controller does not inspect source code or read hidden wiring, ground-truth object coordinates, a future frame or the correct handle to decide its test action. It is **not** a raw-screen, OS-mouse agent, either. The shared public history contains two 72-value brightness grids (home and probe difference), 12 camera-derived displacement readings, the probe handle, four public marks and the clutch setting. The current goal card is supplied separately. Each adapter encodes these observations in its own way; the controller selects A/B and the simulator executes. Human players drag the mouse; action labels are available during training, not as test answers. [Observation construction and camera judge](../../lategoal/rig.py) · [Frozen model inputs and outputs](../../studies/joint_primer/game_pilots.py).

**怎么读。** 青框是目标，不是模型想象的画面。横试后四舱复位，目标卡才出现；两张卡都选好把手，才从同一起点分别执行。三条指定轨道中至少两条符合算单卡成功；两张都过，这一对才通过。主画面放大游戏动作，左下角是完整 JSTP 当时的原始相机端点。模型读相机特征、公开仪表和目标卡，不读隐藏接线或真值坐标，也不操作系统鼠标。

**Recording conditions.** JSTP, ACT, Diffusion Policy, V-JEPA 2.1, DINO-WM and SmolVLA are the configurations in Table 2, not untuned general-purpose agents. The fixed seed is 9301. Each budget uses the same 16,384 physical training histories and a different number of labeled goal–action examples. We ran all 18 checkpoints on layout 113670000, original wiring and clutch, fixed probe A. Zero-based card pairs (0,1) and (2,3) are the two consecutive examples in §5.1. These are fresh executions on a previously published layout, **not** a newly blind evaluation. There are 36 paired attempts and 72 formal strokes. Misses are retained. Compare coverage and averages in the [full study](../../studies/fourtrack_foundations/README.md); two displayed pairs do not rank the methods.

**Recording and playback.** The [per-action log](runs.json) contains the checkpoint digest, public history digest, decisions, measured directions, hits and pass/fail flags. The [camera archive](camera.npz) has one uint8 array per model/budget/pair: home, horizontal probe peak, outbound terminal frame, return terminal frame. The game renderer animates between the measured endpoints. Timings in the films are aligned deliberately and do not measure inference speed. They use the single-probe **paper route**, not the unrestricted free-play scoring.

On the original Windows setup, from the repository root:

```powershell
$py = '.\work\vla-env\Scripts\python.exe'
& $py -m studies.joint_primer.harbor_capture record --scratch work/my-harbor-takes
& $py -m studies.joint_primer.harbor_capture collect --scratch work/my-harbor-takes --out work/my-harbor-films
& $py -m studies.joint_primer.harbor_film work/my-harbor-films
```

Recording needs the corresponding original frozen checkpoints and local bases. Rendering from the published traces alone needs Pillow, NumPy, the game code, FFmpeg, and the listed Windows fonts; it does not reload model weights. [Human-playable game](../FOUR-TRACK-GAME.md).
