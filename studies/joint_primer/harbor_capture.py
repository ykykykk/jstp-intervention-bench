"""Record live public camera frames and the two sealed decisions for the comparison films."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from lategoal.rig import EXAM
from .game_round import Dispatch, PAPER_EXAMPLE


ARMS = ("joint", "act", "diffusion", "vjepa21", "dino", "smol")
NAMES = dict(zip(ARMS, ("JSTP", "ACT", "Diffusion Policy", "V-JEPA 2.1", "DINO-WM", "SmolVLA"), strict=True))
BUDGETS = (128, 1024, 16384)
CARDS = (0, 2)  # Both consecutive published examples, not selected from this recording's successes.
SEED = 9301
HERE = Path(__file__).resolve().parent


def digest(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()


def checkpoints(arm,budget):
    if arm=="joint":
        base, name = HERE/"evidence/train", "last.pt"
    elif arm in ("dino","smol"):
        base, name = HERE/"evidence/official", "adapter.pt"
    else:
        base, name = HERE.parent/"fourtrack_foundations/evidence"/arm, "adapter.pt"
    path = base/"models"/f"{arm}-{budget}-{SEED}"/name
    with path.open("rb") as f:
        h=hashlib.file_digest(f, "sha256").hexdigest()
    return {"file":path.relative_to(HERE.parents[1]).as_posix(), "sha256":h}


def record(arm,budget,out):
    from .game_pilots import FrozenPilot
    import torch
    pilot=FrozenPilot(arm,budget,SEED)
    frames, rows = {}, []
    try:
        for pair,card in enumerate(CARDS,1):
            trial=Dispatch(replace(PAPER_EXAMPLE,card=card))
            trial.begin_probe()
            trial.finish_probe()
            view=pilot.observe(trial.observation.copy())
            decisions=[]
            for card_id in trial.card_ids:
                decision=pilot.predict(view,int(card_id))
                decisions.append(dict(card=int(card_id),goal=EXAM[card_id].tolist(),side=decision.side))
                trial.seal(decision.side)
            shots = [trial.rig.base,trial.peak,*(o.frame for o in trial.outcomes)]
            frames[f"pair{pair}"] = np.stack(shots)
            for step,o in zip(decisions,trial.outcomes,strict=True):
                step.update(motion=o.motion.tolist(),hits=o.hits,passed=o.won)
            rows.append(dict(pair=pair,deal=asdict(trial.deal),actions=decisions,
                             paired_pass=all(o.won for o in trial.outcomes),camera_sha256=[digest(f) for f in shots],
                             history_sha256=digest(trial.observation)))
    finally:
        pilot.close()
    del pilot,view
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    stem=f"{arm}-{budget}"
    # The JSON marks completion only after the corresponding camera file has been closed.
    with NamedTemporaryFile(dir=out,suffix=".npz",delete=False) as f:
        tmp=Path(f.name); np.savez_compressed(f,**frames)
    tmp.replace(out/f"{stem}.npz")
    row=dict(arm=arm,budget=budget,training_seed=SEED,weights=checkpoints(arm,budget),pairs=rows)
    with NamedTemporaryFile(dir=out,suffix=".json",mode="w",encoding="utf8",delete=False) as f:
        tmp=Path(f.name); json.dump(row,f,ensure_ascii=False,allow_nan=False)
    tmp.replace(out/f"{stem}.json")
    print("CAPTURED",stem,[[a["side"] for a in p["actions"]] for p in rows],
          [p["paired_pass"] for p in rows],flush=True)


def collect(source,dest):
    archive={}; rows=[]
    for k in BUDGETS:
        for arm in ARMS:
            stem=f"{arm}-{k}"
            row=json.loads((source/f"{stem}.json").read_text(encoding="utf8")); rows.append(row)
            with np.load(source/f"{stem}.npz",allow_pickle=False) as shots:
                for p in row["pairs"]:
                    key=f"pair{p['pair']}"; arrays=shots[key]
                    if [digest(f) for f in arrays]!=p["camera_sha256"]:
                        raise ValueError("camera mismatch: "+stem+"/"+key)
                    archive[f"{stem}-{key}"]=arrays
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/"camera.npz").open("wb") as f: np.savez_compressed(f,**archive)
    book=dict(protocol="paper: one assigned probe; two late opposite goals sealed before execution",
              models=NAMES,budgets=list(BUDGETS),seed=SEED,first_card_ids=list(CARDS),
              frame_order=["home","probe_peak","outbound_end","return_end"],runs=rows)
    (dest/"runs.json").write_text(json.dumps(book,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["record","collect"]); p.add_argument("--scratch",type=Path,required=True)
    p.add_argument("--out",type=Path); p.add_argument("--arm",choices=ARMS)
    a=p.parse_args()
    if a.action=="collect":
        if not a.out: p.error("collect requires --out")
        collect(a.scratch,a.out); return
    os.environ["HF_HUB_OFFLINE"]="1"
    a.scratch.mkdir(parents=True,exist_ok=True)
    for arm in (ARMS if a.arm is None else [a.arm]):
        for k in BUDGETS:
            stem=f"{arm}-{k}"
            if (a.scratch/f"{stem}.json").is_file() and (a.scratch/f"{stem}.npz").is_file(): continue
            print("LOADING",stem,flush=True)
            record(arm,k,a.scratch)


if __name__=="__main__":
    main()
