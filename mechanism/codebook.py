"""描述性解析：不改权重，不用码表帮网络答题；枚举三位排列与极性。"""
import argparse,itertools
from pathlib import Path
import numpy as np
from .data import read,dump,bits,index,CFG
def audit(rel):
    target=read(rel/"observed/circuits.npz")["d8_states"].flatten()
    out={}
    for seed in CFG["seeds"]:
        emitted=read(rel/"sealed/language"/f"joint_latent-{seed}.npz")["d8_message"].flatten()
        counts=np.bincount(8*target+emitted,minlength=64).reshape(8,8)
        if not np.all(counts.sum(1)):raise ValueError("unseen truth state")
        acc=-1
        for p in itertools.permutations(range(3)):
            for inv in range(8):
                tab=index(bits(np.arange(8))[:,p])^inv
                score=float((emitted==tab[target]).mean())
                if score>acc:best=dict(permutation=list(p),xor=inv,truth_to_code=tab.tolist());acc=score
        out[str(seed)]=dict(n=len(target),best_posthoc_accuracy=acc,counts=counts.tolist(),**best)
    return out
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--release",type=Path,default=Path("evidence/mechanism"));p.add_argument("--out",type=Path)
    a=p.parse_args();dump(a.out or a.release/"codebook.json",audit(a.release))
