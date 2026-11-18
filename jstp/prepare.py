from pathlib import Path
import argparse
from .arena import collect
from .common import configure
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["data","features"]);p.add_argument("--work",type=Path,default=Path("work"));p.add_argument("--device",default="cuda")
    a=p.parse_args()
    if a.stage=="data":collect(a.work/"data")
    else:
        from .foundation import cache
        configure(1729,a.device);cache(a.work,a.device)
