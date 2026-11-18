import unittest
import numpy as np
from mechanism.data import CFG,apparatus,episodes,context,scene,images,act,programs,circuit,bits

class Instruments(unittest.TestCase):
    def test_no_motor_candidate_and_no_wiring_labels_in_observation(self):
        first=CFG["train"][0]+19
        b=apparatus(first,1,False);r=episodes(b,first)
        self.assertNotIn("futures",b)
        active=r[r[:,5]==0];self.assertTrue((active[:,6]==0).all())
        same=active[(active[:,2]==0)&(active[:,3]==0)&(active[:,4]==1)]
        a,c=context(b,same)
        np.testing.assert_array_equal(a[:96],c[:96]);np.testing.assert_array_equal(a[192:],c[192:])
        self.assertTrue((a[96:192]!=c[96:192]).any())
        s=scene(first);frame,_=images(s,0)
        np.testing.assert_array_equal(act(s,0,0),frame)
        for d in (-1,1):np.testing.assert_array_equal(act(s,0,d,0),act(s,0,0,d))

    def test_counterfactual_gate_pairs(self):
        rng=np.random.default_rng(13);a=bits(rng.integers(8,size=96));c=programs(rng,96,8);t=c.copy();t[:,0,2]^=1
        x,y=circuit(a,c),circuit(a,t)
        self.assertTrue(np.all((x!=y).any(-1)))
        np.testing.assert_array_equal(circuit(x[:,-1],c[:,::-1])[:,-1],a)

if __name__=="__main__":unittest.main()
