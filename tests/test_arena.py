import unittest
import numpy as np
from jstp.arena import scene,sense,execute
from jstp.rules import rule
from jstp.v7_world import TABLE

class PairedWorld(unittest.TestCase):
    def test_invariants_and_visible_rule(self):
        for split in (False,True):
            for seed in range(70100,70116):
                s=scene(seed,split);c,e=sense(s)
                self.assertEqual(s.singleton==1,split)
                r=rule(c[None],e[None],s.probe[None],s.options[None],"visible_rule")[0]
                actual=np.array([[execute(s,j,a) for a in s.options] for j in (0,1)])
                np.testing.assert_array_equal(np.rint(r*255).astype('uint8'),actual)
                for compound in (False,True):
                    goals,mask=s.goals(compound);use=mask.astype(bool)
                    self.assertEqual(int(use.sum()),TABLE*32*(1+int(compound)))
                    for g in goals:
                        winners=[[np.array_equal(x[:TABLE][use],g[use]) for x in row] for row in actual]
                        self.assertTrue(all(sum(w)==1 for w in winners));self.assertNotEqual(winners[0],winners[1])

    def test_one_bit_observer_cannot_read_other_lanes(self):
        s=scene(34); c,e=sense(s); a=e.copy(); a[:,:TABLE,:64]=c[:TABLE,:64]
        np.testing.assert_array_equal(rule(c[None],e[None],s.probe[None],s.options[None],"one_bit"),
                                      rule(c[None],a[None],s.probe[None],s.options[None],"one_bit"))
if __name__=="__main__":unittest.main()
