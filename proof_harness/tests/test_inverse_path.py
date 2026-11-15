"""Independent exact-arithmetic checks, never counted as AI proof successes.

Path 0--1--2. First POSITIVE return to 0, including a first-step hold.
For killing, missing mass is retained as absorption, not renormalized.
"""
from fractions import Fraction as F
import unittest

def first_return(P, target, times):
    alive={target:F(1)}
    output=[]
    for _ in range(times):
        nxt={}; mass=F(0)
        for x,prob in alive.items():
            for y,transition in enumerate(P[x]):
                if y==target:mass+=prob*transition
                else:nxt[y]=nxt.get(y,F(0))+prob*transition
        output.append(mass);alive=nxt
    return output

class ExactPathTests(unittest.TestCase):
    def test_centrality_recovery_from_first_returns(self):
        a,b=F(1,5),F(1,7)
        m0,m2=F(3,4),F(2,3)
        for numerator in range(4,24):
            m1=F(numerator,10)
            P=[[1-a/m0,a/m0,0],[a/m1,1-(a+b)/m1,b/m1],[0,b/m2,1-b/m2]]
            r1,r2=first_return(P,0,2)
            self.assertEqual(r1,1-a/m0)
            self.assertEqual(a*a/(m0*r2),m1)

    def test_absorption_recovery_from_first_returns(self):
        a,b=F(1,5),F(1,7)
        for numerator in range(0,13):
            q0,q1,q2=F(1,10),F(numerator,20),F(1,10)
            P=[[1-a-q0,a,0],[a,1-a-b-q1,b],[0,b,1-b-q2]]
            self.assertTrue(all(sum(row)<=1 and min(row)>=0 for row in P))
            r1,r2,r3=first_return(P,0,3)
            self.assertEqual(r1,1-a-q0)
            self.assertEqual(1-a-b-r3/r2,q1)

    def test_killing_is_not_renormalized(self):
        P=[[F(1,4),F(1,4)],[F(1,4),F(1,4)]]
        self.assertEqual(first_return(P,0,2),[F(1,4),F(1,16)])

if __name__=='__main__':unittest.main()
