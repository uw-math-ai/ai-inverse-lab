"""Integration tests require the installed pinned Lean/Mathlib environment."""
import unittest
from proof_harness.catalog import TASK_BY_ID, Task
from proof_harness.verifier import ROOT, LEAN, verify, lean_environment

@unittest.skipUnless(LEAN.exists(),'run setup first')
class LeanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env=lean_environment()
    def check(self,task,steps,baseline='B0'):
        return verify(task,steps,baseline,ROOT/'runs/tests'/self._testMethodName,self.env)
    def test_valid_renewal(self):
        result=self.check(TASK_BY_ID['renewal3'],[{'op':'rw','terms':['h3','h2','h1']},{'op':'ring'}])
        self.assertTrue(result['accepted'],result)
    def test_false_claim_rejected(self):
        task=Task('false_claim','test','deliberately false','(x : ℝ)','x = x+1','x')
        self.assertFalse(self.check(task,[{'op':'ring'}])['accepted'])
    def test_sorry_rejected(self):
        self.assertEqual(self.check(TASK_BY_ID['renewal3'],[{'op':'exact','term':'sorry'}])['status'],'gate_rejected')
    def test_normal_equation(self):
        result=self.check(TASK_BY_ID['control_pairing'],[
            {'op':'have','name':'hp2','type':'W h = p','steps':[{'op':'exact','term':'hJ eqn'}]},
            {'op':'rw','terms':['hp2']},{'op':'exact','term':'hp'}])
        self.assertTrue(result['accepted'],result)
    def test_absorption(self):
        result=self.check(TASK_BY_ID['absorption_difference'],[{'op':'ring'}])
        self.assertTrue(result['accepted'],result)

    def test_empty_is_abstention(self):
        self.assertEqual(self.check(TASK_BY_ID['renewal3'],[])['status'],'not_attempted')

    def test_parenthesized_lean_projection(self):
        result=self.check(TASK_BY_ID['path_recovery'],[
            {'op':'have','name':'hp','type':'d*m1 = a^2/m0','steps':[{'op':'exact','term':'(eq_div_iff hm1).mp h'}]},
            {'op':'have','name':'hc','type':'d*m1*m0 = a^2','steps':[{'op':'exact','term':'(eq_div_iff hm0).mp hp'}]},
            {'op':'field_simp','terms':['hm0','hd']},{'op':'nlinarith','terms':['hc']}])
        self.assertTrue(result['accepted'],result)

if __name__=='__main__':unittest.main()
