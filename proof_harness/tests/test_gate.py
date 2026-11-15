import unittest
from proof_harness.gate import GateError, compile_steps

class GateTests(unittest.TestCase):
    def test_absent_pack_rejected(self):
        with self.assertRaises(GateError):
            compile_steps([{'op':'exact','term':'Baseline.scalar_recover a b c h1 h2 h3'}], 'B0', {'a','b','c','h1','h2','h3'})

    def test_present_pack_allowed(self):
        self.assertIn('Baseline.scalar_recover',compile_steps([{'op':'exact','term':'Baseline.scalar_recover a b c h1 h2 h3'}], 'B5', {'a','b','c','h1','h2','h3'}))

    def test_command_injection(self):
        for s in ['sorry','by exact h','h\naxiom fake : False','h; run_tac evil','h /- escape -/',
                  'Classical.choice h','Lean.ofReduceBool h','False.elim h','h`','h\\n']:
            with self.subTest(s=s),self.assertRaises(GateError):
                compile_steps([{'op':'exact','term':s}], 'B0', {'h'})

    def test_no_arbitrary_tactics(self):
        for op in ['sorry','aesop','exact?','run_tac','native_decide','import','set_option']:
            with self.subTest(op=op), self.assertRaises(GateError):compile_steps([{'op':op}], 'B0',set())

    def test_simp_is_only(self):
        self.assertEqual(compile_steps([{'op':'simp','terms':['h']}],'B0',{'h'}),'simp only [h]')

    def test_derived_fact_scope(self):
        s=compile_steps([{'op':'have','name':'aux','type':'x = x','steps':[{'op':'rfl'}]},
                         {'op':'exact','term':'aux'}], 'B0',{'x'})
        self.assertIn('exact aux',s)
        with self.assertRaises(GateError):compile_steps([{'op':'exact','term':'aux'}],'B0',{'x'})

    def test_no_recursive_local_scope_leak(self):
        with self.assertRaises(GateError):
            compile_steps([{'op':'have','name':'aux','type':'x = x','steps':[
                {'op':'have','name':'hidden','type':'x = x','steps':[{'op':'rfl'}]}, {'op':'exact','term':'hidden'}]},
                {'op':'exact','term':'hidden'}], 'B0',{'x'})

    def test_lambda_binder_is_local_and_not_a_namespace_escape(self):
        self.assertIn('fun z',compile_steps([{'op':'exact','term':'(fun z : ℝ => z) x'}],'B0',{'x'}))
        with self.assertRaises(GateError):
            compile_steps([{'op':'exact','term':'(fun Baseline : ℝ => Baseline.scalar_recover) x'}],'B0',{'x'})

    def test_parenthesized_projection(self):
        self.assertIn('.mp',compile_steps([{'op':'exact','term':'(eq_div_iff hm).mp h'}],'B0',{'hm','h'}))
        for term in ['(eq_div_iff hm).unsafe h','(eq_div_iff hm).mpx h','(h).sorry']:
            with self.subTest(term=term),self.assertRaises(GateError):
                compile_steps([{'op':'exact','term':term}],'B0',{'hm','h'})

if __name__=='__main__':unittest.main()
