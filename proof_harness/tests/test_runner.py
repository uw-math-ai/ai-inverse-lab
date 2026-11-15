"""Orchestration tests use mocked models and verifiers; never counted as experiments."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from proof_harness.run import Budget, episode, require_branch
from proof_harness.backend import BackendError, UsageLimitError
from proof_harness.catalog import TASKS

class RunnerTests(unittest.TestCase):
    def test_other_branch_refused(self):
        with patch('proof_harness.run.git',return_value='main'):
            with self.assertRaises(RuntimeError):require_branch()

    def test_budget_counts_failures_and_stops(self):
        with tempfile.TemporaryDirectory() as d:
            b=Budget(1,Path(d)/'calls.jsonl');b.take('attempt')
            with self.assertRaises(BackendError):b.take('forbidden retry')
            self.assertEqual(len(b.path.read_text().splitlines()),1)

    def test_usage_limit_circuit_breaker(self):
        with tempfile.TemporaryDirectory() as d:
            b=Budget(10,Path(d)/'calls.jsonl');b.take('one');b.stop('usage limit')
            with self.assertRaises(UsageLimitError):b.take('two')
            self.assertEqual(b.count,1)

    def test_revision_reward_once(self):
        t=TASKS[0]
        creator={'attempts':[{'task':t.id,'steps':[{'op':'ring'}]}]}
        critic={'feedback':[{'task':t.id,'suggestion':'repair'}]}
        calls=[(creator,{'usage':{}}),(critic,{'usage':{}})]*2
        verifications=[{'task':t.id,'accepted':False,'status':'lean_rejected','feedback':'bad'},
                       {'task':t.id,'accepted':True,'status':'verified','feedback':'ok'}]
        args=argparse.Namespace(rounds=3,model='mock',effort='low',call_timeout=1)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            with patch('proof_harness.run.require_branch'),patch('proof_harness.run.TASKS',[t]),\
                 patch('proof_harness.run.call',side_effect=calls),patch('proof_harness.run.verify',side_effect=verifications):
                r=episode('B0',0,args,folder,Budget(6,folder/'calls.jsonl'),{})
            self.assertEqual(r['creator_points'],10)
            self.assertEqual(r['critic_process_points'],2)
            self.assertEqual(r['model_calls'],4)
            self.assertEqual(r['verified'],1)

    def test_failure_does_not_fall_back_to_success(self):
        args=argparse.Namespace(rounds=3,model='mock',effort='low',call_timeout=1)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            with patch('proof_harness.run.require_branch'),patch('proof_harness.run.call',side_effect=BackendError('offline')):
                r=episode('B0',0,args,folder,Budget(6,folder/'calls.jsonl'),{})
            self.assertEqual(r['verified'],0)
            self.assertEqual(r['model_calls'],1)
            self.assertEqual(r['errors'],['offline'])

if __name__=='__main__':unittest.main()
