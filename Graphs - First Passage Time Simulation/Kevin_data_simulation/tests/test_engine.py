import itertools
import unittest
import numpy as np
from forward_sim.engine import Cancelled, exact_first_passage, exact_outcomes, simulate, wilson
from forward_sim.graphs import GRAPHS, Graph
from forward_sim.model import Experiment, build_kernel, parse_config


class GraphTests(unittest.TestCase):
    def test_ladder_matches_figure_one(self):
        graph = GRAPHS['paper8']
        expected = {(0,1),(1,2),(2,3),(4,5),(5,6),(6,7),(0,4),(1,5),(2,6),(3,7)}
        self.assertEqual(set(graph.edges), expected)
        self.assertEqual(graph.observed, (1,2,3))
        self.assertEqual(graph.mu, (1.,)*8)
        self.assertEqual(graph.weight, .25)

    def test_grid_matches_figure_four(self):
        graph = GRAPHS['paper9']
        expected = {(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)}
        self.assertEqual(set(graph.edges), expected)
        self.assertEqual(graph.observed, (0,1,2))
        self.assertEqual(graph.mu, (2.,3.,2.,3.,4.,3.,2.,3.,2.))

    def test_paper_unit_kernels_and_detailed_balance(self):
        for graph_id in GRAPHS:
            config = parse_config({'graph':graph_id})
            K, kill, P = build_kernel(config)
            self.assertEqual(config.dt, 1)
            np.testing.assert_allclose(P.sum(axis=1), 1)
            np.testing.assert_array_equal(kill, 0)
            weighted = np.asarray(config.graph.mu)[:,None]*K
            np.testing.assert_allclose(weighted, weighted.T)
        self.assertEqual(build_kernel(parse_config({'graph':'paper8'}))[0][1,1], .25)
        np.testing.assert_array_equal(build_kernel(parse_config({'graph':'paper9'}))[0].diagonal(), 0)

    def test_killing_auto_step_preserves_heat_operator(self):
        zero = parse_config({'graph':'paper9'})
        base, _, _ = build_kernel(zero)
        cfg = parse_config({'graph':'paper9','q':[0,0,0,0,.4,0,0,0,0]})
        K, kill, P = build_kernel(cfg)
        self.assertAlmostEqual(cfg.dt, 1/1.4)
        np.testing.assert_allclose((K-np.eye(9))/cfg.dt, base-np.eye(9)-np.diag(cfg.q), atol=1e-15)
        np.testing.assert_allclose(K.sum(axis=1)+kill, 1)
        self.assertTrue(np.all(P >= 0))


class ValidationTests(unittest.TestCase):
    def test_bad_inputs_rejected(self):
        bad = [{'q':-1},{'q':float('nan')},{'q':float('inf')},{'q':True},{'q':[0]},
               {'T':0},{'T':1.5},{'N':0},{'N':True},{'seed':-1},{'graph':'unknown'},
               {'dt':0},{'dt':2},{'graph':'paper9','q':.1,'dt':1},
               {'N':10000000,'T':1000},{'q':1000001},{'q':10**1000},{'oops':1},
               {'name':False},{'q':{'v0':1}},{'graph':[]}, {'dt':None}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_config(value)

    def test_scalar_and_vector_q_agree(self):
        a = build_kernel(parse_config({'q':.2}))[0]
        b = build_kernel(parse_config({'q':[.2]*8}))[0]
        np.testing.assert_array_equal(a,b)


class ExactTests(unittest.TestCase):
    def test_hand_calculated_first_hit_and_return(self):
        K = np.array([[.5,.4],[.25,.5]])
        r = exact_first_passage(K,3,(0,1))
        np.testing.assert_allclose(r[:,0,1], [.4,.2,.1])
        np.testing.assert_allclose(r[:,1,1], [.5,.1,.05])
        hits, killed, censored = exact_outcomes(K,np.array([.1,.25]),3,(0,1))
        np.testing.assert_allclose(hits,r)
        self.assertAlmostEqual(killed[0,1], .175)
        self.assertAlmostEqual(censored[0,1], .125)
        np.testing.assert_allclose(hits.sum(axis=0)+killed+censored,1)

    def test_exhaustive_paths_including_cemetery(self):
        P = np.array([[.3,.5,.2],[.4,.5,.1],[0,0,1.]])
        r = exact_first_passage(P[:2,:2],4,(0,1))
        expected = np.zeros_like(r)
        for source in (0,1):
            for target in (0,1):
                for path in itertools.product(range(3),repeat=4):
                    probability = 1.
                    current = source
                    first = None
                    for t,destination in enumerate(path,1):
                        probability *= P[current,destination]
                        if destination == target and first is None:
                            first=t
                        current=destination
                    if first:
                        expected[first-1,source,target] += probability
        np.testing.assert_allclose(r,expected,atol=1e-15)

    def test_bipartite_parity_is_not_occupancy_or_time_zero(self):
        cfg=parse_config({'graph':'paper9','T':5})
        K,_,_=build_kernel(cfg)
        r=exact_first_passage(K,5,cfg.graph.observed)
        np.testing.assert_array_equal(r[::2,0,0],0)
        self.assertAlmostEqual(r[1,0,0],1/3)
        self.assertGreater(np.linalg.matrix_power(K,4)[0,0],r[3,0,0])

    def test_one_step_and_long_horizon_accounting(self):
        for graph in GRAPHS:
            for T in (1,17,200):
                cfg=parse_config({'graph':graph,'T':T,'N':100,'q':.3})
                K,kill,_=build_kernel(cfg)
                r=exact_first_passage(K,T,cfg.graph.observed)
                hits,killed,censored=exact_outcomes(K,kill,T,cfg.graph.observed)
                np.testing.assert_allclose(r,hits,atol=1e-14)
                np.testing.assert_allclose(r.sum(axis=0)+killed+censored,1,atol=1e-12)


class MonteCarloTests(unittest.TestCase):
    def test_mass_and_statistical_agreement(self):
        for graph in GRAPHS:
            cfg=parse_config({'graph':graph,'N':30000,'T':17,'q':.12,'seed':91})
            result=simulate(cfg)
            exact=np.array(result['r_exact']);sample=np.array(result['r_empirical'])
            tolerance=6*np.sqrt(exact*(1-exact)/cfg.N)+2/cfg.N
            self.assertTrue(np.all(np.abs(sample-exact)<=tolerance))
            for pair in result['pairs']:
                i=cfg.graph.observed.index(pair['source']);j=cfg.graph.observed.index(pair['target'])
                hits=int(np.array(result['counts'])[:,i,j].sum())
                self.assertEqual(hits+pair['killed_count']+pair['censored_count'],cfg.N)
                self.assertAlmostEqual(pair['hit_empirical'],hits/cfg.N)
                self.assertLess(abs(pair['no_hit_exact']-pair['no_hit_empirical']),.025)

    def test_seed_reproducibility_and_fresh_streams(self):
        cfg=parse_config({'N':1000,'T':8,'seed':123})
        a=simulate(cfg);b=simulate(cfg)
        self.assertEqual(a['counts'],b['counts'])
        self.assertEqual(a['pairs'],b['pairs'])
        c=simulate(parse_config({'N':1000,'T':8,'seed':124}))
        self.assertNotEqual(a['counts'],c['counts'])

    def test_no_arrival_norm_is_explicitly_undefined(self):
        graph=Graph('test','two vertices',1,2,((0,1),),(0,),1,(1.,1.),'test')
        cfg=Experiment(graph,1,10,(0.,0.),1,1.,'explicit','zero finite mass')
        result=simulate(cfg)
        self.assertIsNone(result['metrics']['frne_percent'])
        self.assertEqual(result['pairs'][0]['censored_count'],10)

    def test_cooperative_cancellation(self):
        with self.assertRaises(Cancelled):
            simulate(parse_config({'N':100}),cancelled=lambda:True)

    def test_checkpoints_and_progress(self):
        progress=[]
        result=simulate(parse_config({'N':1200,'T':4}),progress=progress.append)
        self.assertEqual([p['N'] for p in result['convergence']],[100,1000,1200])
        self.assertEqual(progress[-1]['fraction'],1)
        self.assertEqual(progress[-1]['completed_trials'],1200*9)
        self.assertAlmostEqual(result['convergence'][-1]['frne_percent'],result['metrics']['frne_percent'])

    def test_wilson_includes_uncertainty_at_zero_and_one(self):
        low,high=wilson(np.array([0,100]),100)
        self.assertAlmostEqual(low[0],0)
        self.assertGreater(high[0],0)
        self.assertLess(low[1],1)
        self.assertAlmostEqual(high[1],1)


if __name__=='__main__':
    unittest.main()
