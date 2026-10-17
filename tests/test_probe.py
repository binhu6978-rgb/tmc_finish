"""Small checks directly tied to the scientific definitions, not source hashes."""
import unittest
from pathlib import Path
import torch
from src.core import aggregate,delta,dispatch,substate,loo_cosine
from src.data import partition,evaluation_split
from src.semantic import Memory
from src.main import build_initial
from src.analyze import metrics,auc,incremental


class ProbeDefinitions(unittest.TestCase):
    def test_delta_and_unnormalized_aggregation(self):
        b={'w':torch.tensor([1.,2.,3.,4.])}
        d=[{'w':torch.tensor([2.,4.])},{'w':torch.tensor([6.,8.,10.])}]
        got=aggregate(b,d,[1,3])['w']
        torch.testing.assert_close(got,torch.tensor([6.,9.,10.5,4.]))
        # LOO recomputes n, rather than subtracting a weighted vector from got.
        torch.testing.assert_close(aggregate(b,d[:1],[1])['w'],torch.tensor([3.,6.,3.,4.]))
        self.assertEqual(float(got[-1]),4.)
        self.assertEqual(float(delta({'w':torch.tensor([3.])},{'w':torch.tensor([1.])})['w'][0]),2.)

    def test_protective_formula_and_runtime_identity(self):
        b={'w':torch.zeros(1)};d=[{'w':torch.ones(1)}]
        self.assertAlmostEqual(float(aggregate(b,d,[1])['w']),1.)
        self.assertAlmostEqual(float(aggregate(b,d,[1],gamma=[.01])['w']),.05,places=6)
        self.assertEqual(float(aggregate(b,d,[1],gamma=[0.])['w']),0.)

    def test_fixed_cached_partition(self):
        users=partition(Path(__file__).resolve().parents[1]);self.assertEqual(sum(map(len,users.values())),50000)

    def test_dispatch(self):
        a=dispatch(1,50);self.assertEqual(a,dispatch(1,50))
        self.assertTrue(any(len(set(r['clients']))<10 for r in a))
        for r in a:
            for p,c in zip(r['profiles'],r['clients']):
                self.assertTrue(c>= (0 if p<3 else 40 if p<6 else 70))

    def test_initial_backbones(self):
        a=build_initial(1);b=build_initial(1)
        self.assertEqual(len(a),7)
        for x,y in zip(a,b):
            for k in x:self.assertTrue(torch.equal(x[k],y[k]))
        for v in a:
            for k,t in substate(a[-1],v).items():self.assertEqual(t.shape,v[k].shape)

    def test_anchor_time_and_missing_history(self):
        cfg=dict(embedding_dim=128,anchor_ema=.9,reliability_window=5)
        m=Memory(cfg);p=torch.zeros(100,128);p[0,0]=1;n=torch.zeros(100,dtype=torch.long);n[0]=5;v=n>0
        self.assertIsNone(m.score(p,n,v)['original_score'])
        self.assertEqual(m.score(p,n,v)['fixed_score'],1.)
        rel,merged=m.reliability(0,[(1,p,n,v)])
        self.assertEqual(rel[1]['history_status'],'INSUFFICIENT_HISTORY')
        m.update(merged);self.assertEqual(m.score(p,n,v)['original_score'],1.)
        rel,_=m.reliability(1,[(1,p,n,v)])
        self.assertIsNone(rel[1]['reliability'])  # only ONE comparable anchor observation
        rel,_=m.reliability(2,[(1,p,n,v)])
        self.assertEqual(rel[1]['reliability'],1.)

    def test_evaluation_split(self):
        x=evaluation_split([i//100 for i in range(10000)])
        self.assertEqual(len(x['validation']),5000)
        self.assertFalse(set(x['validation'])&set(x['reserved']))

    def test_external_direction_and_utility_sign(self):
        d={'w':torch.tensor([1.,2.])}
        self.assertAlmostEqual(loo_cosine(d,[d],[10]),1.)
        base={'w':torch.zeros(2)}
        updates=[{'w':torch.tensor([2.,-2.])},{'w':torch.zeros(2)}]
        all_logits=aggregate(base,updates,[1,1])['w'].unsqueeze(0)
        without=aggregate(base,updates[1:],[1])['w'].unsqueeze(0)
        target=torch.tensor([0])
        utility=torch.nn.functional.cross_entropy(without,target)-torch.nn.functional.cross_entropy(all_logits,target)
        self.assertGreater(float(utility),0.)

    def test_prediction_statistics_boundaries(self):
        self.assertIsNone(auc([.1,.2],[1.,2.]))
        self.assertEqual(auc([0.,0.],[-1.,1.]),.5)
        self.assertEqual(auc([0.,1.],[-1.,1.]),1.)
        tiny=[dict(round=1,semantic_score=.8,utility_full=.001)]
        self.assertEqual(metrics(tiny,'semantic_score')['status'],'INSUFFICIENT_SAMPLES')
        self.assertEqual(incremental(tiny,'semantic_score')['status'],'INSUFFICIENT_SAMPLES')

    def test_round_blocked_prediction_on_controlled_data(self):
        import math
        rows=[]
        for t in range(6):
            for i in range(10):
                e=math.sin(3*i+t)
                rows.append(dict(round=t,semantic_score=e,utility_full=.001*e,
                    sample_count=500,update_norm=1.,loo_update_cosine=.2,width=.4,depth=2,profile=0))
        result=metrics(rows,'semantic_score')
        self.assertIsNotNone(result['spearman_round_bootstrap_95ci'])
        fit=incremental(rows,'semantic_score')
        self.assertEqual(fit['status'],'ROUND_HELD_OUT_EXPLORATORY')
        self.assertGreater(fit['mse_reduction'],0.)


if __name__=='__main__':unittest.main()
