import numpy as np
import pandas as pd
from dllm_latent.analyze import split_groups, operating_point, decision_metrics, paired_bootstrap


def test_prompt_groups_do_not_leak():
    df=pd.DataFrame({'prompt_group':np.repeat([f'p{i}' for i in range(50)],7)})
    a=split_groups(df,3); b=split_groups(df.sample(frac=1,random_state=8),3)
    assert a==b
    assert len(set(a.values()))==3
    assert df.assign(split=df.prompt_group.map(a)).groupby('prompt_group').split.nunique().max()==1


def test_threshold_ties_and_unattainable_precision():
    y=np.array([1,1,0,0]); score=np.array([.9,.9,.8,.7])
    t=operating_point(y,score,.99,min_selected=2)
    assert t==.9 and decision_metrics(y,score,t)['recall']==1
    assert operating_point(y,score,.99,min_selected=3) is None
    assert decision_metrics(y,score,None)['precision'] is None


def test_cluster_bootstrap_identical_models_have_zero_gain():
    frame=pd.DataFrame({'prompt_group':['a','a','b','b','c','c']})
    y=np.array([1,0,1,0,1,0]); p=np.array([.8,.2,.7,.3,.9,.1]); thresholds={'0.97':.8}
    ci=paired_bootstrap(frame,y,p,p,thresholds,thresholds,20,1729)
    for name,v in ci.items():
        if 'precision_at' not in name:
            assert v['low']==v['high']==0
