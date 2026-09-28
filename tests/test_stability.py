import numpy as np
import pandas as pd
import pytest
from dllm_latent.stability import StabilityChecker, StabilityRule
from dllm_latent.analyze import partition,summarize,raw_score_metrics,bootstrap_gain


def test_fixed_checker_warmup_and_movement():
    checker=StabilityChecker(StabilityRule(3,.99))
    for _ in range(3):
        accepted,_=checker.update(np.array([[1.,0.]],np.float32))
        assert not accepted[0]
    accepted,_=checker.update(np.array([[1.,0.]],np.float32))
    assert accepted[0]
    accepted,_=checker.update(np.array([[0.,1.]],np.float32))
    assert not accepted[0]


def test_missing_history_and_zero_vectors_do_not_pass():
    checker=StabilityChecker()
    for _ in range(6): accepted,_=checker.update(np.zeros((2,3),np.float32))
    assert not accepted.any()
    with pytest.raises(ValueError): StabilityRule(1,.99)


def test_fixed_threshold_does_not_adapt_to_inputs():
    rule=StabilityRule(2,.99)
    np.testing.assert_array_equal(rule.evaluate({'w2_cos_min':[.989,.99,.995,np.nan]}),[False,True,True,False])
    assert rule.cosine_threshold==.99


def test_groups_keep_partition_and_empty_selection_is_explicit():
    assert partition('same prompt',1729)==partition('same prompt',1729)
    m=summarize([1,0],[False,False])
    assert m['precision'] is None and m['recall']==0
    assert 'brier' not in raw_score_metrics([1,0],[.99,.8])


def test_identical_fixed_rules_have_zero_bootstrap_gain():
    df=pd.DataFrame({'prompt_group':['a','a','b','b'],'safe_to_commit':[1,0,1,0]})
    accept=np.array([True,False,True,False])
    ci=bootstrap_gain(df,accept,accept,'safe_to_commit',10,1729)
    assert ci['recall_gain_low']==ci['recall_gain_high']==0


def test_region_rule_rejects_moving_center_even_if_point_is_inside():
    from dllm_latent.stability import RegionRule
    rule=RegionRule(max_distance=1.,max_radius=.2,max_drift=.05)
    assert not rule.evaluate({'w3_raw_center_distance':np.array([.3]),'w3_raw_radius':np.array([.1]),'w3_raw_center_drift':np.array([.2])})[0]
    with pytest.raises(ValueError): RegionRule(-1.,.2,.1)
