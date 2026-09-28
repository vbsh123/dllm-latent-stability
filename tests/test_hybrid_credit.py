import numpy as np
import pytest
import torch
from dllm_latent.decoding import (LatentSettings, RegionSupport, hybrid_distribution,
    no_geometry_distribution, update_token_credit, decode)
from test_decoding import FixedDistribution, ConstantLatentModel


def test_radius_bonus_manual_aaba_history_and_no_bonus_for_new_anchors():
    cfg=LatentSettings(radius=.05);h=torch.tensor([[10.,0.]])
    region=RegionSupport(h,cfg);balance=torch.zeros(1);active=torch.tensor([True])
    logits=torch.tensor([[.6,.4]]).log();expected=0.;w=.6**.2
    for value,hit in ((10.,False),(10.,True),(20.,False),(10.,True)):
        q,info=hybrid_distribution(logits,balance,active,cfg,'no_geometry_radius',
            region=region,hidden=torch.tensor([[value,0.]]))
        expected=.7*expected+w*(1+int(hit))
        assert bool(info['close'][0])==hit
        assert float(balance[0])==pytest.approx(expected)
        manual=np.array([.6*(1+expected)**.65,.4]);manual/=manual.sum()
        np.testing.assert_allclose(q[0].numpy(),manual,rtol=1e-6)
    assert region.count.tolist()==[2]
    assert float(balance[0])==pytest.approx(4.023*w)


def test_credit_hybrid_retains_token_specific_history_and_adds_position_credit():
    cfg=LatentSettings();active=torch.tensor([True]);position=torch.zeros(1);tokens=torch.zeros(1,2)
    reference_tokens=np.zeros(2);reference_position=0.
    for probs in ([.6,.4],[.4,.6],[.7,.3]):
        logits=torch.tensor([probs]).log();v=int(np.argmax(probs));w=max(probs)**.2
        reference_tokens*=.7;reference_tokens[v]+=w
        reference_position=.7*reference_position+w
        update_token_credit(logits,tokens,active)
        q,info=hybrid_distribution(logits,position,active,cfg,'no_geometry_credit',token_credit=tokens)
        evidence=reference_tokens.copy();evidence[v]+=reference_position
        manual=np.array(probs)*(1+evidence)**.65;manual/=manual.sum()
        np.testing.assert_allclose(tokens[0].numpy(),reference_tokens,rtol=1e-6)
        np.testing.assert_allclose(q[0].numpy(),manual,rtol=1e-6)
        assert float(info['support'][0])==pytest.approx(reference_position)


@pytest.mark.parametrize('method',['no_geometry_radius','no_geometry_credit','no_geometry_double'])
def test_zero_bonus_recovers_unrestricted_accumulation(method):
    cfg=LatentSettings(bonus_weight=0);active=torch.tensor([True]);a=torch.zeros(1);b=torch.zeros(1)
    token_credit=torch.zeros(1,2);region=RegionSupport(torch.tensor([[10.,0.]]),cfg)
    for probs,value in (([.6,.4],10.),([.4,.6],10.),([.7,.3],20.)):
        logits=torch.tensor([probs]).log();update_token_credit(logits,token_credit,active)
        q,_=hybrid_distribution(logits,a,active,cfg,method,region,torch.tensor([[value,0.]]),token_credit)
        reference,_=no_geometry_distribution(logits,b,active,cfg)
        torch.testing.assert_close(q,reference,rtol=0,atol=0)


def test_double_control_manual_and_inactive_balances_unchanged():
    cfg=LatentSettings();balance=torch.tensor([0.,3.]);active=torch.tensor([True,False])
    logits=torch.tensor([[.6,.4],[.7,.3]]).log()
    q,_=hybrid_distribution(logits,balance,active,cfg,'no_geometry_double')
    assert balance.tolist()==pytest.approx([2*.6**.2,3.])
    torch.testing.assert_close(q[1],logits[1].softmax(-1))


@pytest.mark.parametrize('method',['no_geometry_credit','no_geometry_double'])
def test_nongeometric_hybrids_need_no_hidden_access_and_reset_per_block(method):
    cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6)
    result=decode(FixedDistribution(),torch.tensor([[2]]),cfg,method=method,threshold=.7,fallback='none')
    assert result['complete'] and result['forwards']==2
    assert result['counts']=={method:8} and result['acceptance_counts']['boost_enabled']==8
    assert result['region_observation_counts'] is None
    assert all(e['block_step']==0 for e in result['commits'])


def test_radius_hybrid_first_step_does_not_get_automatic_double_credit():
    cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6)
    result=decode(ConstantLatentModel(),torch.tensor([[2]]),cfg,method='no_geometry_radius',threshold=.7,fallback='none')
    assert result['complete'] and result['forwards']==4
    assert result['counts']=={'no_geometry_radius':8}
    assert result['region_observation_counts']==dict(new=8,reused=8,invalid=0)
    assert all(e['block_step']==1 for e in result['commits'])


def test_bonus_weight_validation():
    for value in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):LatentSettings(bonus_weight=value)


def test_two_requested_hybrids_get_a_paired_comparison(tmp_path):
    import pandas as pd
    from dllm_latent.gsm8k_rollout import write_report
    records=[]
    for method in ('no_geometry_radius','no_geometry_credit'):
        for i in range(2):
            records.append(dict(id=str(i),method=method,correct=True,lenient_correct=True,
                complete=True,format_valid=True,forwards=2,seconds=1.,visible_output_tokens=8,
                generated_span_tokens=8,fallback_commits=0,total_commits=8,peak_gpu_bytes=1024,
                counts={method:8},commits=[]))
    summary=write_report(records,tmp_path,bootstrap=5).set_index('method')
    assert summary.loc['no_geometry_radius','no_geometry_radius_commits']==16
    assert summary.loc['no_geometry_credit','no_geometry_credit_commits']==16
    paired=pd.read_csv(tmp_path/'paired_comparisons.csv')
    assert list(zip(paired.reference,paired.method))==[('no_geometry_credit','no_geometry_radius')]
