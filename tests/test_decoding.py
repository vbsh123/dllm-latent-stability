from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from test_observer import TinyModel
from dllm_latent.decoding import LatentSettings,RegionSupport,credit_distribution,region_distribution,decode
from dllm_latent.features import TraceCredit,softmax
from dllm_latent.third_party.llada_generate import generate
from dllm_latent.gsm8k_rollout import extract_answer,grade,write_report


def settings(**kw):
    return LatentSettings(**dict(dict(radius=.1),**kw))


def track(values,s):
    state=RegionSupport(torch.tensor([[10.,0.]]),s);history=[]
    for x in values:
        support,info=state.update(torch.tensor([[x,0.]]),torch.tensor([True]),torch.tensor([.6]))
        history.append((float(support[0]),int(info['region_id'][0])))
    return state,history


def test_returning_region_retains_credit_and_new_region_also_earns_credit():
    state,history=track([10,10,20,10],settings())
    hit=.6**.2
    assert [r for _,r in history]==[0,0,1,0]
    assert [c for c,_ in history]==pytest.approx([hit,1.7*hit,hit,1.833*hit])
    assert state.credit[:2,0].tolist()==pytest.approx([1.833*hit,.7*hit])
    assert state.anchor[:2,0,0].tolist()==[10,20]
    state.update(torch.tensor([[20.,0.]]),torch.tensor([True]),torch.tensor([.6]))
    assert state.credit[:2,0].tolist()==pytest.approx([.7*1.833*hit,1.49*hit])


def test_fixed_anchors_prevent_walking_drift_and_nearest_region_gets_one_hit():
    state,history=track([10,10.5,11,11.5,12,10],settings())
    assert [r for _,r in history]==[0,0,0,1,1,0]
    assert state.anchor[:2,0,0].tolist()==[10,11.5]
    # 10.8 lies in both regions: .7/11.5 < .8/10, so B alone receives credit.
    _,info=state.update(torch.tensor([[10.8,0.]]),torch.tensor([True]),torch.tensor([.6]))
    assert int(info['region_id'][0])==1


def test_invalid_vectors_decay_but_do_not_destroy_saved_regions():
    state,history=track([10,float('nan'),0,float('inf'),10],settings())
    assert [r for _,r in history]==[0,-1,-1,-1,0]
    assert int(state.count[0])==1
    assert history[-1][0]==pytest.approx((1+.7**4)*.6**.2)


def test_legacy_settings_rejected_and_new_parameters_validated():
    with pytest.raises(TypeError):LatentSettings(window=3)
    for kw in (dict(radius=0),dict(decay=1.1),dict(alpha=-1),dict(gamma=0),dict(radius=float('nan'))):
        with pytest.raises(ValueError):settings(**kw)


def test_regional_evidence_boosts_current_candidate_across_token_changes():
    h=torch.tensor([[10.,0.]])
    state=RegionSupport(h,settings())
    for t,probs in enumerate(([.6,.4],[.4,.6],[.6,.4])):
        logits=torch.tensor([probs]).log()
        q,info=region_distribution(logits,h,torch.tensor([True]),state)
        support=.6**.2*sum(.7**j for j in range(t+1))
        v=int(np.argmax(probs));evidence=np.zeros(2);evidence[v]=support
        expected=np.array(probs)*(1+evidence)**.65;expected/=expected.sum()
        np.testing.assert_allclose(q.numpy()[0],expected,rtol=1e-6)
        assert float(info['support'][0])==pytest.approx(support)
        assert int(state.count[0])==1


def test_combined_is_sum_before_log_fusion_not_or_of_gates():
    h=torch.tensor([[10.,0.]]);logits=torch.tensor([[.6,.4]]).log()
    token_credit=torch.tensor([[0.,4.]])
    q,info=region_distribution(logits,h,torch.tensor([True]),RegionSupport(h,settings()),token_credit)
    evidence=np.array([.6**.2,4.]);expected=np.array([.6,.4])*(1+evidence)**.65;expected/=expected.sum()
    np.testing.assert_allclose(q.numpy()[0],expected,rtol=1e-6)
    assert q.argmax()==1  # Combined can select the historical token, not current raw top1.


def test_gpu_credit_equations_match_existing_independent_numpy_version():
    logits=np.array([[1.,2.,0.],[2.,1.,0.]],dtype=np.float32)
    state=torch.zeros(2,3);reference=TraceCredit((2,3))
    for active in ([True,True],[True,False],[True,True]):
        q=credit_distribution(torch.tensor(logits),state,torch.tensor(active))
        expected,_=reference.update(softmax(logits),np.array(active))
        np.testing.assert_allclose(q.numpy(),expected,rtol=2e-6,atol=1e-7)
        np.testing.assert_allclose(state.numpy(),reference.credit,rtol=2e-6)
        logits=logits[:,::-1].copy()


@pytest.mark.parametrize('block,steps',[(8,8),(4,8),(4,4)])
def test_actual_baseline_matches_unmodified_upstream(block,steps):
    model=TinyModel().eval();prompt=torch.tensor([[1,2]])
    cfg=dict(gen_length=8,block_length=block,steps=steps,mask_id=6)
    expected=generate(model,prompt,**cfg)
    actual=decode(model,prompt,cfg,method='baseline')
    assert torch.equal(expected,actual['tokens']) and actual['complete']
    assert actual['forwards']==steps
    assert len(actual['commits'])==8


class FeedbackModel(nn.Module):
    def __init__(self):super().__init__();self.inputs=[]
    def forward(self,x):
        self.inputs.append(x.clone())
        logits=torch.full((*x.shape,7),-30.)
        logits[...,0]=0
        logits[:,1,1]=6.;logits[:,2,2]=5.
        ready=bool(x[0,1]==1 and x[0,2]==2)
        logits[:,3:,3 if ready else 4]=6. if ready else 1.
        return SimpleNamespace(logits=logits)


def test_policy_changes_actual_next_forward_inputs_and_saves_forwards():
    cfg=dict(gen_length=4,block_length=4,steps=4,mask_id=6);prompt=torch.tensor([[0]])
    baseline=FeedbackModel();fast=FeedbackModel()
    b=decode(baseline,prompt,cfg,method='baseline')
    c=decode(fast,prompt,cfg,method='confidence',threshold=.95)
    assert b['forwards']==4 and c['forwards']==2 and c['complete']
    assert baseline.inputs[1][0,2]==6 and fast.inputs[1][0,2]==2


def test_budget_failure_is_explicit_not_silently_filled():
    model=TinyModel();cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6)
    result=decode(model,torch.tensor([[1,2]]),cfg,method='confidence',threshold=1.,fallback='none')
    assert not result['complete'] and result['unresolved_masks']==8 and result['forwards']==4
    assert result['unchanged_context_steps']==4
    assert not result['commits']


def test_latent_hooks_removed_and_fallback_counted():
    model=TinyModel();cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6)
    result=decode(model,torch.tensor([[1,2]]),cfg,method='latent',latent=settings(),threshold=1.)
    assert result['complete'] and result['counts']['fallback']==8
    assert not model.model.transformer.blocks[15]._forward_hooks


def test_answer_grading_allows_different_wording_but_not_missing_or_fraction_marker():
    assert grade('A different explanation.\n#### 1,200.00','Solution\n#### 1200')['correct']
    assert not grade('The answer is 1200','#### 1200')['correct']
    assert grade('The answer is 1200','#### 1200')['lenient_correct']
    assert not grade('#### 1200','#### 1200',complete=False)['correct']
    assert extract_answer('#### 1/2') is None
    assert extract_answer('#### 1e3') is None
    assert extract_answer('#### 2\nCorrection\n#### 3')=='3'


def test_gpu_guard_before_hub_or_dataset_load(monkeypatch,tmp_path):
    import sys
    from dllm_latent.gsm8k_rollout import main
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(sys,'argv',['rollout','--out',str(tmp_path)])
    with pytest.raises(SystemExit,match='require CUDA'):main()


def test_actual_report_keeps_errors_and_paired_contexts(tmp_path):
    records=[]
    for i in range(3):
        for method in ('baseline','credit','latent'):
            records.append(dict(id=str(i),method=method,output='#### 2',correct=not(method=='latent' and i==0),lenient_correct=True,
                complete=not(method=='latent' and i==0),format_valid=True,forwards=4 if method=='baseline' else 2,
                seconds=float(i+1),visible_output_tokens=2+3*i,generated_span_tokens=10,fallback_commits=1,total_commits=4,
                peak_gpu_bytes=1024,commits=[dict(step=0,position=0,token=2,token_text='2',reason=method)]))
    summary=write_report(records,tmp_path,bootstrap=10).set_index('method')
    assert summary.loc['latent','accuracy']==pytest.approx(2/3)
    assert summary.loc['baseline','output_tps']==pytest.approx(15/6)
    assert summary.loc['latent','output_tps']==pytest.approx(13/6)
    assert summary.loc['baseline','full_span_tps']==pytest.approx(30/6)
    assert (tmp_path/'paired_commitments.csv').exists()
    assert (tmp_path/'accuracy_vs_forwards.png').exists()


class FixedDistribution(nn.Module):
    def forward(self,x):
        logits=torch.full((*x.shape,7),-30.)
        logits[...,0]=np.log(.4);logits[...,1]=np.log(.6)
        return SimpleNamespace(logits=logits)


def test_actual_credit_threshold_commits_and_resets_credit_per_block():
    cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6);prompt=torch.tensor([[2]])
    cd=decode(FixedDistribution(),prompt,cfg,method='credit',threshold=.7,fallback='none')
    confidence=decode(FixedDistribution(),prompt,cfg,method='confidence',threshold=.7,fallback='none')
    assert cd['complete'] and cd['forwards']==4 and cd['counts']=={'credit':8}
    assert all(e['block_step']==1 and e['score']>=.7 for e in cd['commits'])
    assert not confidence['complete'] and not confidence['commits']


def test_zero_decay_retains_regions_but_only_current_hit_credit():
    state,history=track([10,20,10],settings(decay=0))
    assert [r for _,r in history]==[0,1,0]
    assert state.credit[:2,0].tolist()==pytest.approx([.6**.2,0])


class TiedFeedback(nn.Module):
    def __init__(self):
        super().__init__();self.inputs=[];self.device=torch.device('cpu')
    def forward(self,x,attention_mask=None):
        self.inputs.append(x.clone())
        logits=torch.full((*x.shape,7),-30.)
        logits[...,0]=np.log(.4)
        n=int((x[:,1:]!=6).sum())
        for pos in range(x.shape[1]):logits[:,pos,1+(pos+n)%4]=np.log(.6)
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize('block',[2,4])
def test_upstream_parity_includes_equal_confidence_commit_order(block):
    cfg=dict(gen_length=4,block_length=block,steps=4,mask_id=6);prompt=torch.tensor([[0]])
    original=TiedFeedback();ours=TiedFeedback()
    expected=generate(original,prompt,**cfg)
    actual=decode(ours,prompt,cfg,method='baseline')['tokens']
    assert torch.equal(expected,actual)
    assert all(torch.equal(a,b) for a,b in zip(original.inputs,ours.inputs))


class ConstantLatentModel(TinyModel):
    def forward(self,x,attention_mask=None):
        h=torch.full((*x.shape,4),10.)
        for block in self.model.transformer.blocks:h,_=block(h)
        logits=torch.full((*x.shape,7),-30.)
        logits[...,0]=np.log(.4);logits[...,1]=np.log(.6)
        return SimpleNamespace(logits=logits)


def test_latent_logit_boost_commits_multiple_tokens_at_confidence_threshold():
    cfg=dict(gen_length=8,block_length=8,steps=8,mask_id=6)
    result=decode(ConstantLatentModel(),torch.tensor([[2]]),cfg,method='latent',threshold=.7,fallback='none')
    assert result['complete'] and result['forwards']==2
    assert result['counts']=={'latent':8}
    assert all(e['score']>=.7 and e['latent_support']==pytest.approx(1.7*.6**.2) for e in result['commits'])


def test_many_hits_do_not_bypass_confidence_threshold():
    cfg=dict(gen_length=8,block_length=8,steps=8,mask_id=6)
    result=decode(ConstantLatentModel(),torch.tensor([[2]]),cfg,method='latent',threshold=.95,fallback='none')
    assert not result['complete'] and result['forwards']==8 and not result['commits']


def test_stationary_single_candidate_region_matches_credit_decoding_exactly():
    cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6)
    args=dict(threshold=.7,fallback='none')
    credit=decode(ConstantLatentModel(),torch.tensor([[2]]),cfg,method='credit',**args)
    latent=decode(ConstantLatentModel(),torch.tensor([[2]]),cfg,method='latent',**args)
    assert torch.equal(credit['tokens'],latent['tokens'])
    assert credit['forwards']==latent['forwards']==4
    assert [e['score'] for e in credit['commits']]==[e['score'] for e in latent['commits']]


def test_region_bank_growth_never_evicts_and_positions_are_independent():
    state=RegionSupport(torch.tensor([[1.,0.],[1.,0.]]),settings())
    for value in (1.,2.,4.,8.,16.,32.):
        state.update(torch.tensor([[value,0.],[1.,0.]]),torch.tensor([True,True]),torch.tensor([.6,.8]))
    support,info=state.update(torch.tensor([[1.,0.],[1.,0.]]),torch.tensor([True,True]),torch.tensor([.6,.8]))
    assert state.count.tolist()==[6,1]
    assert info['region_id'].tolist()==[0,0]
    assert float(support[0])==pytest.approx((1+.7**6)*.6**.2)
    assert float(support[1])==pytest.approx(sum(.7**j for j in range(7))*.8**.2)
    assert state.anchor[:6,0,0].tolist()==[1,2,4,8,16,32]


def test_zero_fusion_strength_recovers_raw_distribution():
    logits=torch.tensor([[.2,.5,.3]]).log();h=torch.tensor([[10.,0.]])
    state=RegionSupport(h,settings(alpha=0))
    for _ in range(5):q,_=region_distribution(logits,h,torch.tensor([True]),state)
    torch.testing.assert_close(q,torch.softmax(logits,dim=-1),rtol=0,atol=0)


class FallbackRankingModel(TinyModel):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x,attention_mask=None):
        h=torch.full((*x.shape,4),10.)
        if self.calls: h[:,3]=20.
        for block in self.model.transformer.blocks:h,_=block(h)
        probs=torch.tensor([.6,.7,.6,.61 if self.calls else .6,.55])
        logits=torch.full((*x.shape,7),-30.)
        logits[...,0]=(1-probs).log();logits[...,1]=probs.log()
        self.calls+=1
        return SimpleNamespace(logits=logits)


def test_latent_fallback_ranks_boosted_not_raw_confidence():
    cfg=dict(gen_length=4,block_length=4,steps=4,mask_id=6)
    result=decode(FallbackRankingModel(),torch.tensor([[2]]),cfg,method='latent',
                  threshold=1.,latent=settings(layer=1),fallback='top1')
    assert result['complete'] and result['counts']=={'fallback':4}
    # Second step: stable .6 beats new-region .61 after the regional logit boost.
    assert [e['position'] for e in result['commits'][:2]]==[0,1]
    assert all(e['region_id']>=0 for e in result['commits'])


@pytest.mark.parametrize('method',['credit','latent','combined'])
def test_boost_attribution_separates_new_crossings_from_already_confident(method):
    cfg=dict(gen_length=8,block_length=8,steps=8,mask_id=6)
    prompt=torch.tensor([[2]])
    boosted=decode(ConstantLatentModel(),prompt,cfg,method=method,threshold=.7,fallback='none')
    assert boosted['complete']
    assert boosted['acceptance_counts']==dict(boost_enabled=8,already_confident=0,fallback=0,scheduled=0,changed_candidate=0)
    assert all(e['raw_max_p']<.7<=e['score'] and e['boost_enabled'] for e in boosted['commits'])
    already=decode(ConstantLatentModel(),prompt,cfg,method=method,threshold=.5,fallback='none')
    assert already['acceptance_counts']==dict(boost_enabled=0,already_confident=8,fallback=0,scheduled=0,changed_candidate=0)
    assert not any(e['boost_enabled'] for e in already['commits'])


def test_fallback_and_scheduled_commits_are_not_credited_to_boost():
    cfg=dict(gen_length=8,block_length=8,steps=8,mask_id=6)
    for method,category in [('baseline','scheduled'),('credit','fallback'),('latent','fallback')]:
        result=decode(ConstantLatentModel(),torch.tensor([[2]]),cfg,method=method,threshold=1.)
        assert result['acceptance_counts'][category]==8
        assert result['acceptance_counts']['boost_enabled']==0
        assert result['acceptance_counts']['already_confident']==0
        assert not any(e['boost_enabled'] for e in result['commits'])


def test_winner_changes_reported_separately_from_position_threshold_crossings(monkeypatch):
    import dllm_latent.decoding as decoding
    def changed_distribution(logits,credit,active):
        q=torch.zeros_like(logits);q[:,0]=.99;q[:,1]=.01
        return q
    monkeypatch.setattr(decoding,'credit_distribution',changed_distribution)
    cfg=dict(gen_length=8,block_length=8,steps=8,mask_id=6)
    result=decode(ConstantLatentModel(),torch.tensor([[2]]),cfg,method='credit',threshold=.5)
    assert result['acceptance_counts']['boost_enabled']==0
    assert result['acceptance_counts']['already_confident']==8
    assert result['acceptance_counts']['changed_candidate']==8
    assert all(e['raw_top1']==1 and e['token']==0 and e['raw_selected_p']==pytest.approx(.4) for e in result['commits'])


def test_mask_valued_proposals_are_not_counted_as_actual_boost_commits(monkeypatch):
    import dllm_latent.decoding as decoding
    def mask_distribution(logits,credit,active):
        q=torch.zeros_like(logits);q[:,6]=1.
        return q
    monkeypatch.setattr(decoding,'credit_distribution',mask_distribution)
    cfg=dict(gen_length=8,block_length=8,steps=8,mask_id=6)
    result=decode(ConstantLatentModel(),torch.tensor([[2]]),cfg,method='credit',threshold=.95,fallback='none')
    assert not result['complete'] and result['unresolved_masks']==8
    assert sum(result['acceptance_counts'].values())==0


def test_boost_summary_fractions_and_legacy_missing_data():
    from dllm_latent.gsm8k_rollout import summarize_boost_attribution
    records=[dict(acceptance_counts=dict(boost_enabled=4,already_confident=3,fallback=1,scheduled=0,changed_candidate=2))]
    summary=summarize_boost_attribution(records)
    assert summary['boost_attribution_available']
    assert summary['boost_enabled_commits']==4
    assert summary['boost_enabled_fraction']==.5
    assert summary['boost_enabled_fraction_of_threshold']==pytest.approx(4/7)
    assert summary['boost_changed_candidate_commits']==2
    old=summarize_boost_attribution([{}])
    assert not old['boost_attribution_available'] and old['boost_enabled_commits'] is None
    mixed=summarize_boost_attribution(records+[{}])
    assert not mixed['boost_attribution_available'] and mixed['boost_enabled_fraction'] is None


def test_no_geometry_accumulates_across_token_changes_with_manual_formula():
    from dllm_latent.decoding import no_geometry_distribution
    credit=torch.zeros(2);cfg=settings();expected=0.
    for probs in ([.6,.4],[.2,.8],[.7,.3]):
        logits=torch.tensor([probs,[.5,.5]]).log()
        q,info=no_geometry_distribution(logits,credit,torch.tensor([True,False]),cfg)
        expected=.7*expected+max(probs)**.2
        evidence=np.zeros(2);evidence[np.argmax(probs)]=expected
        expected_q=np.array(probs)*(1+evidence)**.65;expected_q/=expected_q.sum()
        assert float(credit[0])==pytest.approx(expected)
        assert float(credit[1])==0
        np.testing.assert_allclose(q[0].numpy(),expected_q,rtol=1e-6)
        assert info['region_id'].tolist()==[-1,-1]


def test_no_geometry_matches_region_fusion_when_all_states_match():
    from dllm_latent.decoding import no_geometry_distribution
    cfg=settings();h=torch.tensor([[10.,0.]])
    region=RegionSupport(h,cfg);credit=torch.zeros(1);active=torch.tensor([True])
    for probs in ([.6,.4],[.4,.6],[.9,.1],[.3,.7]):
        logits=torch.tensor([probs]).log()
        a,_=region_distribution(logits,h,active,region)
        b,_=no_geometry_distribution(logits,credit,active,cfg)
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_no_geometry_needs_no_hidden_access_and_resets_per_block():
    cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6)
    # This fake model has no transformer or hidden-state hook interface at all.
    result=decode(FixedDistribution(),torch.tensor([[2]]),cfg,method='no_geometry',threshold=.7,fallback='none')
    assert result['complete'] and result['forwards']==4
    assert result['counts']=={'no_geometry':8}
    assert result['acceptance_counts']['boost_enabled']==8
    assert result['region_observation_counts'] is None
    assert all(e['block_step']==1 and e['region_id']==-1 for e in result['commits'])


def test_region_creation_reuse_counts_and_missing_summary():
    from dllm_latent.gsm8k_rollout import summarize_region_observations
    cfg=dict(gen_length=8,block_length=8,steps=8,mask_id=6)
    result=decode(ConstantLatentModel(),torch.tensor([[2]]),cfg,method='latent',threshold=.7,fallback='none')
    assert result['region_observation_counts']==dict(new=8,reused=8,invalid=0)
    summary=summarize_region_observations([result])
    assert summary['new_region_fraction']==.5
    assert summary['new_region_observations']==8
    old=summarize_region_observations([{}])
    assert not old['region_observations_available'] and old['new_region_fraction'] is None


def test_no_geometry_comparison_is_in_report(tmp_path):
    import pandas as pd
    records=[]
    for method in ('no_geometry','latent'):
        for i in range(2):
            records.append(dict(id=str(i),method=method,correct=True,lenient_correct=True,
                complete=True,format_valid=True,forwards=2,seconds=1.,visible_output_tokens=8,
                generated_span_tokens=8,fallback_commits=0,total_commits=8,peak_gpu_bytes=1024,commits=[]))
    write_report(records,tmp_path,bootstrap=5)
    paired=pd.read_csv(tmp_path/'paired_comparisons.csv')
    assert list(zip(paired.reference,paired.method))==[('no_geometry','latent')]
