import numpy as np
import pytest
import torch
from dllm_latent.decoding import LatentSettings,RegionSupport,region_distribution,decode
from dllm_latent.policy_observer import PolicyHistory
from dllm_latent.parity import select_parity_indices
from test_decoding import ConstantLatentModel


@pytest.mark.parametrize('values',[
    [10.]*7,
    [10.,10.,20.,10.,20.],
    [10.,10.4,10.8,11.2,11.6,10.],
    [10.,float('nan'),20.,10.],
])
def test_discovery_and_rollout_identical_bank_boost_and_acceptance(values):
    cfg=LatentSettings();observer=PolicyHistory(cfg,threshold=.7)
    rollout=RegionSupport(torch.tensor([[10.,0.]]),cfg)
    for t,value in enumerate(values):
        vector=np.array([[value,0]],dtype=np.float32)
        logits=torch.tensor([[.6,.4] if t%2==0 else [.4,.6]]).log()
        diagnostic=observer.update(vector,[True],logits)
        q,info=region_distribution(logits,torch.from_numpy(vector),torch.tensor([True]),rollout)
        assert diagnostic['policy_accept'][0]==bool(q.max()>=.7)
        assert diagnostic['policy_max_p'][0]==pytest.approx(float(q.max()))
        assert diagnostic['policy_top1'][0]==int(q.argmax())
        assert diagnostic['policy_support'][0]==pytest.approx(float(info['support'][0]))
        for key in ('anchor','scale','count','credit'):
            assert torch.equal(getattr(observer.state,key),getattr(rollout,key))


def test_no_warmup_first_visit_earns_paper_weighted_credit():
    state=PolicyHistory(threshold=.7);scores=[];accept=[]
    for _ in range(3):
        result=state.update([[10.,0.]],[True],torch.tensor([[.6,.4]]).log())
        scores.append(float(result['policy_support'][0]));accept.append(bool(result['policy_accept'][0]))
    assert scores==pytest.approx([.6**.2,1.7*.6**.2,2.19*.6**.2])
    assert accept==[False,True,True]


def test_inactive_positions_do_not_accumulate_or_create_regions():
    obs=PolicyHistory();logits=torch.tensor([[.6,.4]]).log()
    obs.update([[10.,0.]],[True],logits)
    before=obs.state.credit.clone()
    result=obs.update([[1000.,0.]],[False],logits)
    assert not result['policy_accept'][0]
    assert torch.equal(before,obs.state.credit) and int(obs.state.count[0])==1


@pytest.mark.parametrize('method',['baseline','confidence','credit','latent','combined','no_geometry','no_geometry_radius','no_geometry_credit','no_geometry_double'])
def test_trace_mode_never_changes_tokens_counters_or_forward_count(method):
    cfg=dict(gen_length=8,block_length=8,steps=8,mask_id=6)
    model=ConstantLatentModel();prompt=torch.tensor([[2]])
    a=decode(model,prompt,cfg,method=method,trace=True)
    b=decode(model,prompt,cfg,method=method,trace=False)
    assert torch.equal(a['tokens'],b['tokens'])
    assert a['acceptance_counts']==b['acceptance_counts']
    assert a['region_observation_counts']==b['region_observation_counts']
    assert a['counts']==b['counts'] and a['forwards']==b['forwards'] and a['complete']==b['complete']
    assert a['commits'] and b['commits']==[]
    assert len(a['commits'])==sum(a['counts'].values())


def test_parity_selection_covers_lengths_and_reports_small_smoke_size():
    lengths=list(range(40,0,-1))
    chosen=select_parity_indices(lengths,20)
    assert len(set(chosen))==20
    assert lengths[chosen[0]]==1 and lengths[chosen[-1]]==40
    assert select_parity_indices([10,2,50],20)==[1,0,2]
    with pytest.raises(ValueError):select_parity_indices([1],0)


def test_collected_shared_policy_decisions_feed_current_timeline_report(tmp_path,monkeypatch):
    import json,sys
    from dataclasses import asdict
    import pandas as pd
    from dllm_latent.collect import Observer
    from dllm_latent.third_party.llada_generate import generate
    from dllm_latent.policy_timelines import main
    run=tmp_path/'run';out=tmp_path/'analysis';run.mkdir()
    cfg=dict(steps=8,gen_length=8,block_length=8,layers=[16],final_norm=False,mask_id=6,
             latent_policy=asdict(LatentSettings()),policy_threshold=.7)
    (run/'manifest.json').write_text(json.dumps(dict(synthetic=True,config=cfg)))
    for g in range(3):
        model=ConstantLatentModel();prompt=torch.tensor([[2]])
        with Observer(model,cfg,1,dict(generation_id=str(g),prompt_group=str(g),task='test')) as obs:
            result=generate(model,prompt,**{k:cfg[k] for k in ('steps','gen_length','block_length','mask_id')})
            rows,_=obs.finish(result,[])
        pd.DataFrame(rows).to_parquet(run/f'{g}.parquet',index=False)
    monkeypatch.setattr(sys,'argv',['policy_timelines','--run',str(run),'--out',str(out),'--bootstrap','5'])
    main()
    events=pd.read_parquet(out/'lat_layer16_first_triggers.parquet')
    assert set(events.loc[events.observed,'trigger_step'])=={1}
    assert (out/'comparisons.csv').exists()
