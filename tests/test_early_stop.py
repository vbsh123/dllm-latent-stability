from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from test_observer import TinyModel
from dllm_latent.decoding import decode, finalized_stop_mask
from dllm_latent.gsm8k_rollout import output_lengths, summarize_tpf


class OutOfOrderEOS(TinyModel):
    def __init__(self,eos=5):super().__init__();self.eos=eos;self.inputs=[]
    def forward(self,x,attention_mask=None):
        self.inputs.append(x.clone())
        h=torch.full((*x.shape,4),10.)
        for block in self.model.transformer.blocks:h,_=block(h)
        logits=torch.full((*x.shape,7),-30.)
        logits[...,0]=np.log(.45);logits[...,1]=np.log(.55)
        logits[:,2,:]=-30.;logits[:,2,self.eos]=8.;logits[:,2,0]=0.
        if x[0,2]==self.eos:
            logits[:,1,:]=-30.;logits[:,1,1]=8.;logits[:,1,0]=0.
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize('method',['baseline','confidence','credit','latent','combined','no_geometry'])
@pytest.mark.parametrize('block',[4,8])
def test_stop_waits_for_prefix_then_skips_suffix_and_preserves_full_run_prefix(method,block):
    cfg=dict(gen_length=8,block_length=block,steps=8,mask_id=6,stop_token_ids=[5])
    prompt=torch.tensor([[5]])  # EOS in prompt must not terminate generation.
    full_model=OutOfOrderEOS();early_model=OutOfOrderEOS()
    full=decode(full_model,prompt,cfg,method=method,trace=False)
    early=decode(early_model,prompt,dict(cfg,early_stop=True),method=method,trace=False)
    assert early['forwards']==2 and early['eos_stopped'] and early['complete']
    assert early['stop_position']==1 and early['stop_reason']=='eos'
    assert early['unresolved_masks']==6 and early['unresolved_prefix_masks']==0
    assert torch.equal(early['tokens'][:,:3],full['tokens'][:,:3])
    assert full['forwards']>early['forwards'] and full['complete']
    assert all(torch.equal(a,b) for a,b in zip(early_model.inputs,full_model.inputs))
    assert early_model.inputs[1][0,1]==6 and early_model.inputs[1][0,2]==5
    assert not early_model.model.transformer.blocks[15]._forward_hooks


def test_raw_eos_prediction_is_not_a_committed_stop():
    cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6,early_stop=True,stop_token_ids=[5])
    result=decode(OutOfOrderEOS(),torch.tensor([[0]]),cfg,method='confidence',threshold=1.,fallback='none')
    assert not result['eos_stopped'] and not result['complete']
    assert result['stop_reason']=='budget' and result['unresolved_prefix_masks']==8


def test_multiple_stops_earliest_stop_and_zero_length_answer():
    g=torch.tensor([1,4,6,5])
    assert finalized_stop_mask(g,6,torch.tensor([4,5])).tolist()==[False,True,False,False]
    g=torch.tensor([5,6,6])
    assert finalized_stop_mask(g,6,torch.tensor([5])).tolist()==[True,False,False]
    assert not finalized_stop_mask(torch.tensor([6,5,1]),6,torch.tensor([5])).any()


def test_eot_id_can_terminate_and_no_stop_id_leaves_length_behavior():
    cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6,early_stop=True,stop_token_ids=[4,5])
    result=decode(OutOfOrderEOS(eos=4),torch.tensor([[0]]),cfg,method='confidence')
    assert result['eos_stopped'] and result['stop_position']==1
    result=decode(OutOfOrderEOS(),torch.tensor([[0]]),dict(cfg,stop_token_ids=[3]),method='confidence')
    assert result['complete'] and not result['eos_stopped'] and result['stop_reason']=='length'


@pytest.mark.parametrize('stops',[[],[6]])
def test_invalid_stop_configuration_rejected(stops):
    cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6,early_stop=True,stop_token_ids=stops)
    with pytest.raises(ValueError,match='stop_token_ids'):decode(OutOfOrderEOS(),torch.tensor([[0]]),cfg)


@pytest.mark.parametrize('method',['baseline','confidence','credit','latent','combined','no_geometry'])
def test_early_stop_trace_invariance(method):
    cfg=dict(gen_length=8,block_length=4,steps=8,mask_id=6,early_stop=True,stop_token_ids=[5])
    a=decode(OutOfOrderEOS(),torch.tensor([[0]]),cfg,method=method,trace=True)
    b=decode(OutOfOrderEOS(),torch.tensor([[0]]),cfg,method=method,trace=False)
    assert torch.equal(a['tokens'],b['tokens'])
    for key in ('counts','acceptance_counts','region_observation_counts','complete','forwards','stop_position','eos_stopped'):
        assert a[key]==b[key]


def test_tpf_counts_prefix_stop_suffix_and_failed_attempts_explicitly():
    lengths,end=output_lengths([1,5,2,6,6],6,{5})
    assert end==1 and lengths==dict(visible_output_tokens=1,output_tokens_with_stop=2,generated_span_tokens=3)
    records=[dict(**lengths,forwards=2,complete=True,early_stop_enabled=True,eos_stopped=True),
             dict(visible_output_tokens=9,output_tokens_with_stop=9,generated_span_tokens=9,forwards=8,
                  complete=False,early_stop_enabled=True,eos_stopped=False)]
    result=summarize_tpf(records)
    assert result['output_tpf']==.1 and result['output_tpf_with_stop']==.2
    assert result['full_span_tpf']==.3 and result['mean_output_tpf']==.25
    assert result['eos_stop_rate']==.5 and result['early_stop_enabled']
    del records[0]['output_tokens_with_stop']
    assert summarize_tpf(records)['output_tpf_with_stop'] is None


def test_offline_audit_reads_existing_records_without_model(tmp_path,monkeypatch):
    import sys
    import pandas as pd
    from dllm_latent.audit_tpf import main
    run=tmp_path/'old_run';run.mkdir();out=tmp_path/'audit.csv'
    records=[dict(id='x',method='latent',forwards=4,complete=True,visible_output_tokens=6,generated_span_tokens=8)]
    (run/'generations.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    monkeypatch.setattr(sys,'argv',['audit_tpf','--run',str(run),'--out',str(out)])
    main();frame=pd.read_csv(out)
    assert frame.output_tpf.iloc[0]==1.5 and frame.full_span_tpf.iloc[0]==2.
    assert pd.isna(frame.output_tpf_with_stop.iloc[0])
    assert not frame.early_stop_enabled.iloc[0]
