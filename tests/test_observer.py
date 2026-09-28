from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from dllm_latent.collect import Observer, validate_config
from dllm_latent.third_party.llada_generate import generate


class Block(nn.Module):
    def forward(self,x):
        return x + .01*x.mean(dim=1,keepdim=True), None


class TinyModel(nn.Module):
    """Small CPU tensors, 32 trivial modules, no downloaded weights or tokenizer."""
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.transformer=nn.ModuleDict({'blocks':nn.ModuleList([Block() for _ in range(32)]),'ln_f':nn.LayerNorm(4)})
        self.config=SimpleNamespace(_name_or_path='synthetic-test')
        self.device=torch.device('cpu')

    def forward(self,x,attention_mask=None):
        pos=torch.arange(x.shape[1],dtype=torch.float32).unsqueeze(0)
        h=torch.stack([x.float(),pos.expand_as(x),torch.ones_like(x).float(),x.float()*.13],dim=-1)
        for block in self.model.transformer.blocks: h,_=block(h)
        h=self.model.transformer.ln_f(h)
        logits=torch.stack([h[...,0],h[...,1]*2,h[...,2]-.2,h[...,3]+.3,-h[...,1],h[...,0]-.8,torch.full_like(pos,-100)],dim=-1)
        return SimpleNamespace(logits=logits)


def test_upstream_observation_parity_and_commits():
    model=TinyModel().eval(); prompt=torch.tensor([[1,2]])
    cfg=dict(steps=8,gen_length=8,block_length=8,layers=list(range(1,33)),final_norm=True,mask_id=6)
    kwargs={k:cfg[k] for k in ['steps','gen_length','block_length','mask_id']}
    plain=generate(model,prompt,**kwargs)
    with Observer(model,cfg,2,dict(generation_id='test',prompt_group='test',task='test')) as obs:
        observed=generate(model,prompt,**kwargs)
        rows,final=obs.finish(observed,[])
    assert torch.equal(plain,observed)
    assert len(rows)==36  # full sequence: 8+7+...+1 still-masked positions
    assert sum(r['committed_this_step'] for r in rows)==8
    assert all(r['safe_to_commit']==1 for r in rows if r['committed_this_step'])
    assert all('lat_final_norm__cos_lag1' in r for r in rows)
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert {r['block'] for r in rows}=={0}
    assert any(r['step']==0 and r['position']==7 for r in rows)
    assert all('lat_layer01__cos_lag1' in r and 'lat_layer31__cos_lag1' in r for r in rows)


def test_hook_cleanup_on_exception():
    model=TinyModel()
    cfg=dict(steps=8,gen_length=8,block_length=4,layers=[8],final_norm=True,mask_id=6)
    with pytest.raises(RuntimeError):
        with Observer(model,cfg,2,{}): raise RuntimeError('test')
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_invalid_schedule_rejected():
    with pytest.raises(ValueError): validate_config(dict(gen_length=8,block_length=3,steps=8,layers=[8]))
    validate_config(dict(gen_length=8,block_length=4,steps=8,layers=[8]))
    with pytest.raises(ValueError): validate_config(dict(gen_length=8,block_length=4,steps=7,layers=[8]))


@pytest.mark.parametrize('steps',[4,8])
def test_block_observation_parity_resets_and_active_positions(steps):
    model=TinyModel().eval(); prompt=torch.tensor([[1,2]])
    cfg=dict(steps=steps,gen_length=8,block_length=4,layers=list(range(1,33)),final_norm=True,mask_id=6)
    kwargs={k:cfg[k] for k in ['steps','gen_length','block_length','mask_id']}
    plain=generate(model,prompt,**kwargs)
    with Observer(model,cfg,2,dict(generation_id='blocks',prompt_group='blocks',task='test')) as obs:
        observed=generate(model,prompt,**kwargs)
        rows,_=obs.finish(observed,[])
    assert torch.equal(plain,observed)
    assert sum(r['committed_this_step'] for r in rows)==8
    for block in (0,1):
        block_rows=[r for r in rows if r['block']==block]
        assert {r['position'] for r in block_rows}==set(range(block*4,(block+1)*4))
        first=[r for r in block_rows if r['step']==block*steps//2]
        assert len(first)==4
        assert all(r['out_history_count']==0 and r['token_history']=='' for r in first)
        assert all(np.isnan(r['lat_layer16__cos_lag1']) for r in first)
        assert all(np.isnan(r['lat_layer16__a2_c2_distance']) for r in first)
        # Credit starts fresh in each block, with exactly this observation's increment.
        assert all(np.isclose(r['out_credit'],r['p1']**.2) for r in first)
    assert all(r['safe_to_commit']==1 for r in rows if r['committed_this_step'])


def test_base_prompt_has_no_chat_template():
    from dllm_latent.prompt_format import render_prompt
    assert render_prompt('Why?',{'prompt_format':'plain_qa'})=='Question: Why?\nAnswer:'
    assert render_prompt('A story begins',{'prompt_format':'raw'})=='A story begins'


def test_instruct_uses_tokenizer_template_and_generation_prompt():
    from dllm_latent.prompt_format import render_prompt
    class Tokenizer:
        chat_template='checkpoint template'
        def apply_chat_template(self,messages,tokenize,add_generation_prompt):
            assert messages==[{'role':'user','content':'Why?'}]
            assert not tokenize and add_generation_prompt
            return 'formatted user and assistant prefix'
    assert render_prompt('Why?',{'prompt_format':'chat'},Tokenizer())=='formatted user and assistant prefix'
    with pytest.raises(ValueError,match='pinned tokenizer'):
        render_prompt('Why?',{'prompt_format':'chat'})


def test_cpu_collection_refuses_before_model_loading(monkeypatch):
    import sys
    from dllm_latent.collect import main
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(sys,'argv',['collect','--out','unused'])
    with pytest.raises(SystemExit,match='requires a CUDA GPU'):
        main()
