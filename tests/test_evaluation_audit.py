import json
import math
from types import SimpleNamespace
import pytest
import torch
from dllm_latent.answer_audit import exact_number, extract_review_answer
from dllm_latent.audit_gsm8k import audit_run, main
from dllm_latent.gsm8k_rollout import grade
from dllm_latent.decoding import decode, LatentSettings


@pytest.mark.parametrize('text,answer',[
    ('#### 64 dollars.','64'),('Total cost = #### 64 \\]','64'),
    ('#### 1,200.00','1200'),(r'\boxed{\frac{1}{2}}','1/2'),
    ('#### -0.50','-1/2'),('#### 1e3','1000'),
    ('#### 12\nCorrection: #### 9 dollars.','9'),
])
def test_unambiguous_extraction(text,answer):
    r=extract_review_answer(text)
    assert r['candidate']==answer and not r['needs_review']


@pytest.mark.parametrize('text',[
    '#### 24 gallons in 2 weeks.', '#### 12 or 14', '#### 2 + 2 = 4',
    '#### 1,2', '#### 50%', '#### 6-8', '#### 1/0',
    '#### 12\n#### unknown', 'We calculated 64 earlier, but cannot finish.',
    'The answer is 64.', '#### 12 dollars. Actually this is wrong.',
])
def test_ambiguous_extraction_is_not_automatically_scored(text):
    assert extract_review_answer(text)['needs_review']


def test_reproduce_legacy_false_positive_and_false_negative():
    # Last-number parsing falsely credits an incidental number after a wrong answer.
    assert grade('#### 24 gallons in 2 weeks.','#### 2')['lenient_correct']
    assert not grade('#### 24 gallons in 2 weeks.','#### 24')['lenient_correct']
    # Strict skips the malformed final correction, and incorrectly credits an old answer.
    assert grade('#### 12\nCorrection: #### 9 dollars.','#### 12')['correct']
    assert extract_review_answer('#### 12\nCorrection: #### 9 dollars.')['candidate']=='9'
    assert exact_number('1.2E+3')=='1200'


def fixture_run(tmp_path):
    run=tmp_path/'run';run.mkdir()
    cfg=dict(gen_length=4,steps=4,mask_id=6,stop_token_ids=[5],early_stop=True)
    manifest=dict(config=cfg,question_ids=['q'],arguments=dict(methods=['no_geometry_double']))
    q=dict(id='q',question='Question',formatted='Question formatted',answer='Reasoning\n#### 24')
    r=dict(id='q',method='no_geometry_double',question=q['question'],formatted_prompt=q['formatted'],
        output='#### 24 dollars.',gold='24',correct=False,lenient_correct=True,complete=True,
        token_ids=[1,2,5,6],visible_output_tokens=2,output_tokens_with_stop=3,generated_span_tokens=3,
        forwards=2,seconds=1.,early_stop_enabled=True,eos_stopped=True,stop_position=2,
        unresolved_masks=1,counts=dict(no_geometry_double=3),total_commits=3,fallback_commits=0,
        acceptance_counts=dict(boost_enabled=2,already_confident=1,fallback=0,scheduled=0,changed_candidate=0))
    (run/'manifest.json').write_text(json.dumps(manifest))
    (run/'questions.jsonl').write_text(json.dumps(q)+'\n')
    (run/'generations.jsonl').write_text(json.dumps(r)+'\n')
    return run,r


def test_saved_result_audit_is_offline_and_preserves_originals(tmp_path,monkeypatch):
    run,r=fixture_run(tmp_path);before=(run/'generations.jsonl').read_bytes()
    rows,summary,issues=audit_run(run)
    assert not issues and rows[0]['automatic_correct']
    assert summary[0]['output_tpf']==1 and summary[0]['output_tps']==2
    monkeypatch.setattr('sys.argv',['audit','--run',str(run),'--out',str(tmp_path/'audit')])
    main()
    assert (run/'generations.jsonl').read_bytes()==before
    assert '24 dollars' in (tmp_path/'audit'/'review.csv').read_text()


@pytest.mark.parametrize('field,value,issue',[
    ('visible_output_tokens',4,'length_mismatch'),('forwards',0,'invalid_forward_count'),
    ('complete',False,'invalid_finalized_stop'),('gold','2','saved_gold_mismatch'),
    ('formatted_prompt','Leaked other input','prompt_snapshot_mismatch'),
    ('token_ids',[6,2,5,6],'invalid_finalized_stop'),
])
def test_corrupt_results_are_flagged(tmp_path,field,value,issue):
    run,r=fixture_run(tmp_path);r[field]=value
    (run/'generations.jsonl').write_text(json.dumps(r)+'\n')
    _,_,issues=audit_run(run)
    assert any(issue in i['issue'] for i in issues)


@pytest.mark.parametrize('strength',[1,3,6,12,36])
def test_high_strength_decoder_matches_closed_form_and_actual_forward_calls(strength):
    class CountingModel(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,x):
            self.calls+=1
            logits=torch.full((*x.shape,7),-30.)
            logits[...,0]=math.log(.2);logits[...,1]=math.log(.8)
            return SimpleNamespace(logits=logits)
    # Independent scalar acceptance oracle, including one-token fallback each failure.
    predicted=64
    for t in range(1,65):
        credit=strength*.8**.2*(1-.7**t)/(1-.7)
        factor=(1+credit)**.65
        boosted=.8*factor/(.2+.8*factor)
        if boosted>=.95:
            predicted=t;break
    model=CountingModel()
    result=decode(model,torch.tensor([[2]]),dict(gen_length=128,block_length=64,steps=128,mask_id=6),
        method='no_geometry_double',latent=LatentSettings(base_weight=1,bonus_weight=strength-1),trace=False)
    assert result['complete'] and result['forwards']==model.calls==2*predicted
    assert torch.all(result['tokens'][0,1:]==1)
    assert sum(result['acceptance_counts'][k] for k in ('boost_enabled','already_confident','fallback','scheduled'))==128


def test_wrong_saved_summary_is_detected(tmp_path):
    run,_=fixture_run(tmp_path)
    (run/'summary.csv').write_text('method,accuracy,output_tpf\nno_geometry_double,1,99\n')
    _,_,issues=audit_run(run)
    assert {i['issue'] for i in issues}=={'summary_mismatch:accuracy','summary_mismatch:output_tpf'}
