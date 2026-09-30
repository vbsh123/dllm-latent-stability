import json
import pytest
from dllm_latent.opencompass_score import upstream_functions, score_run, main
from test_evaluation_audit import fixture_run


def test_upstream_exact_extraction_edge_cases():
    api=upstream_functions();extract=api['gsm8k_postprocess'];eq=api['Gsm8kEvaluator']().is_equal
    # Preserve upstream behavior, even known limitations; do not quietly improve it.
    assert extract('#### 64 dollars.')=='64'
    assert extract('24 gallons in 2 weeks.')=='2'
    assert extract('1,200')=='200'
    assert extract('64\nQuestion: what is 2+2?')=='64'
    assert extract('No answer')=='NULL'
    assert extract('1/2')=='2'
    assert extract('1e3')=='3'
    assert eq('64.0','64') and not eq('2','24')
    assert api['gsm8k_dataset_postprocess']('work\n#### 1,200')=='1200'


def test_offline_upstream_score_and_incomplete_policy(tmp_path,monkeypatch):
    run,r=fixture_run(tmp_path)
    p=run/'manifest.json';m=json.loads(p.read_text());m['policy']={'latent':{'base_weight':2,'bonus_weight':1}};p.write_text(json.dumps(m))
    before=(run/'generations.jsonl').read_bytes()
    summaries,details=score_run(run,['no_geometry_double'])
    assert summaries[0]['opencompass_accuracy_percent']==100
    assert summaries[0]['legacy_strict_accuracy_percent']==0
    assert summaries[0]['base_weight']==2
    monkeypatch.setattr('sys.argv',['score','--run',str(run),'--out',str(tmp_path/'oc')])
    main();assert (run/'generations.jsonl').read_bytes()==before
    r['complete']=False;(run/'generations.jsonl').write_text(json.dumps(r)+'\n')
    summaries,_=score_run(run)
    assert summaries[0]['opencompass_accuracy_percent']==100
    assert summaries[0]['opencompass_completed_accuracy_percent']==0
    with pytest.raises(ValueError):score_run(run,['credit'])
