import numpy as np
import pandas as pd
from dllm_latent.features import LatentHistory,label_rows
from dllm_latent.stability import AnchoredRegionRule
from dllm_latent.timelines import first_events,Counts,joined_comparison,choose_rule,split_group


def trajectory():
    return pd.DataFrame(dict(generation_id=['g']*4,prompt_group=['p']*4,task=['test']*4,
            position=[0]*4,step=[0,1,2,3],top1=[1,2,2,2],credit_top1=[2,1,2,2],
            final_token=[2]*4,safe_to_commit=[0,1,1,1],stable_to_end=[0,1,1,1],
            committed_this_step=[False,False,False,True],before_final_eos=[True]*4))


def test_first_unsafe_trigger_is_not_replaced_by_later_safe_trigger():
    frame=trajectory(); event=first_events(frame,[True,True,True,True]).iloc[0]
    assert event.trigger_step==0 and not event.safe and event.early
    counts=Counts(); counts.add(pd.DataFrame([event]))
    assert counts.record()['early_unsafe']==1


def test_credit_timeline_uses_its_own_candidate_and_preserves_censoring():
    frame=trajectory()
    credit=first_events(frame,[True,False,True,True],'credit_top1','credit')
    assert credit.iloc[0].safe and credit.iloc[0].proposed_token==2
    latent=first_events(frame,[False,True,True,True])
    never=first_events(frame,[False]*4,'credit_top1','credit')
    joined=joined_comparison(latent,never,.95).iloc[0]
    assert joined.latent_with_credit_censored and np.isnan(joined.credit_step)
    assert joined.lead_lower_bound_if_credit_censored==2
    assert not joined.latent_before_credit


def test_acceptance_stays_aligned_when_input_rows_are_unsorted():
    frame=trajectory().iloc[[3,0,2,1]]
    event=first_events(frame,[False,False,True,True]).iloc[0]
    assert event.trigger_step==1


def test_commit_only_trigger_cannot_pass_tuning_precision_support():
    events=first_events(trajectory(),[False,False,False,True])
    c=Counts(); c.add(events)
    assert c.record()['early_selected']==0 and c.record()['precision'] is None


def test_tuning_can_abstain_and_uses_correct_advance_within_constraints():
    base=dict(early_selected=100,selection_prompts=12,confirmations=2,window=3)
    records=[dict(base,precision=.96,safe_lead_per_position=3.,radius=.1),
             dict(base,precision=.99,safe_lead_per_position=1.,radius=.025)]
    assert choose_rule(records,.97,50,10)['radius']==.025
    assert choose_rule(records,.999,50,10) is None
    assert split_group('p',1729)==split_group('p',1729)


def test_frozen_anchor_rejects_cumulative_drift_and_scale_chasing():
    history=LatentHistory()
    # Reference centroid10, scale10. Probe movement remains small per step,
    # but its center progressively moves away from the fixed initial anchor.
    for x in [10.,10.,10.,10.1,10.2,10.3]:
        features=history.update(np.array([[x,0.]],np.float32))
    rule=AnchoredRegionRule(.04,window=3,confirmations=3)
    assert features['w3_cos_min'][0]==1.
    assert features['a3_c3_distance'][0]<.04
    assert features['a3_c3_radius'][0]==0
    assert features['a3_c3_drift'][0]>.01
    assert not rule.evaluate(features)[0]


def test_frozen_anchor_accepts_compact_stationary_cloud_after_confirmation():
    history=LatentHistory(); rule=AnchoredRegionRule(.01,3,2)
    for i,x in enumerate([10.,10.001,9.999,10.,10.001]):
        f=history.update(np.array([[x,0.]],np.float32))
        assert rule.evaluate(f)[0]==(i==4)


def test_credit_label_can_differ_from_raw_safe_label():
    rows=[dict(position=0,step=0,top1=1,credit_top1=2)]
    label_rows(rows,[2])
    assert rows[0]['safe_to_commit']==0 and rows[0]['credit_safe_to_commit']==1


def test_later_block_timeline_uses_global_steps():
    frame=trajectory()
    frame['step']+=64
    frame['position']+=64
    event=first_events(frame,[False,True,True,True]).iloc[0]
    assert event.position==64 and event.trigger_step==65 and event.baseline_step==67
    assert event.lead_steps==2


def episode(values,rule):
    from dllm_latent.stability import episode_triggers
    tracker=LatentHistory(); records=[]
    for value in values:
        records.append(tracker.update(np.array([[value,0.]],np.float32)))
    stats={key:np.array([r[key][0] for r in records]) for key in records[0]}
    return episode_triggers(stats,np.zeros(len(values),int),np.arange(len(values)),rule)


def test_episode_jump_discards_anchor_and_requires_fresh_reference_and_confirmations():
    # Three reference vectors, one good confirmation, then a large jump.
    # The jump seeds a new region; five NEW observations are needed for K3+C2.
    hit=episode([10.,10.,10.,10.,20.,20.,20.,20.,20.],AnchoredRegionRule(.04,3,2))
    assert np.flatnonzero(hit).tolist()==[8]


def test_return_after_excursion_cannot_resume_old_confirmation_count():
    hit=episode([10.,10.,10.,10.,20.,10.,10.,10.,10.,10.,10.],AnchoredRegionRule(.04,3,2))
    assert np.flatnonzero(hit).tolist()==[10]


def test_later_jump_never_erases_an_earlier_trigger():
    hit=episode([10.]*5+[20.]*5,AnchoredRegionRule(.04,3,2))
    assert np.flatnonzero(hit).tolist()==[4]


def test_episode_drift_failure_resets_even_inside_distance_ball():
    # The probe is within r=.04 of scale10, but center displacement >r/4.
    # It must reset before a later nearby vector could complete confirmation.
    hit=episode([10.,10.,10.,10.35,10.35,10.35,10.35,10.35],AnchoredRegionRule(.04,3,2))
    assert np.flatnonzero(hit).tolist()==[7]


def test_episode_online_prefix_invariance_and_lexical_independence():
    rule=AnchoredRegionRule(.04,3,2)
    values=[10.,10.,10.,10.,20.,20.,20.,20.,20.]
    full=episode(values,rule)
    for n in range(1,len(values)+1):
        assert np.array_equal(episode(values[:n],rule),full[:n])
    # There is no top1 input: lexical A/B/A changes do not reset stable latent vectors.
    assert np.flatnonzero(episode([10.]*5,rule)).tolist()==[4]
