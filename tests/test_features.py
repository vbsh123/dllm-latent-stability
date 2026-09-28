import numpy as np
from dllm_latent.features import TraceCredit, OutputHistory, LatentHistory, label_rows


def test_credit_equations_and_returning_candidate():
    history = [np.array([[.6,.3,.1]],np.float32),np.array([[.2,.7,.1]],np.float32),np.array([[.55,.35,.1]],np.float32)]
    tracker = TraceCredit((1,3)); dense = np.zeros((1,3))
    for p in history:
        v = p.argmax(1)[0]; dense = .7*dense; dense[0,v] += p[0,v]**.2
        logits = np.log(p)+.65*np.log1p(dense)
        ref = np.exp(logits-logits.max()); ref /= ref.sum()
        q,c = tracker.update(p,np.array([True]))
        np.testing.assert_allclose(q,ref,rtol=1e-6)
        np.testing.assert_allclose(c,dense[0,v],rtol=1e-6)
    assert tracker.credit[0,0] > history[-1][0,0]**.2


def test_mask_and_adaptive_credit():
    p=np.array([[.7,.3],[.2,.8]],np.float32); tr=TraceCredit((2,2),adaptive=True)
    q,_=tr.update(p,np.array([True,False]))
    np.testing.assert_allclose(tr.credit,[[.7,0],[0,0]])
    expected=p[0]*np.sqrt([1.7,1]); expected/=expected.sum()
    np.testing.assert_allclose(q[0],expected,rtol=1e-6)


def test_exact_candidate_history_not_previous_winner_probability():
    tr=OutputHistory(1,3)
    history=[np.array([[.6,.3,.1]],np.float32), np.array([[.2,.7,.1]],np.float32),np.array([[.55,.35,.1]],np.float32)]
    for p in history: token,f,h=tr.update(p,np.array([True]))
    assert token[0]==0 and f['out_changed'][0]==1 and f['out_streak'][0]==1
    assert f['out_candidate_top1_fraction'][0]==.5
    np.testing.assert_allclose(f['out_candidate_p_lag1'],[.2])
    np.testing.assert_allclose(f['out_candidate_variance'],[np.var([.6,.2,.55])],rtol=1e-6)
    assert h == ['0,1']


def test_latent_geometry_windows_and_zero_norm():
    tr=LatentHistory()
    for i in range(5): f=tr.update(np.array([[i+1.,0.],[0.,0.]],np.float32))
    assert f['cos_lag1'][0]==1 and f['cos_lag2'][0]==1
    assert f['w4_cos_mean'][0]==1 and f['w4_cos_variance'][0]==0
    assert f['l2'][0]==1 and f['cumulative_l2'][0]==4
    assert np.isnan(f['cos_lag1'][1])
    assert len(tr.hidden)==5 and len(tr.sim)==4


def test_centroid_is_computed_before_including_current_point():
    tr=LatentHistory()
    for point in [[1.,2.],[3.,3.],[5.,5.]]:
        f=tr.update(np.array([point],np.float32))
    scale=np.sqrt(11.5)
    np.testing.assert_allclose(f['w2_raw_center_distance'],[np.sqrt(15.25)/scale],rtol=1e-6)
    np.testing.assert_allclose(f['w2_raw_radius'],[np.sqrt(1.25)/scale],rtol=1e-6)
    np.testing.assert_allclose(f['w2_raw_center_drift'],[2.5/scale],rtol=1e-6)


def test_centroid_separates_angular_stability_and_radial_drift():
    tr=LatentHistory()
    for x in (1.,2.,3.,4.):
        f=tr.update(np.array([[x,0.]],np.float32))
    assert f['w3_cos_min'][0]==1
    assert f['w3_raw_center_distance'][0]>.5
    assert f['w3_unit_center_distance'][0]==0


def test_safe_and_stable_are_distinct():
    rows=[dict(position=0,step=t,top1=v) for t,v in enumerate([1,2,1])]
    label_rows(rows,[1])
    assert [r['safe_to_commit'] for r in rows]==[1,0,1]
    assert [r['stable_to_end'] for r in rows]==[0,0,1]
    assert [r['steps_until_commit'] for r in rows]==[2,1,0]


def test_prefix_invariance_and_label_isolation():
    rng=np.random.default_rng(3); ps=[rng.dirichlet([1,1,1],size=2).astype(np.float32) for _ in range(7)]
    def run(seq):
        tr=OutputHistory(2,3)
        return [tr.update(p,np.ones(2,bool))[1] for p in seq]
    short,long=run(ps[:3]),run(ps)
    for a,b in zip(short,long):
        for k in a: np.testing.assert_allclose(a[k],b[k],equal_nan=True)
