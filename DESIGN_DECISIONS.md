# Radius 2.0 and strength-control development run

| Decision | Choice made | Why | Alternatives considered | Could this affect the scientific result? |
|---|---|---|---|---|
| Radius expansion | Layer16 radius2.0 for both latent and no_geometry_radius | User requested both larger-radius variants | More layers or a larger grid | Yes; increasingly permissive geometry can approach unconditional accumulation |
| Main strength control | no_geometry_double, weight1, extra credit from first observation | Requested double-credit comparison; preserve original control | Delay extra credit one observation | Yes; first observation gets more evidence than radius hybrid |
| Matched optional control | --double-bonus-start 2 delays only unconditional extra credit, independently per position | Radius matching cannot reward initial anchor creation | Silently alter existing double method | Yes; separates first-visit timing from geometric selectivity |
| Cohort/settings | Same200 train questions, old saved policy, early stop, no trace; three methods share one run | Match development runs and rotate timing order | Full heldout tuning | Development only; freeze settings before heldout evaluation |
| Grading | Preserve existing strict/lenient fields for this speed-control round | Avoid mixing a scoring change into comparisons | Replace parser now | Known extraction errors remain; neither metric alone establishes quality parity |

The optional delayed control uses positive position credit to identify previously
observed positions. Active finite softmax distributions have positive p(max), so
every observation leaves positive credit even when decay is zero. Inactive balances
are unchanged. The setting only affects no_geometry_double and is saved in policy.json.
Synthetic tests verify equivalence to the radius hybrid on a single-region trajectory,
including changing top1 identities and staggered position activation. This is an
engineering check, not evidence that geometry helps. No model inference ran locally.

# Additive hybrids: unrestricted credit plus selective evidence

| Decision | Choice made | Why | Alternatives considered | Could this affect the scientific result? |
|---|---|---|---|---|
| Radius hybrid | One position balance earns w plus lambda*w on an existing-region match | Implements agreed base-plus-match-reward recurrence | Sum two region-local balances | High; old bonuses persist across later region changes |
| Radius/first visit | Layer16, radius1.00; creation never counts as closeness | Explicit user radius request; avoid tautological first-hit reward | Smaller radius or bonus on creation | High |
| Increment | w=p(top1)^gamma, lambda1 default | Preserve accepted confidence-weighted formulation | Literal +1/+1 | High |
| Credit hybrid | Unrestricted scalar mapped to current top1 plus lambda times full token-credit vector before log fusion | Preserve actual token-identity evidence from paper | Boost only current candidate with both balances | High; historical winner remains possible |
| Token recurrence | Original fixed beta .7/gamma .2; scalar saved latent coefficients | Leave paper comparator intact | Override both using one setting | Asymmetric if user changes scalar settings; documented |
| Combination | Sum credits inside one log(1+E), common alpha/threshold/fallback | Same decoding framework | Sum log boosts; OR gates | High |
| Strength control | Optional no_geometry_double: extra increment unconditionally, including first visit | Distinguish selector from generic stronger boost | No control; compensate alpha | High; intentionally stronger first visit |
| Compatibility | New methods opt-in, defaults unchanged; materialize bonus_weight in saved policy | Reproducibility without extra default expense | Change old combined method | Organizational |
| Trace/counters | Existing local boost-attribution and new per-method reason counts | Do not imply component-specific causality | Attribute every hybrid commit to radius | Important interpretation |

Earlier decisions follow.

# EOS early stopping and TPF update

| Decision | Choice made | Why | Alternatives considered | Could this affect the scientific result? |
|---|---|---|---|---|
| Stop rule | Committed EOS/EOT and all prior generated positions finalized; check after each insertion | Matches paper C.7 description, avoids premature stop on masked prefix | First predicted EOS; wait until block complete | High for TPF, runtime and completeness |
| Stop IDs | Tokenizer EOS and valid eot_id; persist exact IDs | Same endpoint as existing answer truncation | EOS-only | Possible paper mismatch; inspect IDs |
| Prefix versus suffix | Allow suffix masks after finalized stop; prefix masks still fail | Suffix is not part of completed answer | Require entire span | High |
| Trajectory preservation | No suffix pruning, no input truncation; stop is sole behavioral change | Preserve old prefix for direct parity | Stop attending to suffix after EOS seen | Runtime still includes full-span forward cost |
| Default | Explicit --early-stop, prior default remains off | Keep old commands reproducible | Change all defaults | Must record mode and not mix comparisons |
| Primary TPF | Sum completed visible tokens / sum all forwards; also include-stop/full-span and mean-ratio variants | Explicit numerators, failures and aggregation | 256/mean forwards everywhere | High; exact paper aggregation unverified |
| New radii | Layer16 .50 and1.00 | User requested best tested radius and larger boundary | Other layers or larger grids | Development sensitivity, not heldout |
| Matched comparator | Fresh Credit with early stop | Old fixed-span time cannot serve as matched reference | Reuse prior Credit | Additional 200 answers |
| EOS parity | Full baseline remains upstream-matched with stopping disabled; stopped prefixes checked against full runs | Upstream sampler has no stop mode | Compare stopped whole tensor to full tensor | Engineering test avoids false mismatch |
| Offline audit | Recompute metrics from saved lengths/forwards without inference | Resolve current reporting gap cheaply | Guess counterfactual stop time | Cannot recover early-stop wall time |

Historical decisions below remain applicable except for the new opt-in stop mode.

# No-geometry control and large-radius comparison

| Decision | Choice made | Why | Alternatives considered | Could this affect the scientific result? |
|---|---|---|---|---|
| Control recurrence | One per-position discounted p(top1)^gamma balance regardless of token changes | User authorized accumulation with no closeness test; isolate geometry | Unit increments; token-specific credit | High; uses same weighting as latent |
| Control implementation | No hidden access or anchors, exact shared fusion/acceptance/fallback | Explicit removal of geometry, not a huge-radius approximation | Infinite-radius hidden checker | Time includes saved hook/search overhead; compare forwards too |
| Larger radius | .50 at layer16, 10x original .05 | Material intervention after .10 barely changed aggregate behavior | .20, 1.0, multi-layer sweep | High, exploratory development choice |
| Cohort | Same 200 train questions, control plus latent in one run | User's ongoing 200-question experiment, shared load/rotating order | Another small pilot | Selection bias persists; no heldout claim |
| Geometry counters | New/reused/invalid active observations, first visits included | Determine whether radius changes matching, not just acceptance | Only count committed positions | Interpretation; denominator explicitly documented |
| Defaults | Control is opt-in | Avoid silently increasing default experiment cost | Add to all-method default | Operational only |

Earlier decisions follow.

# Boost attribution instrumentation

| Decision | Choice made | Why | Alternatives considered | Could this affect the scientific result? |
|---|---|---|---|---|
| Boost-enabled position | Raw max < tau and enhanced max >= tau on an actual threshold insertion | Position would not pass raw-confidence selection on this state | Compare only selected token's raw probability | Interpretation: winner changes are separate |
| Partition | Boost-enabled / already-confident / fallback / scheduled, non-mask insertions only | Distinguish boosting from ordinary acceptance and forced progress | Count all method commits as boosted | High for claims about the source of acceleration |
| Winner audit | Separate enhanced winner != raw argmax count; trace selected token raw probability | Boost can change token identity without enabling a new position | Ignore alternative winner | Interpretation only |
| Collection | GPU counters, available with no-trace; transfer after loop | Avoid traced timing requirement | Full trajectory logging | Small unmeasured overhead; no policy changes |
| Historical results | Missing counters reported unavailable | Cannot reconstruct raw confidence from generated text | Treat missing as zero | Prevent false conclusions |
| Scope | Same-state local threshold counterfactual only | Prior boosted decisions already changed the trajectory | Claim exact forwards saved | No causal speed or stability claim |

Earlier policy decisions follow and remain applicable unless superseded above.

# Current policy: persistent regions and Credit-style logit fusion (v2)

This section supersedes the historical decisions below. User requested preserving
nonconsecutive evidence for returning regions and replacing direct latent commitment
with the paper's logit boosting and confidence acceptance. No classifier is used.

| Decision | Choice made | Why | Alternatives considered | Could this affect the scientific result? |
|---|---|---|---|---|
| Region memory | Per-position bank, no eviction during active block; every saved credit decays | Preserve A/A/B/A and B's evidence as requested | Reset-on-excursion; capped bank | High; new algorithm and memory cost |
| Anchor formation | First finite nonzero visit freezes anchor and its L2 norm; first visit earns credit | Count singleton B immediately, analogous to first token occurrence | K-vector compact warm-up | High; explicitly removes old three-vector delay |
| Geometry | Nearest normalized L2 anchor within radius .05, frozen forever | Bound drift without moving centers | Cosine, covariance, moving centroid | High; radius/layer require development sensitivity |
| Overlap | Exactly one region, nearest; oldest on exact ties | Avoid multiple credits from one observation | All matches, oldest match | High; greedy regions depend on order |
| Credit increment | Binary match/new-region indicator times current p(top1)^gamma | User requested paper formula; retain binary closeness | Unit hits; graded distance | High; earlier unit-hit rule superseded |
| Coefficients | beta=.7, alpha=.65, gamma=.2, tau=.95 | Match fixed CreditDecoding comparison | Earlier .9/count3 or tuned coefficients | High; not validated for latent signal |
| Vocabulary mapping | Current raw top1 alone receives full current-region credit | Transfer nearby latent evidence across token changes | Region token histogram, boost entire distribution | High; mapping is new, not specified by paper |
| Acceptance | max of boosted softmax >= common tau; no latent count gate | Explicit user correction | Direct latent trigger | High |
| Combined ablation | Sum token credit and mapped region credit before same log fusion | Remove obsolete OR gate consistently | Product/separate weighted boosts; omit combined | High; double evidence strength is a confound |
| Fallback | Shared prior top1 default retained; latent now ranks boosted confidence | Match each method's actual distribution | No fallback | High; separately reported |
| Invalid/inactive states | Invalid active state earns nothing but decays bank; inactive state leaves bank untouched | Preserve valid past evidence without NaN poisoning | Reset invalid state | Small except numerical failures |
| Bank allocation | Dynamic geometric GPU growth; no hidden cap/eviction | Preserve requested history, allocate only as needed | Preallocate worst case; CPU offload | Compute/memory cost; all-layer full-sequence can be large |
| Diagnostic parity | Shared region_distribution with logits on model device; save boosted probabilities and configured tau | Same evidence and acceptance as rollout | Offline approximation or stale gate labels | High |
| Compatibility | Version marker, reject old setting keys and old diagnostic manifests | Prevent silent reinterpretation | Automatic migration | Recollection required |
| Audit trace | Region ID and weighted regional credit on commitments; combined reason separate | Make retained-region use inspectable | Counts alone | Organizational; tracing off for speed |

Historical decision log follows. Superseded defaults below are not current settings.

---

# Latest agreed stability and measurement update

This section supersedes earlier default supporting-mean and independent-diagnostic
choices below; earlier sections remain history.

| Decision | Choice made | Why | Alternatives considered | Could this affect the scientific result? |
|---|---|---|---|---|
| Primary closeness rule | Frozen compact reference, binary hits, decay, large-excursion reset; mean gate OFF | User approved simpler rule; avoid first-hit r/4 restriction | Keep mean gate or replace it with sliding drift | High |
| Mean guard | drift_ratio=null by default; optional ablation preserves old behavior | Separate a scientific sensitivity from the primary rule | Remove all old code | High if old policy JSON reused |
| Shared state | RegionSupport called by rollout and diagnostic PolicyHistory | Exact same decisions on identical vectors | Reimplement offline from insufficient summaries | High; explicit parity tests |
| Current timeline analysis | policy_timelines consumes recorded policy_accept; no alternate radius search | Do not tune a different notion of stability | Legacy timelines grid | New policies require recollection |
| Hidden memory | Shared-policy state per captured layer in diagnostics in addition to old feature ring | Avoid raw hidden dumps while preserving exact policy | Offline vector storage | Additional GPU state per layer and host feature RAM; profile pilot |
| Timing instrumentation | No trace by default; counters on GPU, transfers after loop | Remove avoidable per-step CPU synchronization | Default traced timing | Wall-clock comparison changes |
| Parity cohort | Up to20 evenly spaced token-length ranks, including extremes | Cover more than first example | First20 regardless of length | GPU validation still needed |
| Trace parity | One selected prompt per method, exact tokens/counts/forwards | Check instrumentation cannot change generation | Skip validation | Engineering assurance,not quality evidence |
| Fallback | Keep prior explicit top1 default; separate counts and no-fallback option | User asked explanation,not silent removal | Disable immediately | High; no-fallback can stall |
| Stalled-context audit | Count steps without any actual token insertion | Repeated inputs can create trivial latent stability | Ignore unchanged inputs | Important no-fallback confound |

Default earliest detection is observation7, explicitly tested:3 reference points then
scores1,1.9,2.71,3.439. No real-model settings were tuned during this update.

# Current scope: actual generated-answer evaluation

The latest user request supersedes diagnostics-only gating. Actual GSM8K policy
rollouts are now the primary experiment; baseline-token agreement is not the quality
criterion. The table below records the new autonomous choices. Earlier entries are
retained as decision history and do not override this scope.

| Decision | Choice made | Why | Alternatives considered | Could this affect the scientific result? |
|---|---|---|---|---|
| Quality target | Actual final numeric GSM8K answer,allow different wording | Explicit user correction | Baseline-token agreement | High; answers can improve or regress independently |
| Development/evaluation | Train questions for tuning,test after freezing | Avoid test leakage | Tune on original mixed diagnostic prompts | High |
| Latent support | Binary inside-radius hits,decay.9,threshold3 initially | Requested accumulation; simple starting values | Consecutive mode or graded closeness | High,untuned |
| Region resets | Moderate miss decays;distance>2r or support-mean drift>r/4 resets | Permit nonconsecutive hits without cross-region credit | Reset every miss; multiple-anchor bank | High |
| New anchor | K3 compact reference points,scale/center frozen,reference gives zero credit | Separate region formation from confirmation | Count formation points as support | High,delays acceptance |
| Drift check | Unweighted running mean of close supporting vectors | Ignore isolated moderate misses while bounding support | Old rolling K-center test | High; prior diagnostic thresholds not validated here |
| Actual layer |16 initial;select1..32 on development;single GPU hook | Runtime-relevant policy cost | All32 in rollout | High; layer16 not proven best |
| Policy baselines | Standard,confidence,credit,latent,combined | Isolate each source of acceleration | Only credit versus latent | Interpretation and rental cost |
| Credit decision | Paper boosted distribution's max>=tau,not raw credit threshold | Faithful Eq6/7 use | Arbitrary credit-count cutoff | High |
| Shared fallback | On zero proposals,commit best method candidate;count and expose | Bounded practical completion | --fallback none | High; not specified by Algorithm1 |
| Baseline parity | Preserve upstream full top-k input shape/native softmax dtype | Ties can change commitment order | Slice-local top-k | High |
| Combined conflicts | Credit candidate where its gate passes,raw candidate for latent-only;credit fallback | Match each source to candidate | Always raw/always boosted | High |
| Stopping | No EOS early termination,same block cap;incomplete is incorrect | Explicit bounded comparison | EOS stopping with completed prefix | High,suffix work can affect speedup |
| Prompt | Native chat,question plus final #### instruction,no gold/examples | Straightforward answer-level grading | Paper-specific few-shot setup | High |
| Numeric extraction | Last standalone #### line,Decimal equality;secondary last-number score | Accept wording and decimal equivalents | Official first-match/string equality | Format sensitivity |
| Runtime | Per-method warmup,rotating order,CUDA synchronization;trace overhead included | Measure actual cost,not just fewer forwards | Cold timing/no trace | Hardware variance remains |
| Storage/provenance | All outputs/commits/gold predictions,question IDs,policy,source/package manifest | Auditable comparison | Aggregate accuracy only | Organizational and reproducibility |
| Initial budget |100 train questions,explicit500 test example;no automatic sweep | Start small before rental scaling | Full dataset/search immediately | Statistical power |

These are implemented in decoding.py and gsm8k_rollout.py; settings are saved in
policy.json per run. No real-model tuning has occurred. See ROLLOUT_EXPERIMENT.md
for formulas, CLI options, strict/consecutive sensitivities and heldout workflow.

# Design decision history and current rollout choices

**Current update:** Instruct/block64 is now supported and recommended for the closer
CreditDecoding comparison; Base/full remains available. See the update section below,
which supersedes earlier Base-only/full-sequence-only scope statements.

The user's corrections supersede the original classifier design: training-free,
Base checkpoint, full-sequence baseline, diagnostics only, all-layer investigation,
and closeness tuning with a CreditDecoding timeline comparison. The project is now
standalone. Earlier classifiers are inactive in archive/classifier_v1.

Each row records the decision, choice made, reason, alternatives considered, and
potential effect on the scientific result. Requested choices are marked as such;
other details are autonomous implementation choices. No classifier is trained, but
radius/window/layer selection **does use development labels**. Calling that tuning
free of label use would be incorrect. No real-model radius has been tuned yet.

| Decision | Choice made | Why | Alternatives considered | Could this affect the scientific result? |
|---|---|---|---|---|
| Project location | Standalone dllm-latent-stability, requested | Independent experiment | RollingForcing directory | Organizational only |
| Method scope | Training-free diagnostics, requested; no decoder intervention | Test the phenomenon before changing the sampler | Immediate confidence boost or classifier | High: observations do not establish counterfactual safety |
| Model | Requested LLaDA-8B-Base; HF revision 0f2787f2d87eac5eed8a087d5ecd24277e6255b2 | Reproducible open model in the prior-work family | Instruct, Dream, additional checkpoints | High: paper credit settings were tuned on Instruct |
| Sampler | Unmodified generate.py at upstream 9182493720ed723ef8031210d85959364e51cbe0 | Preserve baseline behavior | Rewritten sampler | High: fake parity tested, real parity still needed |
| Decoding | Requested full sequence; 256 tokens/256 steps, temperature0, CFG0, no early stop/cache | Long observable histories; one baseline token per step | 128 steps; length512; block64 from paper | High: schedule changes opportunity and safety |
| Base formatting | Plain Question/Answer continuation, no chat template | Base has no Instruct role framing | Raw continuation, few-shot examples | High: prompt framing and instruction-following quality |
| Layers | All32 transformer layers plus final norm, requested | Search layer dependence without extra forwards | Earlier quarter-depth sample | High: multiple comparisons and observation overhead |
| Hidden tensor | Complete post-layer residual output, before next layer; separate post-final-RMSNorm output before LM head | Distinguish internal and directly logit-adjacent states | Attention outputs, internal pre-normalization tensors | High: tensor choice changes geometry |
| Center | Mean of K reference vectors, frozen throughout C subsequent confirmations | Test new vectors against a region that cannot chase them | EMA, rolling-only center, robust center, covariance ellipsoid | High: small samples cannot estimate full4096-dimensional covariance |
| Scale | RMS norm of reference vectors, also frozen; raw float32 Euclidean distances | Compare layer scales while retaining radial movement | Unit sphere, absolute distances, whitening, layer-fitted scales | High: residual anisotropy/common magnitude can dominate |
| Region criteria | All C vectors within r; reference radius <=r/2; maximum rolling K-center displacement from anchor <=r/4 | Require proximity, compactness, and limited drift with one tolerance | Independently tune three bounds; cumulative path length | High: ratios1/2 and1/4 are assumptions requiring sensitivity analysis |
| Window grid | K2/3/4 reference observations, C2/3 subsequent observations | Small causal bounded search; persistence | Longer confirmation, EMA, consecutive token repetition | High: delay versus false stability |
| Radius grid | .01,.025,.05,.1,.2,.4 relative distance | Broad initial logarithmic-like coverage | Data quantiles, denser adaptive search | High: grid may miss a useful operating point; expand only in new development work |
| Missing history | Require complete K+C observations and finite nonzero reference scale | Avoid early artificial stability | Partial windows or imputation | Early positions may never have enough history |
| Drift limitation | Fixed anchor during confirmation; successive windows may form before the first trigger | Bounded causal online check | Anchor for entire generation; change-point detection | Slow drift can still pass; stability does not promise future immobility |
| Partitions | Prompt-hash60% discovery/20% calibration/20% heldout, seed1729 | Separate grid selection, confirmation, final assessment | Two-way split or nested resampling | High: counts fluctuate, near duplicates can cross groups |
| Selection objective | Maximize correct lead over baseline commit per position, subject to early-event precision | Reward earlier safe opportunities | Coverage only, relative-step lead, matched credit precision | High: favors long leads; errors still require separate reporting |
| Precision targets | .95,.97,.99 empirical, support50 early events across10 prompt groups on each development partition | Avoid selecting rules on a handful of tokens | Confidence-bound constraint or greater support | High: no population guarantee, particularly at99%; paired bootstrap reported |
| Selection ties | Higher precision, smaller radius, more confirmations, larger window | Deterministic conservative ties | Smaller latency | Can change chosen setting on small data |
| Calibration failure | Abstain; no runner-up search after inspecting calibration labels | Limit adaptive reuse of validation outcomes | Try successive nominees | May discard useful rules; negative evidence is retained |
| Primary layer | Best discovery objective among calibration survivors per target; report all layers | Independent primary choice before interpreting heldout | Pick best heldout layer | Multiple-layer exploratory findings still require replication |
| Evaluation unit | First proposal per position, including incorrect proposals | Match irreversible early commitment | Token-step precision, first correct proposal | High: never replace a wrong early trigger with a later correct one |
| Baseline commit rows | Retained as observed commit endpoint; only strictly earlier proposals count toward tuning support/precision | Prevent trivial commit-time positives inflating results | Exclude rows entirely | High: endpoint required for valid lead/censoring |
| Credit baseline | Final ACL Eq6/7, alpha.65/beta.7/gamma.2; adaptive gamma1 and alpha=beta=1-mask_ratio | Reproduce main equations and a parameter-adaptive comparator | Retune paper parameters on Base, Eq13 all-vocabulary variant | High: equation reproduction is not rollout reproduction |
| Credit candidate | Enhanced argmax, with its own final-token agreement label | Credit fusion can change the winning token | Raw top1 scored under enhanced distribution | High: using raw winner would misstate credit safety/timing |
| Credit timeline | First max-q crossing .95/.97/.99 on baseline states; no top-k fallback | Isolate output evidence and threshold timing | Full CreditDecoding rollout with selection/fallback | High: shadow timing is not actual CreditDecoding commitment time |
| Censoring | No crossing before baseline commit remains unobserved; report lower bound if latent was earlier | Do not invent future credit timing | Assign baseline step as credit time | High: prevents optimistic fabricated comparisons |
| Labels | SAFE agrees with final baseline token; STABLE is unchanged raw candidate through commit then fixed state | Keep proxy agreement and trajectory stability separate | Ground truth or interventions | High: both can agree with a wrong final answer |
| EOS | Full baseline trajectory primary; separately rerun pre-final-EOS sensitivity | Expose trivial suffix stability | Stop early | High: filtering uses future outcome only to define cohort |
| Prompt sources | Pinned Dolly QA/creative plus GSM8K; balanced tasks, normalized-text dedup,512-token cap without truncation | Practical task diversity | Different corpora, raw continuations, longer contexts | High: public-data contamination, task and length effects |
| Precision/dtype | BF16 inference, float32 online hidden/output statistics and saved continuous columns; batch1 | Fit about40 GB GPU; stable bounded CPU processing | Float32 model, GPU statistics, batches | Numerical changes can shift threshold crossings |
| Storage | Six previous hidden vectors per layer, no raw dumps; four previous distributions, dense vocabulary moments/credits | Sufficient causal history with bounded memory | Sparse credits, approximate sketches, offline raw dumps | New hidden features require recollection; host memory still substantial |
| Statistics | Paired prompt-bootstrap200 draws; safe/unsafe coverage and observed lead separately | Respect correlated positions within prompts | More draws, task-stratified bootstrap, model-based uncertainty | Small heldout sets and many layers limit precision |
| Descriptive comparator | Separate old cosine audit, layer16/K3/.99 plus grid; coarse confidence/streak/credit bins | Retain original movement/ranking diagnostics | Replace with learned conditional model | Does not evaluate the tuned anchor and cannot establish full conditional independence |
| Engineering fixture | Random small tensors plus deliberately stationary/correct acceptance fixture | Exercise abstention and acceptance/report paths | Full checkpoint locally | No scientific evidence; synthetic tags retained |

## Geometry and memory details

For reference h1..hK, anchor a=mean(h) and scale s=sqrt(mean(||h||^2)).
The quantities compared with r,r/2,r/4 are respectively max_j||probe_j-a||/s,
max_k||h_k-a||/s, and max_j||rolling_center_j-a||/s. The last is displacement
from the same anchor, not just displacement from the previous center. It is not
cumulative path length. A temporary out-and-back excursion cannot be hidden by
looking only at the last confirmation point. It still cannot detect future drift.

Cosines at lag1/2, K2/3/4 rolling cosine summaries, L2/norm/cumulative movement,
previous-centroid raw/unit geometry, and anchored statistics are all collected.
RMSNorm divides a vector by its root-mean-square coordinate magnitude (with epsilon)
and applies learned coordinate weights. The reference distance scale above is an
RMS of vector *norms*, not the model's RMSNorm operation.

At32 layers,256 positions,width4096,float32, six retained vectors consume768 MiB
host RAM, plus final norm, current activations, working arrays, full-vocabulary
histories, and scalar rows. No extra model forwards are needed for layers, but
transfer/statistics/storage are substantial. Full-sequence256-step collection has
32896 rows per generation. Timeline analysis projects one layer and one generation
at a time; descriptive all-row analysis does not. Profile smoke on Vast first.

## Sensitivities still required before a strong conclusion

Test tighter/looser cloud and drift ratios on discovery data; longer confirmation
windows require new collection. Compare raw versus unit geometry, Base versus
Instruct, steps256 versus128, and primary versus pre-EOS cohorts. These choices can
change the conclusion and have not been silently resolved by claiming a universal
radius. Fixed credit paper parameters may transfer poorly to Base/full-sequence;
a later matched development tuning of both methods would strengthen comparison.
The decision to defer actual rollout preserves the requested Phase1 scope.

## Update: Instruct and block-mode comparison

The user now explicitly allows block decoding and the same checkpoint as the paper
for a closer comparison. Recommended configuration: configs/credit_instruct_block64.json,
GSAI-ML/LLaDA-8B-Instruct at08b83a6feb34df1a6011b80c3c00c7563e963b07,256 output
tokens/steps,64-token blocks, all32 residual layers plus final norm. The32-layer/4096-wide
architecture and chat template were checked from small remote metadata files only.
No weights were downloaded locally. The paper's exact checkpoint commit is not known;
this matches the named checkpoint, not a verified paper revision.

Decision: preserve independent checkpoint/block sensitivities. Choice: add Instruct/full,
Base/block64 and Instruct/block64 configs; retain Base/full as the CLI default for
backward compatibility, but recommend Instruct/block64 in the Vast runbook. Why: avoid
confounding architecture/checkpoint and schedule effects. Alternatives: replace Base
entirely or expose arbitrary unverified model IDs. Likely impact: high; compare the
four configurations on identical prompt text where possible, checking length eligibility.

Decision: Instruct formatting. Choice: pinned tokenizer chat template, one user message,
assistant generation prefix; no added system prompt or exemplars. Why: native Instruct
format. Alternatives: paper benchmark-specific few-shot wrappers. Likely impact: high;
this is not exact OpenCompass evaluation. Dataset preparation and collection call the
same formatter; prompt snapshot names in the full script include the config hash.

Decision: block histories. Choice: reset latent/output/credit history at block entry;
observe only active-block masks, not future masked blocks. Adaptive credit uses the
active block's mask ratio. Why: histories represent eligible commitment opportunities.
Alternatives: track future blocks before eligibility or global credit/mask ratio.
Likely impact: high; full-sequence and block histories are different experiments.
Global step numbers are retained, so first-trigger lead compares the same position
on the same baseline clock. An unsafe first trigger remains final for that rule.

Decision: paper alignment boundary. Choice: keep the pinned unchanged baseline without
EOS early termination and use shadow credit threshold crossings. Why: preserve Phase1
complete trajectories. Alternatives: reproduce full CreditDecoding policy and benchmark
suite now. Likely impact: high; matched checkpoint/block size is a closer comparison,
not reproduction of speedup, benchmark quality, exact prompts or author model revision.
Closeness radii must be tuned independently for each configuration using the existing
prompt partitions, never transplanted silently from Base to Instruct.

The bootstrap accepts a config path as its first argument and the full-experiment
script as its third argument. `collect --block-length 64` and `--block-length full`
are supported. Changed config/code needs a new run directory. CPU checks cover
unmodified-upstream parity at block boundaries, history/credit resets, active-position
eligibility, native chat-template calls and global-step timeline accounting.

## Explicit reset after an excursion (current mechanism)

Decision: episode-based reference/confirmation instead of independently accepting
overlapping windows. Choice made: collect K consecutive vectors; validate cloud
radius, freeze mean and RMS vector-norm scale, then check every confirmation prefix
against that SAME anchor. Any cloud, point-distance or anchor-displacement failure
invalidates the entire candidate episode. The failing point seeds a new reference
window, and all preceding reference/confirmation points are excluded. Why: prevent
old stability evidence surviving an excursion. Alternatives: discard the failing
point too, sliding-window-only tests, or resetting only on top1 identity changes.
Could affect scientific result: yes; fresh episodes are more conservative and can
detect later than overlapping windows. The current point as a seed allows settling
in a new region without discarding its first observation.

State is independent per position/layer/radius/window/confirmation rule. Reference
compactness is tested when K points are available; confirmation distance and drift
are checked on every subsequent point, not just after C points. No token identity
or future label enters these resets. A top1 change alone is allowed: latent-before-
lexical convergence is still the hypothesis. Distance refers to the position's
hidden vector, not word IDs or a separate token-embedding semantic distance. During
reference formation no frozen anchor exists yet; a failed K-point cloud starts over.

Collection keeps causal C0/C1 prefix statistics in addition to C2/C3. A small scalar
state machine reconstructs each episode in timeline analysis, using the exact matching
reference window at each prefix. This avoids storing separate4096-dimensional anchors
for all36 threshold/window settings at every layer. Raw hidden ring storage stays at
six previous vectors; scalar storage increases. Old traces lacking prefixes must be
recollected. The stateless AnchoredRegionRule.evaluate remains a window comparator;
the actual timeline acceptance uses episode_triggers in stability.py.

Resets apply before the first proposal. Once a method first proposed a token, its
proposal remains in the observational evaluation even if later drift reveals an error.
Baseline generation and independent CreditDecoding history are not reset by the latent
rule. No confidence boost, model training, or actual commitment is introduced.

Validation:36 CPU tests passed, including jump/restart, excursion-and-return,
drift-veto reset even inside the distance ball, prefix causality and preserving an
earlier trigger after a later jump. A small engineered episode integration fixture
checks report/selection paths; it is not scientific evidence.

## TPS and200-question pilot

Decision: delivered-output throughput definition. Choice: sum pre-EOS/EOT output
lengths for complete generations divided by all measured decoder seconds, including
failed attempts; separately report full-span TPS and mean completed output length.
Why: count delivered tokens without hiding failure time or suffix work. Alternatives:
mean per-answer TPS,all256 slots regardless of EOS,or include partial failed outputs.
Could affect scientific result: yes,variable verbosity and EOS placement change TPS;
report accuracy,forward counts and seconds per answer alongside it. Updated CPU tests
verify ratio-of-totals aggregation and that incomplete attempts retain their time cost.
The initial200-question run uses train for development with baseline/credit/latent;
this does not imply frozen-test significance or a validated latent setting.

Shared-policy decisions in the collector are computed on the model device, like the
rollout policy; only diagnostic scalars and legacy hidden-feature vectors transfer
to CPU. This avoids introducing a systematic CPU-versus-GPU reduction difference
between discovery and execution. All-layer policy state adds GPU memory, so retain
the required small Vast memory/parity pilot. Collection overhead is not a speed result.
