"""Actual deterministic decoding policies; no future labels or reference answers."""
from dataclasses import dataclass
import math
import torch
from .third_party.llada_generate import get_num_transfer_tokens


@dataclass(frozen=True)
class LatentSettings:
    layer: int = 16
    radius: float = .05
    decay: float = .7
    alpha: float = .65
    gamma: float = .2
    mechanism: str = 'persistent_regions_v2'

    def __post_init__(self):
        if not 1 <= self.layer <= 32:
            raise ValueError('Require layer1..32')
        if self.mechanism != 'persistent_regions_v2':
            raise ValueError('Unsupported latent mechanism; do not reuse legacy support-gate settings')
        if not all(math.isfinite(v) for v in (self.radius,self.decay,self.alpha,self.gamma)):
            raise ValueError('Latent settings must be finite')
        if not (self.radius>0 and 0<=self.decay<=1 and self.alpha>=0 and self.gamma>0):
            raise ValueError('Invalid latent settings')


class RegionSupport:
    """Persistent, position-specific frozen regions with discounted credit.

    The first valid observation outside all saved regions creates an anchor and
    earns its first p(top1)**gamma credit. Nothing is evicted during the block.
    Each later observation updates exactly one region: the nearest normalized
    anchor within radius (oldest breaks ties), otherwise a new region. Anchors
    and norm scales never move. No support threshold directly commits a token.
    Storage grows geometrically with discovered regions, not with raw history.
    """
    def __init__(self, hidden, settings):
        self.s=settings
        n,d=hidden.shape; device=hidden.device
        self.anchor=torch.zeros(1,n,d,device=device,dtype=torch.float32)
        self.scale=torch.ones(1,n,device=device,dtype=torch.float32)
        self.credit=torch.zeros(1,n,device=device,dtype=torch.float32)
        self.count=torch.zeros(n,device=device,dtype=torch.long)

    def _grow(self):
        self.anchor=torch.cat((self.anchor,torch.zeros_like(self.anchor)),0)
        self.scale=torch.cat((self.scale,torch.ones_like(self.scale)),0)
        self.credit=torch.cat((self.credit,torch.zeros_like(self.credit)),0)

    def update(self, hidden, active, confidence):
        h=hidden.float(); s=self.s
        norm=h.norm(dim=-1)
        valid=active & torch.isfinite(h).all(-1) & torch.isfinite(norm) & (norm>1e-12)
        # Invalid observations do not erase the bank; time still decays its credit.
        self.credit[:,active]*=s.decay
        slots=torch.arange(self.anchor.shape[0],device=h.device)[:,None]
        distance=(h[None]-self.anchor).norm(dim=-1)/self.scale.clamp_min(1e-12)
        distance=distance.masked_fill(slots>=self.count[None],torch.inf)
        nearest,region=distance.min(0)
        matched=valid & (nearest<=s.radius)
        new=valid & ~matched
        if (new & (self.count==self.anchor.shape[0])).any(): self._grow()
        region=torch.where(new,self.count,region)
        positions=torch.where(new)[0]
        self.anchor[region[positions],positions]=h[positions]
        self.scale[region[positions],positions]=h[positions].norm(dim=-1)
        self.count[new]+=1
        positions=torch.where(valid)[0]
        self.credit[region[positions],positions]+=confidence[positions].float().pow(s.gamma)
        support=torch.zeros_like(confidence,dtype=torch.float32)
        support[positions]=self.credit[region[positions],positions]
        return support,dict(support=support,close=matched,new_region=new,
            relative_distance=torch.where(new,torch.zeros_like(nearest),nearest),
            region_id=torch.where(valid,region,torch.full_like(region,-1)),
            region_count=self.count)


def fuse_credit(logits, credit, alpha=.65):
    """CreditDecoding Eq7, shared by token-history and latent-region policies."""
    return torch.softmax(logits.float()+alpha*torch.log1p(credit),dim=-1)


def update_token_credit(logits, credit, active, beta=.7,gamma=.2):
    p=torch.softmax(logits.float(),dim=-1)
    confidence,token=p.max(-1)
    credit.mul_(beta)
    credit.scatter_add_(1,token[:,None],(confidence.pow(gamma)*active)[:,None])


def credit_distribution(logits, credit, active, alpha=.65,beta=.7,gamma=.2):
    """Final ACL Eq6/7, update first, then fuse and normalize (float32)."""
    update_token_credit(logits,credit,active,beta,gamma)
    return fuse_credit(logits,credit,alpha)


def region_distribution(logits, hidden, active, region, token_credit=None):
    """Map current region credit ONLY to current raw top1, then apply Eq7.

    Optional combined ablation sums token and region credit before log fusion.
    It is not the old OR of two acceptance gates. Inactive/invalid observations
    contribute no latent credit; inactive positions are never selected by decode.
    """
    confidence,token=torch.softmax(logits.float(),dim=-1).max(-1)
    support,info=region.update(hidden,active,confidence)
    evidence=torch.zeros_like(logits,dtype=torch.float32)
    evidence.scatter_(1,token[:,None],support[:,None])
    if token_credit is not None: evidence=evidence+token_credit
    return fuse_credit(logits,evidence,region.s.alpha),info


def no_geometry_distribution(logits, credit, active, settings):
    """Control: one balance per position, no hidden states or token-specific memory.

    Every active observation earns p(raw_top1)**gamma regardless of token changes.
    Only current raw top1 receives the balance through the same logit fusion.
    """
    confidence,token=torch.softmax(logits.float(),dim=-1).max(-1)
    credit[active]=settings.decay*credit[active]+confidence[active].pow(settings.gamma)
    support=credit.masked_fill(~active,0)
    evidence=torch.zeros_like(logits,dtype=torch.float32)
    evidence.scatter_(1,token[:,None],support[:,None])
    return fuse_credit(logits,evidence,settings.alpha),dict(support=support,
        region_id=torch.full_like(token,-1))


def finalized_stop_mask(generated, mask_id, stop_ids):
    """Stop tokens qualify only after every earlier generated position is filled."""
    is_stop=(generated[:,None]==stop_ids[None,:]).any(-1)
    prefix_complete=(generated==mask_id).long().cumsum(0)==0
    return is_stop & prefix_complete


@torch.inference_mode()
def decode(model,prompt,config,method='credit',threshold=.95,latent=None,fallback='top1',trace=True):
    """Each policy controls its own model inputs. Batch1, no CFG/cache; optional finalized-prefix EOS stop.

    baseline uses the original fixed schedule; other methods commit all passing
    positions, optionally forcing one best candidate only if none pass. The same
    per-block forward cap applies to every method. Incomplete outputs are reported,
    never silently filled or dropped. Tracing copies only actual commit events.
    """
    if method not in ('baseline','confidence','credit','latent','combined','no_geometry') or fallback not in ('top1','none'):
        raise ValueError('Unknown method/fallback')
    if not 0<threshold<=1: raise ValueError('Threshold must be in (0,1]')
    if prompt.ndim!=2 or prompt.shape[0]!=1: raise ValueError('Require batch1')
    length=config['gen_length']; block=config['block_length']; steps=config['steps']; mask=config['mask_id']
    if block=='full': block=length
    if min(length,block,steps)<=0 or length%block or steps%(length//block) or steps>length:
        raise ValueError('Invalid decoding schedule')
    if (prompt==mask).any(): raise ValueError('Prompt contains mask token')
    early_stop=bool(config.get('early_stop',False))
    configured_stops=config.get('stop_token_ids',[])
    if early_stop and (not configured_stops or mask in configured_stops):
        raise ValueError('Early stop requires non-mask stop_token_ids')
    stop_ids=torch.tensor(configured_stops,device=prompt.device,dtype=torch.long)
    stopped=False
    cap=steps//(length//block); offset=prompt.shape[1]
    x=torch.full((1,offset+length),mask,dtype=torch.long,device=prompt.device);x[:,:offset]=prompt
    use_latent=method in ('latent','combined'); latent=latent or LatentSettings()
    captured={}; handle=None; span=[0,0]
    if use_latent:
        def hook(module,args,result):
            h=result[0] if isinstance(result,tuple) else result
            captured['hidden']=h[0,span[0]:span[1]].detach().float().clone()
        handle=model.model.transformer.blocks[latent.layer-1].register_forward_hook(hook)
    forwards=0; commits=[]; trace_parts=[];complete=True
    reason_codes=torch.arange(8,device=x.device)
    counts=torch.zeros(8,device=x.device,dtype=torch.long)
    unchanged_steps=torch.zeros((),device=x.device,dtype=torch.long)
    # Actual non-mask insertions only; first four buckets partition commitments.
    # Fifth is an overlapping audit for boosted winner changes at accepted positions.
    acceptance_counts=torch.zeros(5,device=x.device,dtype=torch.long)
    region_counts=torch.zeros(3,device=x.device,dtype=torch.long)  # new, reused, invalid active
    try:
        for block_id in range(length//block):
            start=offset+block_id*block;stop=start+block;span[:]=[start,stop]
            schedule=get_num_transfer_tokens(x[:,start:stop]==mask,cap)[0]
            credit=None;region=None;position_credit=None
            for local_step in range(cap):
                active=x[0,start:stop]==mask
                if not active.any(): break
                captured.clear()
                all_logits=model(x).logits;logits=all_logits[0,start:stop];forwards+=1
                # Preserve upstream probability dtype for baseline fixed-schedule parity.
                p=(torch.softmax(all_logits,dim=-1)[0,start:stop] if method=='baseline'
                   else torch.softmax(logits.float(),dim=-1))
                confidence,tokens=p.max(-1)
                if method=='baseline':
                    tokens=logits.argmax(-1)
                    confidence=p.gather(1,tokens[:,None]).squeeze(1)
                raw_confidence,raw_tokens=confidence,tokens
                info=None
                if method in ('credit','combined'):
                    if credit is None: credit=torch.zeros_like(logits,dtype=torch.float32)
                    if method=='credit':
                        q=credit_distribution(logits,credit,active)
                        credit_conf,credit_tokens=q.max(-1)
                    else:
                        update_token_credit(logits,credit,active)
                if use_latent:
                    if 'hidden' not in captured: raise ValueError('Layer hook did not capture hidden state')
                    if region is None: region=RegionSupport(captured['hidden'],latent)
                    latent_q,info=region_distribution(logits,captured['hidden'],active,region,
                        credit if method=='combined' else None)
                    latent_conf,latent_tokens=latent_q.max(-1)
                    region_counts+=torch.stack([info['new_region'].sum(),info['close'].sum(),
                        (active & (info['region_id']<0)).sum()])
                if method=='no_geometry':
                    if position_credit is None: position_credit=torch.zeros(block,device=x.device)
                    latent_q,info=no_geometry_distribution(logits,position_credit,active,latent)
                    latent_conf,latent_tokens=latent_q.max(-1)
                reason=torch.zeros(block,device=x.device,dtype=torch.long)
                if method=='baseline':
                    selected=torch.zeros_like(active)
                    # Match upstream topk's FULL input shape, including -inf
                    # prompt/future slots: slicing changes equal-score tie breaks.
                    ranked=torch.full((x.shape[1],),-torch.inf,dtype=confidence.dtype,device=x.device)
                    ranked[start:stop]=confidence.masked_fill(~active,-torch.inf)
                    selected[torch.topk(ranked,int(schedule[local_step])).indices-start]=True
                    reason[selected]=1
                elif method=='confidence':
                    selected=active & (confidence>=threshold);reason[selected]=2
                elif method=='credit':
                    confidence,tokens=credit_conf,credit_tokens
                    selected=active & (confidence>=threshold);reason[selected]=3
                else:  # All boosted policies use the same confidence acceptance rule.
                    confidence,tokens=latent_conf,latent_tokens
                    selected=active & (confidence>=threshold);reason[selected]={'latent':4,'combined':6,'no_geometry':7}[method]
                if method!='baseline' and not selected.any() and fallback=='top1':
                    selected[confidence.masked_fill(~active,-torch.inf).argmax()]=True
                    reason[selected]=5
                idx=torch.where(selected & active)[0]
                unchanged_steps+=(~(selected & active & (tokens!=mask)).any()).long()
                inserted=selected & active & (tokens!=mask)
                threshold_accepted=inserted & ((reason==2) | (reason==3) | (reason==4) | (reason==6) | (reason==7))
                boost_enabled=threshold_accepted & (raw_confidence<threshold)
                already_confident=threshold_accepted & (raw_confidence>=threshold)
                changed_candidate=threshold_accepted & (tokens!=raw_tokens)
                acceptance_counts+=torch.stack([boost_enabled.sum(),already_confident.sum(),
                    (inserted & (reason==5)).sum(),(inserted & (reason==1)).sum(),changed_candidate.sum()])
                x[0,start+idx]=tokens[idx]
                # Counts include attempted mask-valued proposals; completion exposes them.
                counts+=(reason[:,None]==reason_codes[None,:]).sum(0)
                if trace and idx.numel():
                    trace_parts.append(torch.stack([torch.full_like(idx,forwards-1).float(),
                        torch.full_like(idx,block_id).float(),torch.full_like(idx,local_step).float(),
                        (block_id*block+idx).float(),tokens[idx].float(),confidence[idx].float(),reason[idx].float(),
                        info['support'][idx] if info is not None else torch.full_like(idx,float('nan'),dtype=torch.float32),
                        info['region_id'][idx].float() if info is not None else torch.full_like(idx,-1).float(),
                        raw_confidence[idx],p[idx,tokens[idx]],raw_tokens[idx].float(),boost_enabled[idx].float()],dim=1))
                if early_stop and finalized_stop_mask(x[0,offset:],mask,stop_ids).any():
                    stopped=True
                    break
            if stopped: break
            if (x[0,start:stop]==mask).any():
                complete=False;break
    finally:
        if handle is not None: handle.remove()
    reasons={1:'scheduled',2:'confidence',3:'credit',4:'latent',5:'fallback',6:'combined',7:'no_geometry'}
    # All instrumentation transfers occur AFTER the denoising loop. Algorithmic
    # active/selection checks can still synchronize; this is not a CUDA graph.
    count_values=counts.cpu().tolist()
    count_dict={name:count_values[code] for code,name in reasons.items() if count_values[code]}
    if trace_parts:
        for step,b,local,pos,token,score,why,support,region_id,raw_max_p,raw_selected_p,raw_top1,boost_enabled in torch.cat(trace_parts).cpu().tolist():
            commits.append(dict(step=int(step),block=int(b),block_step=int(local),position=int(pos),
                token=int(token),score=score,reason=reasons[int(why)],latent_support=support if math.isfinite(support) else None,region_id=int(region_id),
                raw_max_p=raw_max_p,raw_selected_p=raw_selected_p,raw_top1=int(raw_top1),
                boost_enabled=bool(boost_enabled)))
    attribution_keys=('boost_enabled','already_confident','fallback','scheduled','changed_candidate')
    attribution=dict(zip(attribution_keys,acceptance_counts.cpu().tolist()))
    geometry_counts=dict(zip(('new','reused','invalid'),region_counts.cpu().tolist())) if use_latent else None
    stop_position=None
    if stopped:
        stop_position=int(torch.where(finalized_stop_mask(x[0,offset:],mask,stop_ids))[0][0])
    prefix_end=stop_position if stopped else length
    return dict(tokens=x,forwards=forwards,complete=complete,unresolved_masks=int((x[0,offset:]==mask).sum()),
                unresolved_prefix_masks=int((x[0,offset:offset+prefix_end]==mask).sum()),
                early_stop_enabled=early_stop,eos_stopped=stopped,stop_position=stop_position,
                stop_reason='eos' if stopped else ('length' if complete else 'budget'),
                counts=count_dict,acceptance_counts=attribution,region_observation_counts=geometry_counts,commits=commits,unchanged_context_steps=int(unchanged_steps))
