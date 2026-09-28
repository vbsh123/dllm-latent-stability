"""Observe the same region bank and logit fusion as actual rollout."""
import numpy as np
import torch
from .decoding import LatentSettings, RegionSupport, region_distribution


class PolicyHistory:
    def __init__(self,settings=None,threshold=.95):
        self.settings=settings or LatentSettings()
        self.threshold=threshold
        self.state=None

    @torch.inference_mode()
    def update(self,hidden,active,logits):
        h=torch.as_tensor(hidden,dtype=torch.float32)
        mask=torch.as_tensor(active,dtype=torch.bool,device=h.device)
        logits=torch.as_tensor(logits,dtype=torch.float32,device=h.device)
        if self.state is None: self.state=RegionSupport(h,self.settings)
        q,info=region_distribution(logits,h,mask,self.state)
        confidence,token=q.max(-1)
        scalar=dict(accept=mask & (confidence>=self.threshold),max_p=confidence,top1=token,
                    support=info['support'],close=info['close'],new_region=info['new_region'],
                    region_id=info['region_id'],region_count=info['region_count'])
        return {f'policy_{k}':np.asarray(v.detach().cpu().numpy().copy()) for k,v in scalar.items()}
