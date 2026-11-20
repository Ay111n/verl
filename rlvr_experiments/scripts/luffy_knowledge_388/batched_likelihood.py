"""Response-only log-probabilities with right-padded independent sequences."""
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def response_logps_batch(model,prompts,responses,chunk_size=512,temperature=0.6,return_raw=False,pad_token_id=0):
    assert len(prompts)==len(responses)>0 and all(prompts) and all(responses)
    assert chunk_size>0 and math.isfinite(float(temperature)) and temperature>0
    base=model.get_base_model() if hasattr(model,"get_base_model") else model
    device=next(base.parameters()).device
    input_lengths=[len(p)+len(r)-1 for p,r in zip(prompts,responses)]
    tokens=torch.full((len(prompts),max(input_lengths)),pad_token_id,dtype=torch.long,device=device)
    for i,(p,r,length) in enumerate(zip(prompts,responses,input_lengths)):
        tokens[i,:length]=torch.as_tensor(p+r[:-1],device=device)
    positions=torch.arange(tokens.shape[1],device=device).unsqueeze(0).expand_as(tokens)
    # Every sequence is right-padded and only real response positions contribute
    # to the loss. Causality prevents those positions from seeing any padding.
    # Omitting an unnecessary explicit mask preserves torch SDPA Flash/GQA;
    # an arbitrary padding mask otherwise expands KV heads and changes kernels.
    states=base.model(input_ids=tokens,position_ids=positions,use_cache=False).last_hidden_state
    hidden=torch.cat([states[i,len(p)-1:len(p)+len(r)-1] for i,(p,r) in enumerate(zip(prompts,responses))])
    lengths=[len(r) for r in responses]
    target=torch.as_tensor([token for r in responses for token in r],device=device)
    assert hidden.shape[0]==target.numel()==sum(lengths)
    result = response_token_logps(base.lm_head, hidden, target, chunk_size, temperature, return_raw)
    if return_raw:
        policy, raw = result
        return list(policy.split(lengths)), list(raw.split(lengths))
    return list(result.split(lengths))


def response_token_logps(head, hidden, target, chunk_size, temperature, return_raw=False):
    """Score pre-aligned causal states; a second loss may reuse their student forward."""
    assert len(hidden) == len(target) > 0
    def token_ce(h,t):
        logits=head(h).float()
        policy=-F.cross_entropy(logits if temperature==1 else logits/temperature,t,reduction="none")
        if not return_raw:return policy
        with torch.no_grad():raw=-F.cross_entropy(logits.detach(),t,reduction="none") if temperature!=1 else policy.detach()
        return torch.stack((policy,raw))
    chunks=[]
    for start in range(0,len(target),chunk_size):
        h=hidden[start:start+chunk_size];t=target[start:start+chunk_size]
        chunks.append(checkpoint(token_ce,h,t,use_reentrant=False) if torch.is_grad_enabled() else token_ce(h,t))
    if return_raw:
        combined = torch.cat(chunks, dim=1)
        return combined[0], combined[1].detach()
    return torch.cat(chunks)
