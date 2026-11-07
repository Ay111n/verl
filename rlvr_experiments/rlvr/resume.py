"""Independent checkpoint resume utilities; no historical result dependency."""
import os,copy
from pathlib import Path

def extended_resume_plan(plan):
    if os.environ.get('RLVR_RESUME_SMOKE') != '1':
        return plan
    if len(plan['updates']) != 50:
        raise ValueError('Resume smoke requires the original 50-update contract')
    plan=copy.deepcopy(plan)
    extra=copy.deepcopy(plan['updates'][0]);extra['update_id']=51
    plan['updates'].append(extra);plan['train_updates']=51
    return plan

def audit_loaded_tensors(engine,path,rank):
    import torch
    def local(t):
        if hasattr(t,'to_local'):t=t.to_local()
        return t.detach().cpu()
    saved=torch.load(Path(path)/f'model_world_size_2_rank_{rank}.pt',weights_only=False)
    live=engine.module.state_dict()
    for name,t in saved.items():
        if name not in live or not torch.equal(local(t),local(live[name])):
            raise RuntimeError(f'Restored model/LoRA bytes mismatch: {name}')
    optim=torch.load(Path(path)/f'optim_world_size_2_rank_{rank}.pt',weights_only=False)
    now=engine.optimizer.state_dict()
    def compare(a,b):
        if isinstance(a,dict):
            return isinstance(b,dict) and set(a)==set(b) and all(compare(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):
            return isinstance(b,(tuple,list)) and len(a)==len(b) and all(compare(x,y) for x,y in zip(a,b))
        if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(local(a),local(b))
        return a==b
    if not compare(optim,now):raise RuntimeError('Restored optimizer tensors/groups mismatch')
    extra=torch.load(Path(path)/f'extra_state_world_size_2_rank_{rank}.pt',weights_only=False)
    if not compare(extra['lr_scheduler'],engine.lr_scheduler.state_dict()):
        raise RuntimeError('Restored full scheduler state mismatch')
    return {'model_lora_exact':True,'adapter_tensor_count':len(saved),
            'optimizer_all_tensors_exact':True,'scheduler_full_state_exact':True}
