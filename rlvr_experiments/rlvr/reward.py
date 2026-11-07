"""Outcome-only R0 reward; validity is recorded as diagnostics."""
from __future__ import annotations
import json
import re
from typing import Any
from rlvr.verifier import VERIFIER_VERSION, verify
VALIDITY_FLAGS = ('thinking_unclosed', 'no_reliable_final', 'empty_final', 'extraction_failed', 'truncated')

def score_response(*, prompt, completion, domain, gold_answer, option_labels=(), profile='R0', finish_reason=None, output_tokens=None, max_response_length=4096, thinking_tokens=None, component_flags=None):
    if profile not in ('R0',):
        raise ValueError(f'Reward profile {profile} is reserved and not implemented')
    flags = component_flags or {}
    for name in ('compute_cost', 'prefix_value', 'consistency', 'structure'):
        if flags.get(name, False):
            raise ValueError(f'Reserved reward component cannot be enabled: {name}')
    parsed = verify(prompt, completion, domain, gold_answer, option_labels)
    if finish_reason not in (None, 'stop', 'length'):
        raise ValueError(f'Unresolved termination reason: {finish_reason!r}')
    if finish_reason is None and (output_tokens is None or max_response_length is None):
        raise ValueError('Termination validity needs native stop/length or observed token count and cap')
    if output_tokens is not None and (not 0 <= output_tokens <= max_response_length):
        raise ValueError('Observed output token count is outside the response contract')
    truncated = finish_reason == 'length' if finish_reason is not None else output_tokens >= max_response_length
    parsed['truncated'] = bool(truncated)
    parsed['final_present'] = parsed['final_region'] is not None
    depth = int(parsed['prefix_open'])
    for marker in re.findall('<think>|</think>', completion):
        depth = depth + 1 if marker == '<think>' else max(0, depth - 1)
    parsed['thinking_unclosed'] = depth > 0
    parsed['thinking_closed'] = bool(parsed['thinking_closed'] and (not parsed['thinking_unclosed']))
    parsed['validity_status_version'] = 'rlvr-validity-v2'
    if not all((type(parsed[name]) is bool for name in (*VALIDITY_FLAGS, 'thinking_closed', 'final_present', 'answer_extractable'))):
        raise ValueError('Validity fields must be resolved booleans; unknown is not valid')
    invalid = any((parsed[name] for name in VALIDITY_FLAGS))
    correctness = float(parsed['correctness'])
    components = {'correctness': {'enabled': True, 'raw': correctness, 'contribution': correctness}}
    info = {k: v for k, v in parsed.items() if k not in ('think_body',)}
    info.update({'reward_profile': profile, 'verifier_version': VERIFIER_VERSION, 'domain': domain, 'thinking_tokens': thinking_tokens, 'thinking_tokens_method': 'probe_v1_retokenized_body', 'output_tokens': output_tokens, 'response_length': output_tokens, 'finish_reason': finish_reason, 'validity_invalid': invalid, 'truncation_source': 'native_finish_reason' if finish_reason is not None else 'length_bound_fallback', 'component_r0': correctness, 'component_consistency': 0.0, 'component_structure': 0.0, 'r0_enabled': True, 'consistency_enabled': False, 'consistency_raw': None, 'structure_enabled': False, 'structure_raw': None, 'correctness_raw': correctness, 'validity_penalty_raw': int(invalid), 'total_reward': correctness, 'components_json': json.dumps(components, ensure_ascii=False, sort_keys=True)})
    return {'reward_score': info['total_reward'], 'reward_extra_info': info}

def _get(config, path, default=None):
    value = config
    for part in path.split('.'):
        if value is None or not hasattr(value, 'get'):
            return default
        value = value.get(part, default if part == path.split('.')[-1] else None)
    return value

class RLVRRewardManager:
    """Duck-compatible RewardManagerBase interface without importing GPU runtime on CPU."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None, **kwargs):
        self.config, self.tokenizer = (config, tokenizer)
        self.profile = _get(config, 'rlvr.reward_profile', 'R0')
        self.component_flags = _get(config, 'rlvr.component_flags', {})
        self.max_response_length = _get(config, 'data.max_response_length', 4096)

    async def run_single(self, data) -> dict[str, Any]:
        item = data[-1:][0]
        response_ids = item.batch['responses']
        response_width = response_ids.shape[-1]
        valid_length = int(item.batch['attention_mask'][-response_width:].sum().item())
        response = self.tokenizer.decode(response_ids[:valid_length], skip_special_tokens=False)
        prompt_ids = item.batch['prompts']
        prompt_mask = item.batch['attention_mask'][:prompt_ids.shape[-1]].bool()
        prompt = self.tokenizer.decode(prompt_ids[prompt_mask], skip_special_tokens=False)
        extra = dict(item.non_tensor_batch.get('extra_info', {}))
        tool = item.non_tensor_batch.get('tool_extra_fields', {}) or {}
        gold = item.non_tensor_batch['reward_model']['ground_truth']
        domain = extra.get('domain', item.non_tensor_batch['data_source'])
        preliminary = verify(prompt, response, domain, gold, extra.get('option_labels', ()))
        body = preliminary['think_body']
        think_tokens = len(self.tokenizer(body, add_special_tokens=False)['input_ids']) if body else 0
        result = score_response(prompt=prompt, completion=response, domain=domain, gold_answer=gold, option_labels=extra.get('option_labels', ()), profile=self.profile, finish_reason=tool.get('native_finish_reason', extra.get('finish_reason')), output_tokens=valid_length, max_response_length=int(tool.get('effective_max_tokens', self.max_response_length)), thinking_tokens=think_tokens, component_flags=self.component_flags)
        result['reward_extra_info'].update({'backend_abort': tool.get('backend_abort', False), 'effective_max_tokens': tool.get('effective_max_tokens', self.max_response_length)})
        result['reward_extra_info'].update({k: extra.get(k) for k in ('sample_id', 'split_role', 'difficulty_bucket', 'answer_type', 'option_count')})
        return result

    @classmethod
    def assemble_rm_scores(cls, data, scores):
        import torch
        width = data.batch['prompts'].size(1)
        lengths = data.batch['attention_mask'][:, width:].sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError('Empty response has no terminal token for reward placement')
        rewards = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        rewards[torch.arange(rewards.size(0), device=rewards.device), lengths - 1] = rewards.new_tensor(scores)
        return rewards
