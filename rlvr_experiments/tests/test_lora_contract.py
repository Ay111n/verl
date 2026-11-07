"""Native PEFT CPU checks; these do not claim actual two-GPU LoRA throughput."""
import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from rlvr.peft_contract import TARGETS, trainability_summary, verify_adapter_optimizer, adapter_B_zero_observation, record_materialized_lora
from rlvr.trainer import compare_repeat_evaluation, repeat_comparison_gate
from rlvr.rollout import EngineObserver, _capture


class LoraContracts(unittest.TestCase):
    def test_native_peft_only_adapter_updates_and_base_remains_identical(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        import verl.workers.engine.fsdp.transformer_impl as native
        from rlvr.peft_contract import install_lora_audit_hook
        torch.manual_seed(42)
        base = Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            max_position_embeddings=32, use_cache=False)).to(torch.bfloat16)
        engine = SimpleNamespace(model_config=SimpleNamespace(lora_adapter_path=None,
            lora_rank=32, lora_alpha=64, target_modules=list(TARGETS), target_parameters=None,
            exclude_modules=None))
        install_lora_audit_hook(native)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"RLVR_RUN_DIR": directory}):
            model = native.FSDPEngine._build_lora_module(engine, base)
            audit = json.loads(next((Path(directory)/'peft_audit').glob('*.json')).read_text())
            self.assertTrue(audit['base_frozen'])
            self.assertTrue(audit['fresh_lora_B_zero'])
            self.assertEqual(audit['trainable_tensors'], 14)
        summary = trainability_summary(model)
        self.assertLess(summary['trainable_parameters'], summary['total_parameters'])
        frozen = {name: param.detach().clone() for name, param in model.named_parameters() if not param.requires_grad}
        adapters = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        # Larger lr is confined to this tiny synthetic CPU test to make changes
        # visible in BF16; experiment lr remains frozen at 1e-6.
        optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"RLVR_RUN_DIR": directory}):
            record_materialized_lora(SimpleNamespace(module=model, optimizer=optimizer))
            audit = json.loads(next((Path(directory)/'peft_audit').glob('*-materialized.json')).read_text())
            self.assertTrue(audit['fresh_lora_B_zero'])
            self.assertTrue(audit['optimizer_state_adapter_only'])
        ids = torch.tensor([[3, 5, 7, 11]])
        loss = model(ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        self.assertTrue(verify_adapter_optimizer(model, optimizer)['optimizer_state_adapter_only'])
        params = dict(model.named_parameters())
        self.assertTrue(all(torch.equal(value, params[name]) for name, value in frozen.items()))
        self.assertTrue(any(not torch.equal(value, params[name]) for name, value in adapters.items()))
        name = next(iter(frozen))
        params[name].grad = torch.zeros_like(params[name])
        with self.assertRaises(RuntimeError): verify_adapter_optimizer(model, optimizer)
        params[name].grad = None
        optimizer.state[params[name]] = {'step': torch.tensor(1.)}
        with self.assertRaises(RuntimeError): verify_adapter_optimizer(model, optimizer)
        params[name].requires_grad_(True)
        with self.assertRaises(RuntimeError): trainability_summary(model)

    def test_meta_parameter_numerical_check_is_deferred_until_materialized(self):
        import torch
        module = SimpleNamespace(named_parameters=lambda: [('base.q_proj.lora_B.default.weight',
            torch.nn.Parameter(torch.empty(4, 2, device='meta')))])
        self.assertIsNone(adapter_B_zero_observation(module))
        module.named_parameters = lambda: [('base.q_proj.lora_B.default.weight', torch.nn.Parameter(torch.zeros(4, 2)))]
        self.assertTrue(adapter_B_zero_observation(module))

    def test_three_configs_enable_same_lora_and_reject_full_parameter_override(self):
        from rlvr.launch import compose_config, validate_algorithm
        for name in ('smoke', 'reward_dev', 'formal_16k'):
            candidate = compose_config(name, [])
            self.assertEqual(candidate.rlvr.training_mode, 'lora')
            self.assertEqual(candidate.actor_rollout_ref.model.lora_rank, 32)
            self.assertEqual(candidate.actor_rollout_ref.model.lora_alpha, 64)
            self.assertEqual(list(candidate.actor_rollout_ref.model.target_modules), list(TARGETS))
            self.assertTrue(candidate.actor_rollout_ref.actor.checkpoint.save_lora_only)
            self.assertEqual(candidate.actor_rollout_ref.rollout.load_format, 'safetensors')
            self.assertTrue(candidate.actor_rollout_ref.rollout.layered_summon)
            if name != 'formal_16k': validate_algorithm(candidate)
        bad = compose_config('reward_dev', ['actor_rollout_ref.model.lora_rank=0'])
        with self.assertRaises(ValueError): validate_algorithm(bad)
        bad = compose_config('reward_dev', ['actor_rollout_ref.rollout.load_format=dummy'])
        with self.assertRaises(ValueError): validate_algorithm(bad)

    def test_sampled_token_and_reward_differences_are_observations(self):
        records = [{'sample_id': 's', 'rollout_index': 0, 'seed': 42,
                    'response_token_ids': [1, 2], 'native_finish_reason': 'stop',
                    'native_stop_reason': None, 'total_reward': 1.}]
        altered = copy.deepcopy(records)
        altered[0].update(response_token_ids=[1, 3], native_finish_reason='length', total_reward=0.)
        comparison = {'greedy': compare_repeat_evaluation(records, records),
                      'sampled_g8': compare_repeat_evaluation(records, altered)}
        gate = repeat_comparison_gate(comparison)
        self.assertTrue(gate['passed'])
        self.assertFalse(gate['sampled_exact_tokens_required'])
        self.assertEqual(gate['sampled_mismatching_responses_observed'], 1)
        altered[0]['seed'] = 43
        comparison['sampled_g8'] = compare_repeat_evaluation(records, altered)
        self.assertFalse(repeat_comparison_gate(comparison)['passed'])
        altered[0]['seed'] = 42
        comparison['greedy'] = compare_repeat_evaluation(records, altered)
        self.assertFalse(repeat_comparison_gate(comparison)['passed'])

    def test_native_rollout_observer_records_active_adapter(self):
        class Engine:
            async def generate(self, **kwargs):
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1], finish_reason='stop', stop_reason=None)])
        async def observe():
            capture = {}
            token = _capture.set(capture)
            try:
                params = SimpleNamespace(seed=42, max_tokens=4096)
                adapter = SimpleNamespace(lora_int_id=123, lora_name='actual-test-adapter')
                values = [value async for value in EngineObserver(Engine()).generate(
                    sampling_params=params, lora_request=adapter)]
                self.assertEqual(len(values), 1)
                self.assertEqual(capture['native_lora_request'], {
                    'enabled': True, 'lora_int_id': 123, 'lora_name': 'actual-test-adapter'})
            finally:
                _capture.reset(token)
        with mock.patch.dict(os.environ, {}, clear=True): asyncio.run(observe())


if __name__ == '__main__': unittest.main()
