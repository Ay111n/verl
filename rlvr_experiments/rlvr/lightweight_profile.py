"""Bounded consecutive-microbatch diagnostics; no complete-update offload claim."""
import functools
import json
import os
from pathlib import Path
import time
import warnings

_installed = False


def selected_window(index, start=2, count=2):
    return start <= index < start + count


def install():
    global _installed
    if _installed:
        return
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    original = FSDPEngine.forward_backward_batch

    @functools.wraps(original)
    def observed(self, data, loss_function, forward_only=False):
        if forward_only:
            return original(self, data, loss_function, forward_only=forward_only)
        import torch
        from verl.utils import tensordict_utils as tu
        from .profiling import _native_stderr_capture, observed_trace_loss_messages
        from .trace_stream import file_sha256
        rank = torch.distributed.get_rank()
        update = int(tu.get(data, 'rlvr_update_id', default=0))
        run = Path(os.environ['RLVR_RUN_DIR'])
        scratch = Path('/dev/shm') / ('rlvr_light_' + run.name)
        scratch.mkdir(exist_ok=True)
        path = scratch / f'update{update}-rank{rank}-pid{os.getpid()}.json'
        directory = run / 'light_traces'
        directory.mkdir(exist_ok=True)
        if path.exists():
            raise RuntimeError('Refusing to overwrite a sampled trace')
        forward = self.forward_step
        index = 0
        profiler = marker = stderr_ctx = warn_ctx = None
        caught = []
        started = None
        sampled = 0

        def close(success):
            nonlocal profiler
            if profiler is None:
                return
            current = profiler
            profiler = None
            torch.cuda.synchronize()
            window_seconds = time.perf_counter() - started
            marker.__exit__(None, None, None)
            try:
                current.stop()
                export_started = time.perf_counter()
                current.export_chrome_trace(str(path))
                export_seconds = time.perf_counter() - export_started
            finally:
                warn_ctx.__exit__(None, None, None)
                stderr_ctx.__exit__(None, None, None)
            flags = {'param_offload': bool(self.is_param_offload_enabled),
                     'optimizer_offload': bool(self.is_optimizer_offload_enabled)}
            warning_text = [str(item.message) for item in caught]
            stderr = path.with_suffix('.stderr.log')
            losses = observed_trace_loss_messages(stderr.read_text(errors='replace') + '\n' + '\n'.join(warning_text))
            record = {'schema': 'rlvr-light-capture-v1', 'phase': 'benchmark', 'update_id': update,
                      'rank': rank, 'pid': os.getpid(), 'trace_path': str(path),
                      'trace_sha256': file_sha256(path), 'trace_bytes': path.stat().st_size,
                      'capture_status': 'complete' if success and sampled == 2 and not losses else 'insufficient',
                      'scope': 'Two consecutive forward/backward microbatches; not a full actor update',
                      'microbatch_start_zero_based': 2, 'microbatch_count': sampled,
                      'export_seconds': export_seconds, 'elapsed_window_seconds': window_seconds,
                      'actor_offload_flags_actual': flags, 'python_warnings': warning_text,
                      'observed_trace_loss_messages': losses, 'stderr_path': str(stderr),
                      'full_update_offload_fraction_certified': False}
            if record['trace_bytes'] > 512 * 1024 * 1024:
                record['capture_status'] = 'insufficient'
                record['size_limit_exceeded'] = True
            (directory / f'update{update}-rank{rank}.capture.json').write_text(json.dumps(record, indent=2))
            profiler = None

        def forward_observed(*args, **kwargs):
            nonlocal index, profiler, marker, stderr_ctx, warn_ctx, caught, started, sampled
            if index == 4:
                close(True)
            if index == 2:
                torch.cuda.synchronize()
                stderr_ctx = _native_stderr_capture(path.with_suffix('.stderr.log'))
                stderr_ctx.__enter__()
                warn_ctx = warnings.catch_warnings(record=True)
                caught = warn_ctx.__enter__()
                warnings.simplefilter('always')
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                              torch.profiler.ProfilerActivity.CUDA],
                                                  record_shapes=False, profile_memory=False, with_stack=False)
                started = time.perf_counter()
                profiler.start()
                marker = torch.profiler.record_function('rlvr.light.microbatch_window')
                marker.__enter__()
            if selected_window(index):
                sampled += 1
            index += 1
            return forward(*args, **kwargs)

        self.forward_step = forward_observed
        success = False
        try:
            result = original(self, data, loss_function, forward_only=False)
            success = True
            return result
        finally:
            self.forward_step = forward
            close(success)

    FSDPEngine.forward_backward_batch = observed
    original_to = FSDPEngine.to

    @functools.wraps(original_to)
    def observed_to(self, device, model=True, optimizer=True, grad=True):
        started = time.perf_counter()
        result = original_to(self, device, model=model, optimizer=optimizer, grad=grad)
        record = {'phase': getattr(self, 'mode', None), 'device': str(device),
                  'model': bool(model), 'optimizer': bool(optimizer), 'grad': bool(grad),
                  'has_optimizer': self.optimizer is not None,
                  'duration_seconds': time.perf_counter() - started}
        with (Path(os.environ['RLVR_RUN_DIR']) / f'engine_transfer_calls-pid{os.getpid()}.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        return result

    FSDPEngine.to = observed_to
    _installed = True
