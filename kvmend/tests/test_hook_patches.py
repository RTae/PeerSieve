"""The sitecustomize loader patches SGLang 0.5.1's four classes in a fresh interpreter (CPU only)."""
import os
import signal
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_patches_apply_in_a_fresh_interpreter():
    pytest.importorskip("sglang")
    env = dict(os.environ, KVMEND_ENABLE="1",
               PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "hook_site"), ROOT]))
    code = ("import sglang.srt.mem_cache.memory_pool as m, sglang.srt.mem_cache.allocator as a, "
            "sglang.srt.managers.scheduler as s, sglang.srt.layers.radix_attention as r, "
            "sglang.srt.mem_cache.radix_cache as c, sglang.srt.managers.schedule_batch as b\n"
            "print(m.MHATokenToKVPool.set_kv_buffer.__module__, m.MHATokenToKVPool.__init__.__module__, "
            "a.PagedTokenToKVPoolAllocator.free.__module__, s.Scheduler.process_batch_result.__module__, "
            "s.Scheduler.self_check_during_idle.__module__, s.Scheduler.flush_cache.__module__, "
            "s.Scheduler.stream_output_generation.__module__, r.RadixAttention.forward.__module__, "
            "c.RadixCache.cache_finished_req.__module__, b.Req.reset_for_retract.__module__)")
    # A new session, so that a timeout kills every process the import started, not only the interpreter (a child
    # left holding the output pipes would make the test wait for ever).
    proc = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        pytest.fail(f"importing SGLang with the hook took over 300 s:\n{stderr[-3000:]}")
    assert proc.returncode == 0, stderr[-3000:]
    assert stdout.split()[-10:] == ["kvmend.hook"] * 10, stdout[-2000:]
