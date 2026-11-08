"""Load KVMend into every SGLang process, as GhostServe loads its hook: put this folder on PYTHONPATH and set
KVMEND_ENABLE=1. Each target module is patched right after Python imports it, so SGLang itself is not edited."""
import os
import sys

if os.environ.get("KVMEND_ENABLE") == "1":
    import importlib.abc

    _TARGETS = ("sglang.srt.mem_cache.memory_pool", "sglang.srt.mem_cache.allocator", "sglang.srt.managers.scheduler",
                "sglang.srt.layers.radix_attention", "sglang.srt.mem_cache.radix_cache",
                 "sglang.srt.managers.schedule_batch", "sglang.srt.managers.cache_controller",
                 "sglang.srt.mem_cache.memory_pool_host", "sglang.srt.layers.attention.triton_backend")

    class _KVMendFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _TARGETS:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
            run_module = spec.loader.exec_module

            def exec_module(module, _run=run_module, _name=name):
                _run(module)
                from kvmend.hook import PATCHES
                PATCHES[_name](module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _KVMendFinder())
