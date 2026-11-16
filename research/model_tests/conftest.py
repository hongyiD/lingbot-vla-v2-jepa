"""CPU tests for the LingBot model changes.

Triton has no macOS / CPU build, and the grouped-GEMM kernels query the CUDA device when imported. When Triton is
missing, stand-in modules let the model files import; none of their kernels is called by these tests.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GPU_ONLY = ("triton", "lingbotvla.ops.group_gemm")


class _Anything:
    def __mro_entries__(self, bases):
        return (object,)

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _Anything()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything()

    def __getitem__(self, item):
        return _Anything()


def _module_getattr(attr):
    if attr.startswith("__"):
        raise AttributeError(attr)
    return _Anything()


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if any(name == p or name.startswith(p + ".") for p in GPU_ONLY):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__getattr__ = _module_getattr
        module.__path__ = []
        module.__version__ = "3.0.0"
        return module

    def exec_module(self, module):
        pass


if importlib.util.find_spec("triton") is None:
    # torch, torchvision and transformers probe for Triton when first imported; let them see it is absent.
    import torch._dynamo  # noqa: F401
    import torch._inductor.utils  # noqa: F401
    import torchvision  # noqa: F401
    import transformers.utils.import_utils  # noqa: F401

    sys.meta_path.insert(0, _StubFinder())
    _find_spec = importlib.util.find_spec

    def _find_spec_without_triton(name, package=None):
        if name == "triton" or name.startswith("triton."):
            return None
        return _find_spec(name, package)

    importlib.util.find_spec = _find_spec_without_triton
