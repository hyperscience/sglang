import logging

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()

# deepgemm requires CUDA toolkit and will error out on Blackwell GPU if missing.
#   File "/.pyenv/versions/3.12.7/lib/python3.12/site-packages/deep_gemm/__init__.py", line 42, in _ensure_initialized
#     torch.ops.deep_gemm.init(library_root, _find_cuda_home())
#                                            ^^^^^^^^^^^^^^^^^
#   File "/.pyenv/versions/3.12.7/lib/python3.12/site-packages/deep_gemm/__init__.py", line 30, in _find_cuda_home
#     assert cuda_home is not None
#            ^^^^^^^^^^^^^^^^^^^^^
ENABLE_JIT_DEEPGEMM = False

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
