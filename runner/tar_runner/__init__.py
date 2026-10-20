"""TAR model runner."""

import os
import sys
from pathlib import Path

from .compat import legacy_optional_attributes

legacy_optional_attributes()


def _locked_cuda_home() -> None:
    """Kernels compiled at run time (FlashInfer's sampler) use this
    environment's own CUDA toolkit: the pip-installed nvidia/cu13 tree, with
    the compiler pinned in pyproject.toml. Otherwise FlashInfer takes
    whatever /usr/local/cuda the host has, which on the A100 was too old to
    build them. An explicit CUDA_HOME is replaced too, for the same
    reason; TAR_KEEP_CUDA_HOME=1 keeps it.

    The wheels ship libcudart.so.13 but not the unversioned libcudart.so
    that FlashInfer's `-lcudart` looks for, so a one-link directory in the
    environment provides it (FLASHINFER_EXTRA_LDFLAGS)."""
    if os.environ.get("TAR_KEEP_CUDA_HOME") == "1":
        return
    try:
        import nvidia
    except ImportError:   # the HF reference environments: nothing compiles there
        return
    for base in nvidia.__path__:
        home = Path(base) / "cu13"
        if not (home / "bin" / "nvcc").exists():
            continue
        os.environ["CUDA_HOME"] = str(home)
        link = Path(sys.prefix) / "tar-cuda-link"
        so = link / "libcudart.so"
        if not so.exists():
            link.mkdir(exist_ok=True)
            try:
                so.unlink(missing_ok=True)
                so.symlink_to(home / "lib" / "libcudart.so.13")
            except FileExistsError:   # another process made it first
                pass
        flags = os.environ.get("FLASHINFER_EXTRA_LDFLAGS", "")
        if f"-L{link}" not in flags:
            os.environ["FLASHINFER_EXTRA_LDFLAGS"] = f"{flags} -L{link}".strip()
        return


_locked_cuda_home()
