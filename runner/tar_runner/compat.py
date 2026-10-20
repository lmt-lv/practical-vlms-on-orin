"""Compatibility shims for model repos written against transformers 4.x.

transformers 4.x let a processor class declare `optional_attributes`: extra
`__init__` keyword arguments that ProcessorMixin set as plain attributes.
5.x dropped that and rejects any keyword that is not a sub-processor
("Unexpected keyword argument ..."). Molmo2's Hub processor relies on it
(`image_use_col_tokens` and four more flags). The shim restores the 4.x
behaviour, for classes that declare `optional_attributes` only: those
keywords are set as attributes before the 5.x constructor runs. No model
code or numerics change.

Applied in the runner's own process on import (see `__init__.py`) and in
every vLLM process through vLLM's general-plugin entry point (pyproject).
"""

from __future__ import annotations


def legacy_optional_attributes() -> None:
    try:
        from transformers.processing_utils import ProcessorMixin
    except ImportError:   # an environment without transformers (runner/photon)
        return

    if getattr(ProcessorMixin.__init__, "_tar_compat", False):
        return
    original = ProcessorMixin.__init__

    def __init__(self, *args, **kwargs):
        for name in getattr(type(self), "optional_attributes", ()):
            if name != "chat_template" and name in kwargs:
                setattr(self, name, kwargs.pop(name))
        original(self, *args, **kwargs)

    __init__._tar_compat = True
    ProcessorMixin.__init__ = __init__


def register() -> None:
    """vLLM general-plugin hook: runs in every vLLM process."""
    legacy_optional_attributes()
