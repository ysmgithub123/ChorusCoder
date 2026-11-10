"""Independent, immutable profiles for the official training-free AIRCoder."""
from dataclasses import dataclass
from types import MappingProxyType

METHOD = 'AIRCoder-RRF'
QWEN_METHOD = METHOD


@dataclass(frozen=True)
class AIRCoderProfile:
    version: str = 'aircoder_rrf_full_snapshot_online_v1'
    slice_size: int = 800
    num_rags: int = 10
    context_tokens: int = 4096
    crossfile_tokens: int = 3072
    dependency_input_tokens: int = 4000
    dependency_new_tokens: int = 100
    metrics: tuple = ('sparse', 'dense', 'file_es', 'file_dist', 'class_es',
                      'def_es', 'skelton', 'dataflow')
    chunk_memory_bytes: int = 64 * 1024**2
    vector_memory_bytes: int = 256 * 1024**2


@dataclass(frozen=True)
class QwenAIRCoderProfile(AIRCoderProfile):
    version: str = 'aircoder_rrf_qwen3_8b_staged_online_v1'
    dependency_model: str = 'Qwen/Qwen3-8B'
    dependency_prompt_format: str = 'official_qwen3_chat_left_v1'


PROFILES = MappingProxyType({METHOD: QwenAIRCoderProfile()})
METHODS = (QWEN_METHOD,)


def validate_target(benchmark, language, split=None, method=METHOD):
    if method not in PROFILES or benchmark not in ('cceval', 'recceval') or language != 'python' or split not in (None, '', 'python'):
        raise ValueError('AIRCoder-RRF requires cceval/python or recceval/python')
