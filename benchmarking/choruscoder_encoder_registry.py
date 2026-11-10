"""Encoder adapter dispatch and implementation identities."""
from pathlib import Path

from benchmarking import choruscoder_encoders as codet5p
from benchmarking.choruscoder_codesage import CodeSageEncoderConfig, CodeSageEmbeddingAdapter
from benchmarking.choruscoder_codesage import load_tokenizer as codesage_tokenizer


# Config type, native tokenizer loader, adapter constructor, implementation module.
ENCODERS = {
    'codet5p_embedding': (codet5p.EncoderConfig, codet5p.load_tokenizer,
                         codet5p.CodeT5pEmbeddingAdapter, 'choruscoder_encoders.py'),
    'codesage_embedding': (CodeSageEncoderConfig, codesage_tokenizer,
                          CodeSageEmbeddingAdapter, 'choruscoder_codesage.py'),
}


def specification(config):
    item = ENCODERS.get(config.kind)
    if item is None or type(config) is not item[0]:
        raise ValueError('Unsupported declarative encoder configuration')
    return item


def load_tokenizer(model_name, config=None):
    if config is None:
        return codet5p.load_tokenizer(model_name)
    return specification(config)[1](model_name, config)


def create_adapter(model_name, tokenizer, config):
    return specification(config)[2](model_name, tokenizer, config)


def implementation_sources(config):
    source = Path(__file__).with_name(specification(config)[3])
    # Keep the existing CodeT5+ configuration identity as well as its vector namespace.
    if config.kind == 'codet5p_embedding':
        return (source,)
    return (source, Path(__file__), Path(codet5p.__file__))
