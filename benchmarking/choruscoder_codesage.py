"""Pinned CodeSage-small-v2 native EOS, official pooled output and normalization."""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from benchmarking.choruscoder_encoders import EncoderConfig, hub_options, local_checkpoint_identity


@dataclass(frozen=True)
class CodeSageEncoderConfig(EncoderConfig):
    kind: str = 'codesage_embedding'
    model: str = 'codesage/codesage-small-v2'
    revision: str = '4844c2f24b25e181aa43ca058cc73dd2622565c1'
    dimension: int = 1024
    pooling: str = 'official_pooler_output'
    normalization: str = 'l2'
    add_eos_token: bool = True
    add_bos_token: bool = False

    def __post_init__(self):
        if (self.kind != 'codesage_embedding' or not isinstance(self.model, str) or not self.model or
                not isinstance(self.revision, str) or len(self.revision) != 40 or
                any(c not in '0123456789abcdef' for c in self.revision) or
                any(type(v) is not int for v in (self.dimension, self.batch_size,
                                                 self.max_length, self.max_body_tokens)) or
                (self.dimension, self.batch_size, self.max_length, self.max_body_tokens) != (1024, 32, 512, 508) or
                self.pooling != 'official_pooler_output' or self.normalization != 'l2' or
                self.add_eos_token is not True or self.add_bos_token is not False):
            raise ValueError('Invalid CodeSage-small-v2 embedding encoder configuration')


def checkpoint_options(model_name, config):
    options = hub_options(model_name, config)
    if options:
        options['code_revision'] = config.revision
    return options


def load_tokenizer(model_name, config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True,
        add_eos_token=True, add_bos_token=False, **checkpoint_options(model_name, config))
    if (not hasattr(tokenizer, 'byte_decoder') or tokenizer.pad_token_id is None or
            tokenizer.eos_token_id is None or getattr(tokenizer, 'add_eos_token', None) is not True or
            getattr(tokenizer, 'add_bos_token', None) is not False):
        raise ValueError('CodeSage requires its native slow byte-BPE tokenizer with padding and terminal EOS')
    tokenizer.padding_side = 'right'
    return tokenizer


@contextmanager
def native_import_compatibility():
    # The official source imports Conv1D from its former location. Newer 4.x
    # exposes the identical class only in pytorch_utils. Alias during import,
    # then restore the library namespace; no model code or forward is replaced.
    from transformers import modeling_utils
    added = not hasattr(modeling_utils, 'Conv1D')
    if added:
        from transformers.pytorch_utils import Conv1D
        modeling_utils.Conv1D = Conv1D
    try:
        yield
    finally:
        if added:
            del modeling_utils.Conv1D


class CodeSageEmbeddingAdapter:
    def __init__(self, model_name, tokenizer, config):
        import torch
        from transformers import AutoModel
        self.torch, self.tokenizer, self.config = torch, tokenizer, config
        # The pinned official release ships a .bin checkpoint. Avoid background conversion/PR probes.
        with native_import_compatibility():
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                torch_dtype=torch.float32, use_safetensors=False,
                **checkpoint_options(model_name, config)).cuda().eval()
        if self.model.config.model_type != 'codesage' or self.model.config.hidden_size != config.dimension:
            raise ValueError('CodeSage-small-v2 requires its native CodeSage checkpoint with hidden size 1024')

    def namespace(self, model_name):
        from benchmarking.choruscoder_cast import tokenizer_identity
        identity = dict(format='choruscoder_codesage_small_v2_v1', model=model_name,
            commit=getattr(self.model.config, '_commit_hash', None), encoder=asdict(self.config),
            tokenizer=tokenizer_identity(self.tokenizer), precision='float32',
            tokenization=dict(padding_side='right', add_eos_token=True, add_bos_token=False),
            implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        local = local_checkpoint_identity(model_name)
        if local is not None:
            identity['local_checkpoint'] = local
        return json.dumps(identity, sort_keys=True)

    def encode(self, texts):
        torch, policy, tokenizer = self.torch, self.config, self.tokenizer
        vectors = []
        for start in range(0, len(texts), policy.batch_size):
            rows, masks = [], []
            for text in texts[start:start+policy.batch_size]:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > policy.max_body_tokens:
                    raise ValueError('Query cannot fit CodeSage; refusing silent truncation')
                row = tokenizer.build_inputs_with_special_tokens(ids)
                if not row or len(row) > policy.max_length or row != ids+[tokenizer.eos_token_id]:
                    raise ValueError('CodeSage inputs must retain native terminal EOS within the input budget')
                padding = policy.max_length-len(row)
                rows.append(row + [tokenizer.pad_token_id]*padding)
                # Literal source special tokens and the appended EOS are real inputs.
                masks.append([True]*len(row) + [False]*padding)
            values = torch.tensor(rows, dtype=torch.long, device='cuda')
            mask = torch.tensor(masks, dtype=torch.bool, device='cuda')
            with torch.no_grad():
                result = self.model(input_ids=values, attention_mask=mask, return_dict=True)
                vector = getattr(result, 'pooler_output', None)
                if not isinstance(vector, torch.Tensor) or vector.shape != (len(rows), policy.dimension):
                    raise ValueError('Invalid CodeSage native pooler_output; expected [batch, 1024]')
                vectors.append(torch.nn.functional.normalize(vector, p=2, dim=1).cpu().float())
        return torch.cat(vectors) if vectors else torch.empty((0, policy.dimension), dtype=torch.float32)
