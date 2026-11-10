"""CodeT5+ embedding adapter and UniXcoder tokenizer loading."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class EncoderConfig:
    kind: str = 'codet5p_embedding'
    model: str = 'Salesforce/codet5p-110m-embedding'
    revision: str = 'd9f3a534af4252f04b10cd9f78e05037542d10f6'
    dimension: int = 256
    batch_size: int = 32
    max_length: int = 512
    max_body_tokens: int = 508

    def __post_init__(self):
        if (self.kind != 'codet5p_embedding' or not self.model or
                not isinstance(self.revision, str) or len(self.revision) != 40 or
                any(c not in '0123456789abcdef' for c in self.revision) or
                any(type(v) is not int for v in (self.dimension, self.batch_size,
                                                 self.max_length, self.max_body_tokens)) or
                (self.dimension, self.batch_size, self.max_length, self.max_body_tokens) != (256, 32, 512, 508)):
            raise ValueError('Invalid CodeT5+ embedding encoder configuration')


def local_checkpoint_identity(model_name):
    root = Path(model_name)
    if not root.is_dir():
        return None
    return dict(path=str(root.resolve()), files=[[p.name, p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(root.iterdir()) if p.is_file() and
        (p.suffix in ('.bin', '.safetensors', '.py') or p.name in
         ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt',
          'special_tokens_map.json', 'added_tokens.json'))])


def hub_options(model_name, config):
    return {} if Path(model_name).is_dir() else {'revision': config.revision}


def load_tokenizer(model_name, config=None):
    if config is None:
        from transformers import RobertaTokenizer
        return RobertaTokenizer.from_pretrained(model_name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True,
                                             **hub_options(model_name, config))
    if not hasattr(tokenizer, 'byte_decoder') or tokenizer.pad_token_id is None:
        raise ValueError('CodeT5+ requires its native slow byte-BPE tokenizer with padding; '
                         'use a compatible Transformers 4.x environment (5.x removes this slow backend)')
    return tokenizer


class CodeT5pEmbeddingAdapter:
    def __init__(self, model_name, tokenizer, config):
        import torch
        from transformers import AutoModel
        self.torch, self.tokenizer, self.config = torch, tokenizer, config
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                              torch_dtype=torch.float32,
                                              **hub_options(model_name, config)).cuda().eval()
        if (self.model.config.model_type != config.kind or
                getattr(self.model.config, 'embed_dim', None) != config.dimension):
            raise ValueError('Checkpoint must be the CodeT5+ 110M embedding model with its 256-dimensional projection')

    def namespace(self, model_name):
        from benchmarking.choruscoder_cast import tokenizer_identity
        identity = dict(format='choruscoder_codet5p_embedding_v1', model=model_name,
                        commit=getattr(self.model.config, '_commit_hash', None),
                        encoder=asdict(self.config), tokenizer=tokenizer_identity(self.tokenizer),
                        precision='float32',
                        implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        # Local checkpoints lack a Hub commit; detect replacement without hashing
        # multi-hundred-MB weight files on every invocation.
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
                    raise ValueError('Query cannot fit CodeT5+; refusing silent truncation')
                row = tokenizer.build_inputs_with_special_tokens(ids)
                if not row or len(row) > policy.max_length:
                    raise ValueError('CodeT5+ special tokens exceed the input budget')
                padding = policy.max_length-len(row)
                rows.append(row + [tokenizer.pad_token_id]*padding)
                # A literal "<pad>" in source is a real input token. Mask only
                # the padding we appended, matching the native tokenizer.
                masks.append([True]*len(row) + [False]*padding)
            values = torch.tensor(rows, dtype=torch.long, device='cuda')
            mask = torch.tensor(masks, dtype=torch.bool, device='cuda')
            with torch.no_grad():
                # Official forward already projects the first position and L2
                # normalizes. Its result is a tensor, not a hidden-state tuple.
                vector = self.model(input_ids=values, attention_mask=mask)
                if not isinstance(vector, torch.Tensor) or vector.shape != (len(rows), policy.dimension):
                    raise ValueError('Invalid CodeT5+ embedding output shape')
                vectors.append(vector.cpu().float())
        return torch.cat(vectors) if vectors else torch.empty((0, policy.dimension), dtype=torch.float32)


ADAPTERS = {'codet5p_embedding': CodeT5pEmbeddingAdapter}
