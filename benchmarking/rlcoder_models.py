"""Lazy model loading; encoding matches the pinned RLCoder retriever."""
from choruscoder import digest


def tokenizer_identity(tokenizer):
    return digest(dict(vocab=tokenizer.get_vocab(), special=getattr(tokenizer, 'special_tokens_map', {}),
                       cls=tokenizer.cls_token, sep=tokenizer.sep_token, pad=tokenizer.pad_token_id))


def encode_inputs(text, tokenizer, length, definitions):
    return definitions['tokenize'](text, tokenizer, length, is_query=length == 256)


class Encoder:
    def __init__(self, policy):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained(policy['model'], revision=policy['revision']).cuda().eval()
        self.pad = self.model.config.pad_token_id

    def encode(self, rows):
        torch = self.torch
        results = []
        for start in range(0, len(rows), 32):
            inputs = torch.tensor(rows[start:start + 32], device='cuda')
            mask = inputs.ne(self.pad)
            with torch.inference_mode():
                tokens = self.model(inputs, attention_mask=mask)[0]
                pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
                results.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().float())
        return torch.cat(results).numpy()

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
