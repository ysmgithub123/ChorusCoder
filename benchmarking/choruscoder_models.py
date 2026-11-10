"""GPU model lifecycle and persistent caches for ChorusCoder preparation."""
from pathlib import Path
import json
from choruscoder import digest

class Probe:

    def __init__(self, model_name='Salesforce/codet5p-220m', tokens=5, seed=123):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.torch, self.tokens = (torch, tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).cuda().eval()

    def __call__(self, prompts):
        scores = []
        for start in range(0, len(prompts), 4):
            inputs = self.tokenizer(prompts[start:start + 4], return_tensors='pt', padding='longest', truncation=True).to('cuda')
            with self.torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.tokens, return_dict_in_generate=True, output_scores=True, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                result = self.torch.zeros(len(prompts[start:start + 4]), device='cuda')
                for logits in outputs.scores:
                    result += self.torch.softmax(logits, dim=1).max(dim=1).values
                scores.extend(result.cpu().tolist())
            del inputs, outputs, result
        return scores

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()

class DenseIndex:

    def __init__(self, cache_dir, tokenizer, model_name='microsoft/unixcoder-base', encoder_config=None):
        import torch
        from transformers import RobertaModel
        from embedding_cache import SQLiteEmbeddingCache
        from benchmarking.choruscoder_cache_compat import UNIXCODER_VECTOR_FORMAT
        self.torch, self.tokenizer = (torch, tokenizer)
        self.encoder_adapter = None
        if encoder_config is None:
            self.model = RobertaModel.from_pretrained(model_name).cuda().eval()
            namespace = json.dumps({'format': UNIXCODER_VECTOR_FORMAT, 'model': model_name, 'commit': getattr(self.model.config, '_commit_hash', None)}, sort_keys=True)
        else:
            from benchmarking.choruscoder_encoder_registry import create_adapter
            self.encoder_adapter = create_adapter(model_name, tokenizer, encoder_config)
            self.model = self.encoder_adapter.model
            namespace = self.encoder_adapter.namespace(model_name)
        self.cache = SQLiteEmbeddingCache(Path(cache_dir) / f'embeddings_{digest(namespace)[:16]}.sqlite3', namespace)

    def encode(self, texts):
        if getattr(self, 'encoder_adapter', None) is not None:
            return self.encoder_adapter.encode(texts)
        vectors = []
        torch = self.torch
        for start in range(0, len(texts), 32):
            rows = []
            for text in texts[start:start + 32]:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > 508:
                    raise ValueError('Query cannot fit UniXcoder; refusing silent truncation')
                row = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token, '<encoder-only>', self.tokenizer.sep_token]) + ids + [self.tokenizer.sep_token_id]
                rows.append(row + [self.tokenizer.pad_token_id] * (512 - len(row)))
            values = torch.tensor(rows, device='cuda')
            mask = values.ne(self.tokenizer.pad_token_id)
            with torch.no_grad():
                tokens = self.model(values, attention_mask=mask)[0]
                vector = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
                vectors.append(torch.nn.functional.normalize(vector, p=2, dim=1).cpu())
        return torch.cat(vectors) if vectors else torch.empty((0, self.model.config.hidden_size))

    def retrieve(self, queries, files, quotas):
        import numpy as np
        import time
        self.last_index_seconds = 0.0
        chunks = []
        for file in files:
            ids = self.tokenizer.encode(file['text'], add_special_tokens=False)
            for start in range(0, len(ids), 160):
                end = min(start + 192, len(ids))
                text = self.tokenizer.decode(ids[start:end])
                chunks.append({'id': 'E' + digest([file['path'], start, end, text])[:12], 'path': file['path'], 'text': text, 'start_token': start, 'end_token': end})
                if end == len(ids):
                    break
        if not chunks:
            return [[] for _ in queries]
        keys = [self.cache.key(c['text']) for c in chunks]
        unique = dict(zip(keys, (c['text'] for c in chunks)))
        cached = self.cache.get_many(unique)
        missing = [key for key in unique if key not in cached]
        if missing:
            started = time.perf_counter()
            encoded = self.encode([unique[key] for key in missing]).numpy()
            values = dict(zip(missing, encoded))
            self.cache.put_many(values)
            cached.update(values)
            self.last_index_seconds = time.perf_counter() - started
        matrix = np.stack([cached[key] for key in keys])
        qvectors = self.encode([q['text'] for q in queries]).numpy()
        results = []
        for query, quota in zip(qvectors, quotas):
            scores = matrix @ query
            ranked = np.lexsort((np.arange(len(chunks)), -scores))[:quota]
            results.append([dict(chunks[int(i)], score=float(scores[i])) for i in ranked])
        return results

    def close(self):
        self.cache.close()
        self.encoder_adapter = None
        del self.model
        self.torch.cuda.empty_cache()
