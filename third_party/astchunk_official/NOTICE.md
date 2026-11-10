# Official cAST reference snapshot

Source: https://github.com/yilinjz/astchunk

Pinned version: 0.1.0; commit `e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd`.
The six files listed in upstream.json are byte-for-byte upstream files, verified
with SHA-256. LICENSE retains the upstream MIT license and copyright.

The runtime adaptation is benchmarking/choruscoder_cast.py. Its recursive
split/greedy merge logic references ASTChunkBuilder.assign_nodes_to_windows
and merge_adjacent_windows. It uses exact source-span tokenizer counts (192
UniXcoder content tokens), not the paper's 2000 non-whitespace character budget.
It preserves original bytes/trivia, retains oversize leaves with explicit
fallback splitting, and uses the existing parser compatibility API. It does
not import or install the upstream package or upgrade Tree-sitter.

Upstream rebuild_code synthesizes whitespace, and its recursion can discard
oversize leaves without children. This adaptation intentionally does neither.
Overlap and expansion are disabled. This is a budget-adapted cAST comparison,
not an end-to-end reproduction of the paper's models and experiment settings.

Citation: Yilin Zhang et al. (2025), cAST: Enhancing Code Retrieval-Augmented
Generation with Structural Chunking via Abstract Syntax Tree, Findings of EMNLP.
https://aclanthology.org/2025.findings-emnlp.430/
