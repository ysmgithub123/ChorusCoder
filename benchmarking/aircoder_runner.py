"""Shared runtime/ABI helpers for the retained AIRCoder Qwen3-8B runner."""
import importlib.metadata
import sys
from pathlib import Path

TREE_SITTER_VERSIONS = {
    "tree-sitter": "0.25.2",
    "tree-sitter-c-sharp": "0.23.1",
    "tree-sitter-java": "0.23.5",
    "tree-sitter-python": "0.25.0",
    "tree-sitter-typescript": "0.23.2",
}


def validate_tree_sitter_stack():
    installed = {}
    mismatches = []
    for package, expected in TREE_SITTER_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        installed[package] = actual
        if actual != expected:
            mismatches.append(f"{package}=={actual} (expected {expected})")
    if mismatches:
        install_spec = " ".join(f"'{name}=={version}'" for name, version in TREE_SITTER_VERSIONS.items())
        raise RuntimeError(
            "The AIRCoder Tree-sitter native packages do not match the validated ABI stack: "
            + ", ".join(mismatches)
            + ". Reinstall them in the AIRCoder environment with: "
            + f"python -m pip install --no-cache-dir --force-reinstall {install_spec}"
        )
    return installed


def import_aircoder(root):
    src = Path(root).resolve() / "src"
    if not (src / "retriever.py").exists():
        raise FileNotFoundError(
            f"AIRCoder source was not found at {src}. Run python scripts/setup_third_party.py first "
            "or pass --aircoder-root."
        )
    sys.path.insert(0, str(src))
    import prompts
    import utils_repo
    from astchunk.astchunk_builder import ASTChunkBuilder
    import retriever as retriever_module
    from utils_transformer import get_tokenizer, load_snt_transformer

    return prompts, utils_repo, ASTChunkBuilder, retriever_module, get_tokenizer, load_snt_transformer


def build_llm_struct_retriever(retriever_module, tokenizer, embedding_model, metrics, language):
    """Build the official RRF retriever without its unused native data-flow parser.

    The upstream StructRetriever initializer always creates a Tree-sitter
    DataFlowExtractor for Python.  In this adapter data-flow dependencies come
    from the LLM (`dataflow_src="llm"`), and StructRetriever.query never reads
    that parser.  Initializing it is therefore unnecessary and can crash in
    the native Tree-sitter extension on some server builds.
    """
    supported = {"sparse", "dense", "file_es", "file_dist", "class_es", "def_es", "skelton", "dataflow"}
    unexpected = sorted(set(metrics) - supported)
    if unexpected:
        raise ValueError(f"Unsupported AIRCoder metrics: {unexpected}")

    retriever = retriever_module.StructRetriever.__new__(retriever_module.StructRetriever)
    retriever.targets = list(metrics)
    retriever.sparse = retriever_module.SparseRetriever(tokenizer)
    retriever.dense = retriever_module.DenseRetriver(tokenizer, embedding_model)
    retriever.skelton = retriever_module.SkeltonRetriever(tokenizer, embedding_model)
    retriever.df_extractor = None
    retriever.lang = language
    retriever.dataflow_src = "llm"
    return retriever


def install_safe_ast_text_access(ASTChunkBuilder):
    """Avoid py-tree-sitter Node.text, which can segfault in native code.

    The replacement derives node text from the original UTF-8 source using
    Tree-sitter's byte offsets.  It preserves the exact source text while
    avoiding the unstable node_get_text binding used by AIRCoder's ASTChunk.
    """
    import contextvars

    import astchunk.astchunk as astchunk_module
    import astchunk.astchunk_builder as builder_module
    import astchunk.astnode as astnode_module
    import astchunk.preprocessing as preprocessing_module

    if getattr(ASTChunkBuilder, "_safe_text_access_installed", False):
        return

    active_source = contextvars.ContextVar("aircoder_ast_source", default=None)

    def source_text(source, node):
        if source is None:
            raise RuntimeError("AST source bytes are unavailable")
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    original_node_init = astnode_module.ASTNode.__init__

    def safe_node_init(self, ts_node, node_size, ancestors=[]):
        original_node_init(self, ts_node, node_size, ancestors)
        self._source_bytes = active_source.get()

    def safe_bcode(self):
        source = getattr(self, "_source_bytes", None)
        if source is None:
            source = active_source.get()
        if source is None:
            raise RuntimeError("AST source bytes are unavailable")
        return source[self.node.start_byte : self.node.end_byte]

    def safe_get_node_token_count(node, tokenizer):
        return len(tokenizer(source_text(active_source.get(), node))["input_ids"])

    def safe_extract_definitions(node, language):
        source = active_source.get()
        definitions = astchunk_module.TreeSitterDefs(language)
        class_type = definitions.cls_def()
        function_type = definitions.fn_def()
        classes = []
        functions = []
        worklist = [node]
        while worklist:
            current = worklist.pop()
            if current.type == class_type:
                classes.append(source_text(source, current).split("\n", 1)[0])
            elif current.type == function_type:
                functions.append(source_text(source, current).split("\n", 1)[0])
            worklist.extend(reversed(current.children))
        return {"classes": classes, "functions": functions}

    def safe_build_chunk_ancestors(self, node_ancestors):
        source = getattr(self.ast_window[0], "_source_bytes", None)
        if source is None:
            source = active_source.get()
        ancestors = []
        for node in node_ancestors:
            if node.type in {self.cls_def, self.fn_def}:
                ancestors.append(source_text(source, node).split("\n", 1)[0])
        return ancestors

    original_chunkify = ASTChunkBuilder.chunkify

    def safe_chunkify(self, code, **configs):
        token = active_source.set(str(code).encode("utf-8"))
        try:
            return original_chunkify(self, code, **configs)
        finally:
            active_source.reset(token)

    astnode_module.ASTNode.__init__ = safe_node_init
    astnode_module.ASTNode.bcode = property(safe_bcode)
    preprocessing_module.get_node_token_count = safe_get_node_token_count
    builder_module.get_node_token_count = safe_get_node_token_count
    astchunk_module.extract_definitions = safe_extract_definitions
    astchunk_module.ASTChunk.build_chunk_ancestors = safe_build_chunk_ancestors
    ASTChunkBuilder.chunkify = safe_chunkify
    ASTChunkBuilder._safe_text_access_installed = True
