"""Independent configurations for the paper experiments."""
from dataclasses import dataclass, replace
import math
from benchmarking.choruscoder_encoders import EncoderConfig
from benchmarking.choruscoder_codesage import CodeSageEncoderConfig

@dataclass(frozen=True)
class ApiSupportConfig:
    """Frozen API-first rules, with one budget shared by every trial stage."""
    version: str = 'api_first_support_v1'
    trial_queries: int = 8
    maximum_raw_supports: int = 4
    interface_candidates: int = 32
    interface_tokens: int = 192
    minimum_similarity: float = 0.4
    minimum_margin: float = 0.015

    def __post_init__(self):
        if self.version == 'api_first_support_thresholds_v2':
            counts = (self.trial_queries, self.maximum_raw_supports, self.interface_candidates, self.interface_tokens)
            if counts != (8, 4, 32, 192) or any((type(v) is not int for v in counts)) or any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) for v in (self.minimum_similarity, self.minimum_margin))) or (not 0 <= self.minimum_similarity <= 1) or (not 0 <= self.minimum_margin <= 2):
                raise ValueError('Invalid API interface thresholds or eight-query/four-support policy')
            return
        expected = ('api_first_support_v1', 8, 4, 32, 192, 0.4, 0.015)
        values = (self.version, self.trial_queries, self.maximum_raw_supports, self.interface_candidates, self.interface_tokens, self.minimum_similarity, self.minimum_margin)
        if values != expected or any((type(v) is not int for v in values[1:5])):
            raise ValueError('API-first requires the frozen eight-query/four-support policy')

@dataclass(frozen=True)
class CallableSupportConfig:
    """One raw replacement after the API-first stages; share their query cap."""
    version: str = 'callable_replace_one_v1'
    maximum_extra_queries: int = 1
    maximum_combinations: int = 4

    def __post_init__(self):
        if self.version != 'callable_replace_one_v1' or type(self.maximum_extra_queries) is not int or self.maximum_extra_queries != 1 or (type(self.maximum_combinations) is not int) or (self.maximum_combinations != 4):
            raise ValueError('Callable replacement requires one extra query and four combinations')

@dataclass(frozen=True)
class SupportUnitConfig:
    version: str = 'visible_support_units_v1'
    operation_lines: int = 3
    max_operation_lines: int = 6
    max_operation_tokens: int = 96
    max_pair_candidates: int = 32
    max_pair_query_tokens: int = 256

    def __post_init__(self):
        for field in ('operation_lines', 'max_operation_lines', 'max_operation_tokens', 'max_pair_candidates', 'max_pair_query_tokens'):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f'Invalid support unit setting: {field}')
        if self.operation_lines > self.max_operation_lines:
            raise ValueError('Operation line goal exceeds its hard limit')

@dataclass(frozen=True)
class QueryContextConfig:
    """Tokenizer-counted prompt and repository budgets; None means remaining space."""
    version: str = 'query_context_v1'
    prompt_tokens: int = 4096
    crossfile_tokens: int = 1024
    evidence_body_tokens: int = 896
    infile_tokens: int | None = 2944
    chunk_tokens: int = 192
    overlap_tokens: int = 32

    def __post_init__(self):
        for name in ('prompt_tokens', 'crossfile_tokens', 'evidence_body_tokens', 'chunk_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'Context token budget must be a positive integer: {name}')
        if not self.evidence_body_tokens <= self.crossfile_tokens < self.prompt_tokens:
            raise ValueError('Evidence must fit the cross-file and total prompt budgets')
        if self.chunk_tokens > 508:
            raise ValueError("Repository chunks must fit UniXcoder's 508 content tokens")
        if type(self.overlap_tokens) is not int or not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError('Overlap must be non-negative and smaller than a chunk')
        if self.infile_tokens is not None and (type(self.infile_tokens) is not int or self.infile_tokens < 1):
            raise ValueError('In-file token cap must be positive or None for remaining space')

@dataclass(frozen=True)
class RecallQuotaConfig:
    """An optional immutable table indexed by actual query count (one-based)."""
    total: int = 20
    quotas_by_query_count: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        if type(self.total) is not int or self.total < 1:
            raise ValueError('Recall total must be a positive integer')
        try:
            table = tuple((tuple(row) for row in self.quotas_by_query_count))
        except TypeError as exc:
            raise ValueError('Recall quotas must be a table of integer rows') from exc
        if len(table) > 8:
            raise ValueError('Recall quota tables support at most eight queries')
        for count, row in enumerate(table, 1):
            if len(row) != count or any((type(n) is not int or n < 1 for n in row)) or sum(row) != self.total:
                raise ValueError('Recall quota rows must match query count and sum to total')
        object.__setattr__(self, 'quotas_by_query_count', table)

@dataclass(frozen=True)
class RepositoryChunkConfig:
    """Repository segmentation is independent of support/query segmentation."""
    version: str = 'cast_repository_token192_v1'
    mode: str = 'cast_token'
    max_tokens: int = 192
    overlap: int = 0
    expansion: bool = False
    upstream_commit: str = 'e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'

    def __post_init__(self):
        if self.mode != 'cast_token' or type(self.max_tokens) is not int or (not 1 <= self.max_tokens <= 508):
            raise ValueError('Invalid repository AST chunk configuration')
        if self.overlap != 0 or self.expansion is not False:
            raise ValueError('AST repository controls require zero overlap and no expansion')
        if self.upstream_commit != 'e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd':
            raise ValueError('Repository AST controls require the pinned upstream commit')

@dataclass(frozen=True)
class SemanticSupportConfig:
    """Prefix-only cosine selection, independent of structural far scoring."""
    version: str = 'prefix_target_cosine_v1'
    near_segments: int = 2
    extra_segments: int = 2
    similarity_threshold: float = 0.65
    query_tokens: int = 508

    def __post_init__(self):
        for name in ('near_segments', 'extra_segments', 'query_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'Invalid semantic support setting: {name}')
        if self.near_segments + self.extra_segments > 8 or self.query_tokens > 508:
            raise ValueError('Semantic supports exceed encoder/query limits')
        value = self.similarity_threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (not 0 <= value <= 1):
            raise ValueError('Support similarity threshold must be finite and in [0, 1]')

@dataclass(frozen=True)
class OpBlockChunkControlConfig:
    version: str = 'opblock_chunk_controls_v1'
    recursive_split: bool = False
    head_body: bool = False
    partial_overlap: bool = False
    full_prefix: bool = False
    target_tokens: int = 384
    query_tokens: int = 508

    def __post_init__(self):
        for name in ('recursive_split', 'head_body', 'partial_overlap', 'full_prefix'):
            if type(getattr(self, name)) is not bool:
                raise ValueError('Chunk switches must be boolean')
        if self.target_tokens != 384 or self.query_tokens != 508:
            raise ValueError('Chunk controls require the 384/508 encoder protection')

@dataclass(frozen=True)
class FusedSupportConfig:
    version: str = 'support_distance_semantic_fusion_v1'
    semantic_weight: float = 0.5
    distance_weight: float = 0.5
    distance_unit: str = 'gap_physical_lines'
    distance_curve: str = 'reciprocal_one_plus_distance'
    threshold_kind: str = 'fused'
    overlap_policy: str = 'allow_partial_exclude_union_covered'

    def __post_init__(self):
        weights = (self.semantic_weight, self.distance_weight)
        if any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) or (not 0 <= v <= 1) for v in weights)) or not math.isclose(sum(weights), 1):
            raise ValueError('Fusion weights must be finite, nonnegative and sum to one')
        curves = {('gap_physical_lines', 'reciprocal_one_plus_distance'), ('source_recency_rank', 'gaussian_rank_with_floor')}
        if (self.distance_unit, self.distance_curve) not in curves or self.threshold_kind != 'fused' or self.overlap_policy != 'allow_partial_exclude_union_covered':
            raise ValueError('Unknown fused support policy')
        if self.distance_curve == 'gaussian_rank_with_floor' and (not all((hasattr(self, k) for k in ('distance_floor', 'distance_amplitude', 'distance_decay')))):
            raise ValueError('Gaussian rank policy requires explicit curve parameters')

@dataclass(frozen=True)
class GaussianRankSupportConfig(FusedSupportConfig):
    version: str = 'support_gaussian_rank_fusion_v1'
    distance_unit: str = 'source_recency_rank'
    distance_curve: str = 'gaussian_rank_with_floor'
    distance_floor: float = 0.4
    distance_amplitude: float = 6.0
    distance_decay: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        values = (self.distance_floor, self.distance_amplitude, self.distance_decay)
        if any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) for v in values)) or self.distance_floor < 0 or self.distance_amplitude <= 0 or (self.distance_decay <= 0) or ((self.distance_unit, self.distance_curve) != ('source_recency_rank', 'gaussian_rank_with_floor')):
            raise ValueError('Invalid Gaussian rank curve parameters')

@dataclass(frozen=True)
class CallableSupportAblationConfig:
    """Controls applied around the complete callable selection pipeline."""
    version: str = 'callable_support_ablations_v1'
    support_partition: str = 'syntax'
    query_composition: str = 'merged'

    def __post_init__(self):
        if not self.version or (self.support_partition, self.query_composition) not in {('syntax', 'merged'), ('line3', 'independent')}:
            raise ValueError('Callable ablation must change partition or final query composition')

@dataclass(frozen=True)
class SupportAblationConfig:
    """Independent controls over support partition, selection and retrieval."""
    version: str = 'gaussrank_support_ablations_v1'
    support_partition: str = 'syntax'
    selection: str = 'gaussian'
    query_composition: str = 'independent'
    prefix_tokens: int | None = None

    def __post_init__(self):
        combinations = {('syntax', 'gaussian', 'merged', None), ('line3', 'gaussian', 'independent', None), ('syntax', 'prefix_all', 'independent', 100)}
        if self.prefix_tokens is not None and type(self.prefix_tokens) is not int or (self.support_partition, self.selection, self.query_composition, self.prefix_tokens) not in combinations:
            raise ValueError('A support ablation must change one registered strategy')

@dataclass(frozen=True)
class RLCoderConfig:
    version: str = 'rlcoder_official_top10_v2'
    upstream_commit: str = '164d8d88cde324a38f5da70c4f858cc4679ef08e'
    model: str = 'nov3630/RLRetriever'
    revision: str = 'ec587f5d8635462fff3cfb06f9a946148acda08b'
    query_lines: int = 20
    query_length: int = 256
    candidate_length: int = 512
    bm25_topk: int = 100
    retrieval_topk: int = 10
    index_cache_bytes: int = 64 * 1024 ** 2
    vector_cache_bytes: int = 256 * 1024 ** 2

    def __post_init__(self):
        if not self.version or not self.model or len(self.revision) != 40 or any((c not in '0123456789abcdef' for c in self.revision)) or (len(self.upstream_commit) != 40) or any((c not in '0123456789abcdef' for c in self.upstream_commit)) or any((type(v) is not int or v < 1 for v in (self.query_lines, self.bm25_topk, self.retrieval_topk, self.index_cache_bytes, self.vector_cache_bytes))) or (self.query_length != 256) or (self.candidate_length != 512):
            raise ValueError('Invalid RLCoder inference configuration')

@dataclass(frozen=True)
class CodeRAGQueryConfig:
    version: str = 'coderag_bounded_query_v1'
    source: str = 'full_visible_prefix'
    lines_per_chunk: int = 3
    chunk_tokens: int = 250
    query_tokens: int = 508

    def __post_init__(self):
        if self.source != 'full_visible_prefix' or not self.version or type(self.lines_per_chunk) is not int or (self.lines_per_chunk < 1) or (type(self.chunk_tokens) is not int) or (not 1 <= self.chunk_tokens <= 250) or (type(self.query_tokens) is not int) or (not self.chunk_tokens < self.query_tokens <= 508):
            raise ValueError('Invalid CodeRAG source/chunk/query configuration')

@dataclass(frozen=True)
class QueryProfile:
    candidate_mode: str
    target_mode: str
    query_mode: str
    max_segments: int = 4
    fixed_segments: bool = True
    strategy_version: str = 'paper_query_v1'
    candidate_lines: int | None = None
    scoring_mode: str = 'probe'
    selection_mode: str = 'independent_topk'
    support_unit_config: SupportUnitConfig | None = None
    context_config: QueryContextConfig | None = None
    repository_chunk_config: RepositoryChunkConfig | None = None
    semantic_support_config: SemanticSupportConfig | None = None
    recall_config: RecallQuotaConfig | None = None
    chunk_control_config: OpBlockChunkControlConfig | None = None
    prefix_query_tokens: int | None = None
    fused_support_config: FusedSupportConfig | None = None
    support_ablation_config: SupportAblationConfig | None = None
    coderag_query_config: CodeRAGQueryConfig | None = None
    rlcoder_config: RLCoderConfig | None = None
    api_support_config: ApiSupportConfig | None = None
    callable_support_config: CallableSupportConfig | None = None
    callable_ablation_config: CallableSupportAblationConfig | None = None
    encoder_config: EncoderConfig | None = None

    def __post_init__(self):
        if self.candidate_mode not in ('window', 'operation_units_exclude_cut', 'coderag_bounded_line3'):
            raise ValueError('Unsupported paper candidate policy')
        if self.query_mode not in ('single', 'multi_prefix') or not self.fixed_segments:
            raise ValueError('Paper methods require fixed single or multi-prefix queries')
        if self.scoring_mode not in ('none', 'source_recency', 'probe'):
            raise ValueError('Unsupported paper scoring policy')
        if self.selection_mode not in ('independent_topk', 'fused_distance_semantic'):
            raise ValueError('Unsupported paper selection policy')
        if not isinstance(self.context_config, QueryContextConfig) or not isinstance(self.recall_config, RecallQuotaConfig):
            raise ValueError('Context and recall policies are required')
        if self.repository_chunk_config is not None and (self.repository_chunk_config.max_tokens != self.context_config.chunk_tokens or self.repository_chunk_config.overlap != self.context_config.overlap_tokens):
            raise ValueError('Repository and context chunk budgets must agree')
        if self.encoder_config is not None and (not isinstance(self.encoder_config, EncoderConfig)):
            raise ValueError('Invalid encoder configuration')
        if self.callable_support_config is not None and self.api_support_config is None:
            raise ValueError('Callable replacement requires API-first support selection')
        if self.api_support_config is not None and (not isinstance(self.fused_support_config, GaussianRankSupportConfig) or self.max_segments != 4):
            raise ValueError('API-first requires four Gaussian-ranked support slots')
        if self.callable_ablation_config is not None and self.callable_support_config is None:
            raise ValueError('Callable ablations require the complete pipeline')
        if self.support_ablation_config is not None and self.support_ablation_config.selection != 'prefix_all':
            raise ValueError('Only the Prefix100All support control is registered')
        if self.prefix_query_tokens is not None and (self.prefix_query_tokens != 100 or self.query_mode != 'single'):
            raise ValueError('Prefix100 uses a 100-token single query')
        if self.rlcoder_config is not None and self.recall_config.total != self.rlcoder_config.bm25_topk:
            raise ValueError('RLCoder recall must match its first-stage quota')
CALLABLE_REPLACE_ONE_METHOD = 'ChorusCoder'
CODET5P110M_METHOD = 'ChorusCoder-CodeT5p110M'
CODESAGE_METHOD = 'ChorusCoder-CodeSageSmallV2'
PREFIX100_QUERY_METHOD = 'cAST-Prefix100'
PREFIX100_CODET5P110M_METHOD = 'cAST-Prefix100-CodeT5p110M'
PREFIX100_CODESAGE_METHOD = 'cAST-Prefix100-CodeSageSmallV2'
CODERAG_SELECT1_METHOD = 'CodeRAG-cAST'
RLCODER_METHOD = 'RLCoder'
CALLABLE_ABLATION_METHODS = ('MergeQuery', 'Line3Select')
PROFILES = {'ChorusCoder': QueryProfile(candidate_mode='operation_units_exclude_cut', target_mode='last_operation_unit', query_mode='multi_prefix', max_segments=4, fixed_segments=True, strategy_version='callable_replace_one_v1', candidate_lines=3, scoring_mode='source_recency', selection_mode='fused_distance_semantic', support_unit_config=SupportUnitConfig(version='visible_support_units_v1', operation_lines=3, max_operation_lines=6, max_operation_tokens=96, max_pair_candidates=32, max_pair_query_tokens=256), context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=192, overlap_tokens=0), repository_chunk_config=RepositoryChunkConfig(version='cast_repository_token192_v1', mode='cast_token', max_tokens=192, overlap=0, expansion=False, upstream_commit='e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'), semantic_support_config=SemanticSupportConfig(version='prefix_target_cosine_v1', near_segments=2, extra_segments=2, similarity_threshold=0.4, query_tokens=508), recall_config=RecallQuotaConfig(total=30, quotas_by_query_count=((30,), (15, 15), (12, 10, 8), (10, 8, 6, 6))), chunk_control_config=OpBlockChunkControlConfig(version='opblock_chunk_controls_v1', recursive_split=True, head_body=False, partial_overlap=True, full_prefix=True, target_tokens=384, query_tokens=508), prefix_query_tokens=None, fused_support_config=GaussianRankSupportConfig(version='support_gaussian_rank_fusion_v1', semantic_weight=0.5, distance_weight=0.5, distance_unit='source_recency_rank', distance_curve='gaussian_rank_with_floor', threshold_kind='fused', overlap_policy='allow_partial_exclude_union_covered', distance_floor=0.4, distance_amplitude=6.0, distance_decay=1.0), support_ablation_config=None, coderag_query_config=None, rlcoder_config=None, api_support_config=ApiSupportConfig(version='api_first_support_v1', trial_queries=8, maximum_raw_supports=4, interface_candidates=32, interface_tokens=192, minimum_similarity=0.4, minimum_margin=0.015), callable_support_config=CallableSupportConfig(version='callable_replace_one_v1', maximum_extra_queries=1, maximum_combinations=4), callable_ablation_config=None, encoder_config=None), 'ChorusCoder-CodeT5p110M': QueryProfile(candidate_mode='operation_units_exclude_cut', target_mode='last_operation_unit', query_mode='multi_prefix', max_segments=4, fixed_segments=True, strategy_version='callable_codet5p110m_v1', candidate_lines=3, scoring_mode='source_recency', selection_mode='fused_distance_semantic', support_unit_config=SupportUnitConfig(version='visible_support_units_v1', operation_lines=3, max_operation_lines=6, max_operation_tokens=96, max_pair_candidates=32, max_pair_query_tokens=256), context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=192, overlap_tokens=0), repository_chunk_config=RepositoryChunkConfig(version='cast_repository_token192_v1', mode='cast_token', max_tokens=192, overlap=0, expansion=False, upstream_commit='e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'), semantic_support_config=SemanticSupportConfig(version='prefix_target_cosine_v1', near_segments=2, extra_segments=2, similarity_threshold=0.4, query_tokens=508), recall_config=RecallQuotaConfig(total=30, quotas_by_query_count=((30,), (15, 15), (12, 10, 8), (10, 8, 6, 6))), chunk_control_config=OpBlockChunkControlConfig(version='opblock_chunk_controls_v1', recursive_split=True, head_body=False, partial_overlap=True, full_prefix=True, target_tokens=384, query_tokens=508), prefix_query_tokens=None, fused_support_config=GaussianRankSupportConfig(version='support_gaussian_rank_fusion_v1', semantic_weight=0.5, distance_weight=0.5, distance_unit='source_recency_rank', distance_curve='gaussian_rank_with_floor', threshold_kind='fused', overlap_policy='allow_partial_exclude_union_covered', distance_floor=0.4, distance_amplitude=6.0, distance_decay=1.0), support_ablation_config=None, coderag_query_config=None, rlcoder_config=None, api_support_config=ApiSupportConfig(version='api_first_support_thresholds_v2', trial_queries=8, maximum_raw_supports=4, interface_candidates=32, interface_tokens=192, minimum_similarity=0.4, minimum_margin=0.015), callable_support_config=CallableSupportConfig(version='callable_replace_one_v1', maximum_extra_queries=1, maximum_combinations=4), callable_ablation_config=None, encoder_config=EncoderConfig(kind='codet5p_embedding', model='Salesforce/codet5p-110m-embedding', revision='d9f3a534af4252f04b10cd9f78e05037542d10f6', dimension=256, batch_size=32, max_length=512, max_body_tokens=508)), 'ChorusCoder-CodeSageSmallV2': QueryProfile(candidate_mode='operation_units_exclude_cut', target_mode='last_operation_unit', query_mode='multi_prefix', max_segments=4, fixed_segments=True, strategy_version='callable_codesage_small_v2_v1', candidate_lines=3, scoring_mode='source_recency', selection_mode='fused_distance_semantic', support_unit_config=SupportUnitConfig(version='visible_support_units_v1', operation_lines=3, max_operation_lines=6, max_operation_tokens=96, max_pair_candidates=32, max_pair_query_tokens=256), context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=192, overlap_tokens=0), repository_chunk_config=RepositoryChunkConfig(version='cast_repository_token192_v1', mode='cast_token', max_tokens=192, overlap=0, expansion=False, upstream_commit='e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'), semantic_support_config=SemanticSupportConfig(version='prefix_target_cosine_v1', near_segments=2, extra_segments=2, similarity_threshold=0.4, query_tokens=508), recall_config=RecallQuotaConfig(total=30, quotas_by_query_count=((30,), (15, 15), (12, 10, 8), (10, 8, 6, 6))), chunk_control_config=OpBlockChunkControlConfig(version='opblock_chunk_controls_v1', recursive_split=True, head_body=False, partial_overlap=True, full_prefix=True, target_tokens=384, query_tokens=508), prefix_query_tokens=None, fused_support_config=GaussianRankSupportConfig(version='support_gaussian_rank_fusion_v1', semantic_weight=0.5, distance_weight=0.5, distance_unit='source_recency_rank', distance_curve='gaussian_rank_with_floor', threshold_kind='fused', overlap_policy='allow_partial_exclude_union_covered', distance_floor=0.4, distance_amplitude=6.0, distance_decay=1.0), support_ablation_config=None, coderag_query_config=None, rlcoder_config=None, api_support_config=ApiSupportConfig(version='api_first_support_thresholds_v2', trial_queries=8, maximum_raw_supports=4, interface_candidates=32, interface_tokens=192, minimum_similarity=0.4, minimum_margin=0.015), callable_support_config=CallableSupportConfig(version='callable_replace_one_v1', maximum_extra_queries=1, maximum_combinations=4), callable_ablation_config=None, encoder_config=CodeSageEncoderConfig(kind='codesage_embedding', model='codesage/codesage-small-v2', revision='4844c2f24b25e181aa43ca058cc73dd2622565c1', dimension=1024, batch_size=32, max_length=512, max_body_tokens=508, pooling='official_pooler_output', normalization='l2', add_eos_token=True, add_bos_token=False)), 'cAST-Prefix100': QueryProfile(candidate_mode='window', target_mode='window', query_mode='single', max_segments=1, fixed_segments=True, strategy_version='prefix_query_controls_v1', candidate_lines=None, scoring_mode='none', selection_mode='independent_topk', support_unit_config=None, context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=192, overlap_tokens=0), repository_chunk_config=RepositoryChunkConfig(version='cast_repository_token192_v1', mode='cast_token', max_tokens=192, overlap=0, expansion=False, upstream_commit='e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'), semantic_support_config=None, recall_config=RecallQuotaConfig(total=30, quotas_by_query_count=((30,), (15, 15), (12, 10, 8), (10, 8, 6, 6))), chunk_control_config=None, prefix_query_tokens=100, fused_support_config=None, support_ablation_config=None, coderag_query_config=None, rlcoder_config=None, api_support_config=None, callable_support_config=None, callable_ablation_config=None, encoder_config=None), 'cAST-Prefix100-CodeT5p110M': QueryProfile(candidate_mode='window', target_mode='window', query_mode='single', max_segments=1, fixed_segments=True, strategy_version='prefix_query_controls_v1', candidate_lines=None, scoring_mode='none', selection_mode='independent_topk', support_unit_config=None, context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=192, overlap_tokens=0), repository_chunk_config=RepositoryChunkConfig(version='cast_repository_token192_v1', mode='cast_token', max_tokens=192, overlap=0, expansion=False, upstream_commit='e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'), semantic_support_config=None, recall_config=RecallQuotaConfig(total=30, quotas_by_query_count=((30,), (15, 15), (12, 10, 8), (10, 8, 6, 6))), chunk_control_config=None, prefix_query_tokens=100, fused_support_config=None, support_ablation_config=None, coderag_query_config=None, rlcoder_config=None, api_support_config=None, callable_support_config=None, callable_ablation_config=None, encoder_config=EncoderConfig(kind='codet5p_embedding', model='Salesforce/codet5p-110m-embedding', revision='d9f3a534af4252f04b10cd9f78e05037542d10f6', dimension=256, batch_size=32, max_length=512, max_body_tokens=508)), 'cAST-Prefix100-CodeSageSmallV2': QueryProfile(candidate_mode='window', target_mode='window', query_mode='single', max_segments=1, fixed_segments=True, strategy_version='prefix_query_controls_v1', candidate_lines=None, scoring_mode='none', selection_mode='independent_topk', support_unit_config=None, context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=192, overlap_tokens=0), repository_chunk_config=RepositoryChunkConfig(version='cast_repository_token192_v1', mode='cast_token', max_tokens=192, overlap=0, expansion=False, upstream_commit='e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'), semantic_support_config=None, recall_config=RecallQuotaConfig(total=30, quotas_by_query_count=((30,), (15, 15), (12, 10, 8), (10, 8, 6, 6))), chunk_control_config=None, prefix_query_tokens=100, fused_support_config=None, support_ablation_config=None, coderag_query_config=None, rlcoder_config=None, api_support_config=None, callable_support_config=None, callable_ablation_config=None, encoder_config=CodeSageEncoderConfig(kind='codesage_embedding', model='codesage/codesage-small-v2', revision='4844c2f24b25e181aa43ca058cc73dd2622565c1', dimension=1024, batch_size=32, max_length=512, max_body_tokens=508, pooling='official_pooler_output', normalization='l2', add_eos_token=True, add_bos_token=False)), 'RLCoder': QueryProfile(candidate_mode='window', target_mode='window', query_mode='single', max_segments=1, fixed_segments=True, strategy_version='rlcoder_official_top10_v2', candidate_lines=None, scoring_mode='none', selection_mode='independent_topk', support_unit_config=None, context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=508, overlap_tokens=0), repository_chunk_config=None, semantic_support_config=None, recall_config=RecallQuotaConfig(total=100, quotas_by_query_count=()), chunk_control_config=None, prefix_query_tokens=None, fused_support_config=None, support_ablation_config=None, coderag_query_config=None, rlcoder_config=RLCoderConfig(version='rlcoder_official_top10_v2', upstream_commit='164d8d88cde324a38f5da70c4f858cc4679ef08e', model='nov3630/RLRetriever', revision='ec587f5d8635462fff3cfb06f9a946148acda08b', query_lines=20, query_length=256, candidate_length=512, bm25_topk=100, retrieval_topk=10, index_cache_bytes=67108864, vector_cache_bytes=268435456), api_support_config=None, callable_support_config=None, callable_ablation_config=None, encoder_config=None), 'CodeRAG-cAST': QueryProfile(candidate_mode='coderag_bounded_line3', target_mode='official_last_line_block', query_mode='single', max_segments=1, fixed_segments=True, strategy_version='coderag_bounded_query_v1', candidate_lines=None, scoring_mode='probe', selection_mode='independent_topk', support_unit_config=None, context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=192, overlap_tokens=0), repository_chunk_config=RepositoryChunkConfig(version='cast_repository_token192_v1', mode='cast_token', max_tokens=192, overlap=0, expansion=False, upstream_commit='e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'), semantic_support_config=None, recall_config=RecallQuotaConfig(total=30, quotas_by_query_count=((30,), (15, 15), (12, 10, 8), (10, 8, 6, 6))), chunk_control_config=None, prefix_query_tokens=None, fused_support_config=None, support_ablation_config=None, coderag_query_config=CodeRAGQueryConfig(version='coderag_bounded_query_v1', source='full_visible_prefix', lines_per_chunk=3, chunk_tokens=250, query_tokens=508), rlcoder_config=None, api_support_config=None, callable_support_config=None, callable_ablation_config=None, encoder_config=None), 'MergeQuery': QueryProfile(candidate_mode='operation_units_exclude_cut', target_mode='last_operation_unit', query_mode='multi_prefix', max_segments=4, fixed_segments=True, strategy_version='callable_support_ablations_v1', candidate_lines=None, scoring_mode='source_recency', selection_mode='fused_distance_semantic', support_unit_config=SupportUnitConfig(version='visible_support_units_v1', operation_lines=3, max_operation_lines=6, max_operation_tokens=96, max_pair_candidates=32, max_pair_query_tokens=256), context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=192, overlap_tokens=0), repository_chunk_config=RepositoryChunkConfig(version='cast_repository_token192_v1', mode='cast_token', max_tokens=192, overlap=0, expansion=False, upstream_commit='e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'), semantic_support_config=SemanticSupportConfig(version='prefix_target_cosine_v1', near_segments=2, extra_segments=2, similarity_threshold=0.4, query_tokens=508), recall_config=RecallQuotaConfig(total=30, quotas_by_query_count=((30,), (15, 15), (12, 10, 8), (10, 8, 6, 6))), chunk_control_config=OpBlockChunkControlConfig(version='opblock_chunk_controls_v1', recursive_split=True, head_body=False, partial_overlap=True, full_prefix=True, target_tokens=384, query_tokens=508), prefix_query_tokens=None, fused_support_config=GaussianRankSupportConfig(version='support_gaussian_rank_fusion_v1', semantic_weight=0.5, distance_weight=0.5, distance_unit='source_recency_rank', distance_curve='gaussian_rank_with_floor', threshold_kind='fused', overlap_policy='allow_partial_exclude_union_covered', distance_floor=0.4, distance_amplitude=6.0, distance_decay=1.0), support_ablation_config=None, coderag_query_config=None, rlcoder_config=None, api_support_config=ApiSupportConfig(version='api_first_support_v1', trial_queries=8, maximum_raw_supports=4, interface_candidates=32, interface_tokens=192, minimum_similarity=0.4, minimum_margin=0.015), callable_support_config=CallableSupportConfig(version='callable_replace_one_v1', maximum_extra_queries=1, maximum_combinations=4), callable_ablation_config=CallableSupportAblationConfig(version='callable_support_ablations_v1', support_partition='syntax', query_composition='merged'), encoder_config=None), 'Line3Select': QueryProfile(candidate_mode='operation_units_exclude_cut', target_mode='last_operation_unit', query_mode='multi_prefix', max_segments=4, fixed_segments=True, strategy_version='callable_support_ablations_v1', candidate_lines=3, scoring_mode='source_recency', selection_mode='fused_distance_semantic', support_unit_config=SupportUnitConfig(version='visible_support_units_v1', operation_lines=3, max_operation_lines=6, max_operation_tokens=96, max_pair_candidates=32, max_pair_query_tokens=256), context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=192, overlap_tokens=0), repository_chunk_config=RepositoryChunkConfig(version='cast_repository_token192_v1', mode='cast_token', max_tokens=192, overlap=0, expansion=False, upstream_commit='e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'), semantic_support_config=SemanticSupportConfig(version='prefix_target_cosine_v1', near_segments=2, extra_segments=2, similarity_threshold=0.4, query_tokens=508), recall_config=RecallQuotaConfig(total=30, quotas_by_query_count=((30,), (15, 15), (12, 10, 8), (10, 8, 6, 6))), chunk_control_config=OpBlockChunkControlConfig(version='opblock_chunk_controls_v1', recursive_split=True, head_body=False, partial_overlap=True, full_prefix=True, target_tokens=384, query_tokens=508), prefix_query_tokens=None, fused_support_config=GaussianRankSupportConfig(version='support_gaussian_rank_fusion_v1', semantic_weight=0.5, distance_weight=0.5, distance_unit='source_recency_rank', distance_curve='gaussian_rank_with_floor', threshold_kind='fused', overlap_policy='allow_partial_exclude_union_covered', distance_floor=0.4, distance_amplitude=6.0, distance_decay=1.0), support_ablation_config=None, coderag_query_config=None, rlcoder_config=None, api_support_config=ApiSupportConfig(version='api_first_support_v1', trial_queries=8, maximum_raw_supports=4, interface_candidates=32, interface_tokens=192, minimum_similarity=0.4, minimum_margin=0.015), callable_support_config=CallableSupportConfig(version='callable_replace_one_v1', maximum_extra_queries=1, maximum_combinations=4), callable_ablation_config=CallableSupportAblationConfig(version='callable_support_ablations_v1', support_partition='line3', query_composition='independent'), encoder_config=None), 'Prefix100All': QueryProfile(candidate_mode='operation_units_exclude_cut', target_mode='last_operation_unit', query_mode='multi_prefix', max_segments=30, fixed_segments=True, strategy_version='gaussrank_support_ablations_v1', candidate_lines=3, scoring_mode='source_recency', selection_mode='independent_topk', support_unit_config=SupportUnitConfig(version='visible_support_units_v1', operation_lines=3, max_operation_lines=6, max_operation_tokens=96, max_pair_candidates=32, max_pair_query_tokens=256), context_config=QueryContextConfig(version='query_context_cf3072_v1', prompt_tokens=4096, crossfile_tokens=3072, evidence_body_tokens=3072, infile_tokens=None, chunk_tokens=192, overlap_tokens=0), repository_chunk_config=RepositoryChunkConfig(version='cast_repository_token192_v1', mode='cast_token', max_tokens=192, overlap=0, expansion=False, upstream_commit='e8a4f85fa571db6c13a9901d0cab9ac2a9c979cd'), semantic_support_config=None, recall_config=RecallQuotaConfig(total=30, quotas_by_query_count=()), chunk_control_config=OpBlockChunkControlConfig(version='opblock_chunk_controls_v1', recursive_split=True, head_body=False, partial_overlap=True, full_prefix=True, target_tokens=384, query_tokens=508), prefix_query_tokens=None, fused_support_config=None, support_ablation_config=SupportAblationConfig(version='gaussrank_support_ablations_v1', support_partition='syntax', selection='prefix_all', query_composition='independent', prefix_tokens=100), coderag_query_config=None, rlcoder_config=None, api_support_config=None, callable_support_config=None, callable_ablation_config=None, encoder_config=None)}

def default_embedding_model(method):
    profile = PROFILES[method]
    if profile.encoder_config is not None:
        return profile.encoder_config.model
    return profile.rlcoder_config.model if profile.rlcoder_config else 'microsoft/unixcoder-base'

def effective_api_support_config(method, similarity=None, margin=None):
    policy = PROFILES[method].api_support_config
    if similarity is None and margin is None:
        return policy
    if policy is None or policy.version != 'api_first_support_thresholds_v2':
        raise ValueError('Interface threshold overrides require a method with configurable API selection')
    return replace(policy, minimum_similarity=policy.minimum_similarity if similarity is None else similarity, minimum_margin=policy.minimum_margin if margin is None else margin)

def segment_limit(method, value=None):
    profile = PROFILES[method]
    limit = profile.max_segments if value is None else value
    allowed = max(4, profile.max_segments)
    if not 1 <= limit <= allowed:
        raise ValueError(f'--query-max-segments must be in 1..{allowed}')
    if profile.fixed_segments and limit != profile.max_segments:
        raise ValueError(f'{method} requires --query-max-segments {profile.max_segments}')
    return limit

def supports_online_retrieval(method):
    return method in PROFILES and method != 'Prefix100All'

def context_config(method):
    return PROFILES[method].context_config

def recall_config(method):
    return PROFILES[method].recall_config

def support_segment_limit(method, value=None):
    return 0 if PROFILES[method].scoring_mode == 'none' else segment_limit(method, value)

def query_limit(method, value=None):
    profile = PROFILES[method]
    if profile.query_mode == 'single' or (profile.callable_ablation_config is not None and profile.callable_ablation_config.query_composition == 'merged'):
        return 1
    return segment_limit(method, value)

def selection_cache_method(method):
    return method

@dataclass(frozen=True)
class LineRepositoryChunkConfig:
    """Physical-line partition; no parser or upstream AST implementation."""
    version: str = 'repository_fixed_lines_v1'
    mode: str = 'fixed_lines'
    lines_per_chunk: int = 3
    max_tokens: int = 508
    overlap: int = 0
    oversize_policy: str = 'source_character_bisection'

    def __post_init__(self):
        if self.mode != 'fixed_lines' or type(self.lines_per_chunk) is not int or self.lines_per_chunk < 1 or (type(self.max_tokens) is not int) or (not 1 <= self.max_tokens <= 508) or (self.overlap != 0) or (self.oversize_policy != 'source_character_bisection'):
            raise ValueError('Invalid fixed-line repository configuration')
