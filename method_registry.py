"""Public methods retained for the paper's comparison and ablation experiments."""

METHODS = (
    'ChorusCoder',
    'ChorusCoder-CodeT5p110M',
    'ChorusCoder-CodeSageSmallV2',
    'InFile-NoRetrieval',
    'cAST-Prefix100',
    'cAST-Prefix100-CodeT5p110M',
    'cAST-Prefix100-CodeSageSmallV2',
    'RLCoder',
    'CodeRAG-cAST',
    'GrepRAG-0.6B',
    'AIRCoder-RRF',
    'MergeQuery',
    'Line3Select',
    'Prefix100All',
)

CHORUSCODER_METHODS = tuple(method for method in METHODS
                         if not method.startswith(('InFile-', 'GrepRAG-', 'AIRCoder-')))
PREFIX100_ALL = 'Prefix100All'

ENCODER_METHODS = ('ChorusCoder-CodeT5p110M', 'cAST-Prefix100-CodeT5p110M',
                   'ChorusCoder-CodeSageSmallV2', 'cAST-Prefix100-CodeSageSmallV2')
PAPER_METHODS = METHODS
