"""Python grammar loading and parser identities for the paper benchmarks."""
from importlib import import_module, metadata
from pathlib import Path
import hashlib
ROOT = Path(__file__).resolve().parent
GRAMMARS = {'python': ('tree_sitter_python', 'language', 'python', 'python')}

def grammar_variant(language, path=None):
    if language not in GRAMMARS:
        raise ValueError(f'Unsupported query language: {language}')
    return language

def parser_identity(language):
    module, _, _, library_name = GRAMMARS[language]

    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    library = ROOT / 'utils/build' / f'{library_name}-lang-parser.so'
    return {'language': language, 'tree_sitter': version('tree-sitter'), 'grammar_package': version(module.replace('_', '-')), 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest() if library.is_file() else None}

def grammar_identities(language):
    return {language: parser_identity(grammar_variant(language))}

def make_parser(language='python'):
    from tree_sitter import Language, Parser
    module, symbol, old_name, library_name = GRAMMARS[grammar_variant(language)]
    if hasattr(Language, 'build_library'):
        library = ROOT / 'utils/build' / f'{library_name}-lang-parser.so'
        try:
            parser = Parser()
            parser.set_language(Language(str(library), old_name))
            return parser
        except (OSError, ValueError, TypeError) as exc:
            hint = ' Provide the project grammar library compatible with the installed Tree-sitter ABI.'
            raise RuntimeError(f'Cannot load {language} grammar from {library}.{hint}') from exc
    try:
        grammar = import_module(module)
        return Parser(Language(getattr(grammar, symbol)()))
    except (ImportError, AttributeError, OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Cannot load {language} grammar; install the compatible {module.replace('_', '-')} grammar package in this modern Tree-sitter environment") from exc
