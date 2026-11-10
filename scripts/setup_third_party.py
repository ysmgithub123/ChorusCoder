"""Fetch SHA-256-pinned external source files without importing upstream code.

Run once before experiments. For offline installation, --source-root
points to a directory containing AIRCoder/, GrepRAG/ and RLCoder/ snapshots.
Existing files are verified and never overwritten on a checksum mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = {
    'AIRCoder': 'aircoder_upstream.json',
    'GrepRAG': 'greprag_upstream.json',
    'RLCoder': 'rlcoder_upstream.json',
}


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe manifest path: {name}')
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes dependency directory: {name}')
    return target


def install(component, *, source_root=None, check=False, project_root=ROOT):
    manifest = json.loads((project_root / 'benchmarking' / COMPONENTS[component]).read_text(encoding='utf-8'))
    checksums = manifest.get('sha256', manifest.get('files'))
    destination = project_root / 'third_party' / component
    if not destination.resolve().is_relative_to(project_root.resolve()):
        raise ValueError('Dependency directory resolves outside the project')
    repository = manifest['repository'].removeprefix('https://github.com/').removesuffix('.git')
    if manifest['repository'] != 'https://github.com/' + repository:
        raise ValueError('Expected an HTTPS GitHub source repository')
    verified = 0
    for name, expected in checksums.items():
        target = safe_path(destination, name)
        if target.is_file():
            payload = target.read_bytes()
        elif check:
            raise FileNotFoundError(f'{component}/{name}: run python scripts/setup_third_party.py first')
        elif source_root is not None:
            payload = safe_path(Path(source_root) / component, name).read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/{repository}/{manifest["commit"]}/{name}'
            request = urllib.request.Request(url, headers={'User-Agent': 'ChorusCoder-source-setup'})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {component}/{name}; existing files are not overwritten')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also prevents overwriting a concurrently created file.
            with target.open('xb') as stream:
                stream.write(payload)
        verified += 1
    return {'component': component, 'commit': manifest['commit'], 'verified_files': verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', nargs='+', choices=tuple(COMPONENTS), default=list(COMPONENTS))
    parser.add_argument('--source-root', type=Path, help='Local directory containing upstream component snapshots')
    parser.add_argument('--check', action='store_true', help='Validate installed files without downloading')
    args = parser.parse_args()
    try:
        for component in args.component:
            print(json.dumps(install(component, source_root=args.source_root, check=args.check)))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
