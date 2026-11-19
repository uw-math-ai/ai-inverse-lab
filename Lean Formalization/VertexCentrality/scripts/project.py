"""Shared blueprint data and local verification provenance."""
from pathlib import Path
import hashlib
import json
import subprocess
import yaml
import re

ROOT = Path(__file__).resolve().parents[1]


class NodeLoader(yaml.SafeLoader):
    """Reject ambiguous duplicate keys in blueprint metadata."""


def node_mapping(loader, mapping):
    result = {}
    for key_node, value_node in mapping.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError('Blueprint keys must be unique strings')
        result[key] = loader.construct_object(value_node, deep=True)
    return result


NodeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, node_mapping)


def nodes():
    """Read the shared YAML/LaTeX nodes without requiring a private coordinator package."""
    items = {}
    for path in sorted((ROOT/'blueprint/nodes').glob('*.tex')):
        if not path.resolve().is_relative_to((ROOT/'blueprint/nodes').resolve()):
            raise ValueError(f'Node escapes blueprint directory: {path.name}')
        pieces = path.read_text().split('---', 2)
        if len(pieces) != 3 or pieces[0].strip():
            raise ValueError(f'Missing frontmatter: {path.name}')
        node = yaml.load(pieces[1], Loader=NodeLoader)
        if not isinstance(node, dict):
            raise ValueError(f'Invalid metadata: {path.name}')
        for key in ('id', 'title', 'type', 'status', 'source'):
            if not isinstance(node.get(key), str) or not node[key].strip():
                raise ValueError(f'Missing {key}: {path.name}')
        if 'body' in node or 'path' in node or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]*', node['id']):
            raise ValueError(f'Invalid/reserved metadata: {path.name}')
        json.dumps(node, allow_nan=False)
        for key in ('dependencies', 'lean_declarations'):
            values = node.get(key)
            if (not isinstance(values, list) or
                    not all(isinstance(v, str) and v for v in values) or
                    len(values) != len(set(values))):
                raise ValueError(f'Invalid {key}: {path.name}')
        if not isinstance(node.get('order'), int) or not pieces[2].strip():
            raise ValueError(f'Missing order/body: {path.name}')
        if node['id'] in items:
            raise ValueError(f'Duplicate node: {node["id"]}')
        items[node['id']] = dict(node, path=path.name, body=pieces[2].removeprefix('\n'))
    if not items:
        raise ValueError('No blueprint nodes found')
    visiting, done = set(), set()
    def visit(key):
        if key not in items:
            raise ValueError(f'Unknown dependency: {key}')
        if key in visiting:
            raise ValueError(f'Dependency cycle: {key}')
        if key in done:
            return
        visiting.add(key)
        for dependency in items[key]['dependencies']:
            visit(dependency)
        visiting.remove(key)
        done.add(key)
    for key in items:
        visit(key)
    return sorted(items.values(), key=lambda n: n['order'])


def fingerprint():
    """Source bytes, not build artifacts; dirty edits invalidate old build evidence."""
    files = [ROOT / n for n in ('lean-toolchain', 'lakefile.toml', 'lake-manifest.json')]
    files += [f for f in ROOT.rglob('*.lean') if not any(
        part.startswith('.') for part in f.relative_to(ROOT).parts)]
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def swarm_snapshot():
    config_file = ROOT / '.local/swarm.json'
    if not config_file.exists():
        return dict(available=False, error='No coordinator configured')
    try:
        from swarmalize.client import Client
        config = json.loads(config_file.read_text())
        token = Path(config['token_file']).read_text().strip()
        result = Client(config['url'], token, timeout=2).call('overview')
        return dict(available=True, **result)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError):
        return dict(available=False, error='Coordinator unavailable; task state is not current')


def snapshot():
    evidence_file = ROOT / '.local/checks.json'
    evidence = json.loads(evidence_file.read_text()) if evidence_file.exists() else None
    if evidence:
        evidence['current'] = evidence.get('fingerprint') == fingerprint()
    items = nodes()
    counts = {status: sum(n['status'] == status for n in items)
              for status in sorted({n['status'] for n in items})}
    return dict(nodes=items, status_counts=counts, swarm=swarm_snapshot(),
                branch=git('branch', '--show-current'),
                revision=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain', '--', '.')),
                evidence=evidence, phase='Corrected reconstruction checked · see scope and axiom audit')
