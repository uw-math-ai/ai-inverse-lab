"""Shared blueprint data and local verification provenance."""
from pathlib import Path
import hashlib
import json
import subprocess
from swarmalize.blueprint import Blueprint
from swarmalize.client import Client

ROOT = Path(__file__).resolve().parents[1]


def nodes():
    blueprint = Blueprint(ROOT / 'blueprint/nodes')
    check = blueprint.blueprint_check()
    if not check['ok']:
        raise ValueError(check['diagnostics'])
    return sorted((blueprint.blueprint_get(n['id']) for n in blueprint.blueprint_list()),
                  key=lambda n: n['order'])


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
        config = json.loads(config_file.read_text())
        token = Path(config['token_file']).read_text().strip()
        result = Client(config['url'], token, timeout=2).call('overview')
        return dict(available=True, **result)
    except (OSError, ValueError, KeyError, RuntimeError):
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
