"""Check the local Lean project; record source-bound build evidence, never proof certification."""
from datetime import datetime, timezone
import json
import subprocess
from project import ROOT, fingerprint, git, nodes

nodes()  # Validate dependency graph first.
initial = fingerprint()
result = subprocess.run(['lake', 'build'], cwd=ROOT)
evidence = dict(time=datetime.now(timezone.utc).isoformat(), revision=git('rev-parse','HEAD'),
                fingerprint=initial, passed=result.returncode == 0,
                source_unchanged=initial == fingerprint(), command='lake build',
                scope='Project build only; may accept sorry. Not a theorem or axiom audit.')
(ROOT/'.local').mkdir(exist_ok=True)
target=ROOT/'.local/checks.json'
temp=target.with_suffix('.tmp')
temp.write_text(json.dumps(evidence,indent=2)+'\n')
temp.replace(target)
raise SystemExit(result.returncode or (0 if evidence['source_unchanged'] else 1))
