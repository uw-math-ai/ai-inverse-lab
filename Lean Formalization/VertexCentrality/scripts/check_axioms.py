"""Fail CI on missing endpoint audits or nonstandard transitive axioms."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'propext', 'Classical.choice', 'Quot.sound'}


def validate(output, expected):
    records = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", output, re.S)
    audited = set()
    for name, values in records:
        axioms = {x.strip() for x in values.split(',') if x.strip()}
        if axioms - ALLOWED:
            raise ValueError(f'{name}: disallowed axioms {sorted(axioms - ALLOWED)}')
        audited.add(name)
    audited.update(re.findall(r"'([^']+)' does not depend on any axioms", output))
    if audited != expected:
        raise ValueError(f'Audit mismatch: missing={sorted(expected-audited)}, extra={sorted(audited-expected)}')
    return len(audited)


if __name__ == '__main__':
    # A green Lean invocation alone would accept a printed sorryAx: exercise rejection.
    assert validate("'Good' depends on axioms: [propext, Quot.sound]", {'Good'}) == 1
    for bad in ("'Bad' depends on axioms: [sorryAx]", ''):
        try:
            validate(bad, {'Bad'})
        except ValueError:
            pass
        else:
            raise AssertionError('Audit failed to reject invalid evidence')
    source = (ROOT/'scripts/audit_scope.lean').read_text()
    expected = set(re.findall(r'^#print axioms (\S+)', source, re.M))
    if not expected:
        raise ValueError('No audit endpoints configured')
    result = subprocess.run(['lake','env','lean','scripts/audit_scope.lean'], cwd=ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (ROOT/'.local').mkdir(exist_ok=True)
    (ROOT/'.local/axiom-audit.log').write_text(result.stdout)
    if result.returncode:
        print(result.stdout)
        raise SystemExit(result.returncode)
    print(f'Passed axiom audit for {validate(result.stdout, expected)} endpoints.')
