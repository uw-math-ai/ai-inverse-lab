"""Compile only harness-generated Lean sources and retain the evidence."""
import json
import os
from pathlib import Path
import subprocess
import time
from .gate import compile_steps, GateError

ROOT=Path(__file__).resolve().parent
MATHLIB=ROOT/'.tools/mathlib'
ELAN=ROOT/'.tools/elan'
LEAN=ELAN/'toolchains/leanprover--lean4---v4.24.0/bin/lean'
LAKE=ELAN/'toolchains/leanprover--lean4---v4.24.0/bin/lake'

def lean_environment():
    env=os.environ.copy(); env['ELAN_HOME']=str(ELAN)
    # Pin the binaries, independently of any global Lean default.
    env['PATH']=str(LEAN.parent)+os.pathsep+env.get('PATH','')
    proc=subprocess.run([str(LAKE),'env','printenv','LEAN_PATH'],cwd=MATHLIB,env=env,
                        capture_output=True,text=True,check=True)
    paths=[str((MATHLIB/p).resolve()) if not Path(p).is_absolute() else p
           for p in proc.stdout.strip().split(os.pathsep) if p]
    env['LEAN_PATH']=os.pathsep.join([str(ROOT/'lean'),*paths])
    return env

def verify(task, steps, baseline, destination, env=None):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    result={'task':task.id,'accepted':False,'status':'gate_rejected','seconds':0}
    if steps == [] or steps is None:
        result.update(status='not_attempted',feedback='No proof submitted',seconds=time.monotonic()-start)
        (destination/(task.id+'.json')).write_text(json.dumps(result,indent=2))
        return result
    try:
        if len(json.dumps(steps))>32000: raise GateError('proof too large')
        proof=compile_steps(steps,baseline,task.locals.split())
    except (GateError,TypeError) as exc:
        result['feedback']=str(exc)
    else:
        name='Candidate_'+task.id
        source=('import Foundations\nset_option maxHeartbeats 400000\nset_option maxRecDepth 1024\n'
                +'open scoped RealInnerProductSpace\n'
                +f'theorem {name} {task.binders} : {task.goal} := by\n'
                +'\n'.join('  '+s for s in proof.splitlines())+f'\n#print axioms {name}\n')
        path=destination/(task.id+'.lean'); path.write_text(source)
        try:
            proc=subprocess.run([str(LEAN),str(path.resolve())],env=env or lean_environment(),
                                cwd=ROOT/'lean',capture_output=True,text=True,timeout=45)
            output=proc.stdout+proc.stderr
            (destination/(task.id+'.log')).write_text(output)
            accepted=proc.returncode==0 and 'sorryAx' not in output and 'depends on axioms' in output
            result.update(accepted=accepted,status='verified' if accepted else 'lean_rejected',
                          feedback=output[:6000],returncode=proc.returncode,
                          source=str(path.relative_to(ROOT)),proof_commands=steps)
        except subprocess.TimeoutExpired:
            result.update(status='verifier_timeout',feedback='Lean exceeded 45s; no points awarded')
    result['seconds']=time.monotonic()-start
    (destination/(task.id+'.json')).write_text(json.dumps(result,indent=2))
    return result
