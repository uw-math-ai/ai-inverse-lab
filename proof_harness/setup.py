"""Install pinned dependencies locally without modifying global Lean defaults."""
import os
from pathlib import Path
import shutil
import subprocess
from .verifier import ROOT, MATHLIB, ELAN, LEAN, LAKE, lean_environment
from .run import require_branch

PIN='f897ebcf72cd16f89ab4577d0c826cd14afaafc7'

def main():
    require_branch();(ROOT/'.tools').mkdir(exist_ok=True)
    env=os.environ.copy();env['ELAN_HOME']=str(ELAN)
    if not LEAN.exists():
        elan=shutil.which('elan')
        if not elan: raise SystemExit('Install the elan launcher first; setup does not change your shell profile.')
        subprocess.run([elan,'toolchain','install','leanprover/lean4:v4.24.0'],env=env,check=True)
    if not MATHLIB.exists():
        subprocess.run(['git','clone','--depth','1','--branch','v4.24.0',
                        'https://github.com/leanprover-community/mathlib4.git',str(MATHLIB)],check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=MATHLIB,text=True).strip()
    if actual!=PIN:raise SystemExit('Mathlib revision mismatch; refusing to change an existing checkout')
    env['PATH']=str(LEAN.parent)+os.pathsep+env.get('PATH','')
    subprocess.run([str(LAKE),'exe','cache','get','Mathlib/Tactic.lean',
                    'Mathlib/Analysis/InnerProductSpace/Adjoint.lean',
                    'Mathlib/LinearAlgebra/Matrix/Charpoly/Basic.lean'],env=env,cwd=MATHLIB,check=True)
    subprocess.run([str(LEAN),'-o','Foundations.olean','Foundations.lean'],
                   cwd=ROOT/'lean',env=lean_environment(),check=True)
    print('Pinned baseline library compiled successfully.')

if __name__=='__main__':main()
