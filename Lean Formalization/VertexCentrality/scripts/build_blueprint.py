"""Build leanblueprint in a nested monorepo project (its CLI assumes a Git-root Lake project)."""
from pathlib import Path
import os
import subprocess
import sys
from project import ROOT

# Make installed plasTeX and its helpers available even without shell activation.
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
subprocess.run([sys.executable, str(ROOT/'scripts/render_blueprint.py')], check=True)
for directory in ('web', 'print'):
    (ROOT/'blueprint'/directory).mkdir(exist_ok=True)
subprocess.run(['latexmk','-xelatex','-interaction=nonstopmode','-halt-on-error',
                '-outdir=../print','print.tex'], cwd=ROOT/'blueprint/src', check=True)
subprocess.run(['plastex','-c','plastex.cfg','web.tex'], cwd=ROOT/'blueprint/src', check=True)
