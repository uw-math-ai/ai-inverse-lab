"""Small proof-command language. No raw Lean commands or raw tactics are accepted."""
import re
from .catalog import facts

class GateError(ValueError):
    pass

DEFINITIONS = set("ℝ Type Set Function.Injective Submodule.span inner LinearMap AddCommGroup Module NormedAddCommGroup InnerProductSpace True False".split())
KEYWORDS = {"fun", "forall", "∀"}
SAFE_METHODS = {"symm", "trans", "mp", "mpr", "1", "2"}
DOC = """Submit proofs as lists of JSON commands. Raw Lean code is NOT accepted.
Allowed operations (required fields):
  {"op":"rw", "terms":["h1", "← h2"], "at":""} (at optional; blank means goal)
  {"op":"simp", "terms":["h1", "LinearMap.map_sub"], "at":""} (always simp ONLY)
  {"op":"exact", "term":"allowedLemma arg1 arg2"}; {"op":"apply", "term":"..."}
  {"op":"have", "name":"aux", "type":"proposition", "steps":[commands]}
  {"op":"intro", "names":["x","hx"]}
  {"op":"change", "term":"proposition"}
  {"op":"ring"}; {"op":"rfl"}; {"op":"assumption"}; {"op":"constructor"}
  {"op":"linarith", "terms":[]}; {"op":"nlinarith", "terms":[]}
  {"op":"norm_num", "terms":[]}; {"op":"field_simp", "terms":["hm"]}
  {"op":"all_goals", "steps":[commands]}
Terms may contain only local names, the supplied declaration names, ordinary arithmetic,
parentheses, numerals, underscores for implicit arguments, and .symm/.trans/.mp/.mpr.
No imports, axioms, sorry, quotations, arbitrary tactics, external search or filesystem tools.
Available arithmetic tactics and their library dependencies are trusted shared infrastructure.
Do not use simp with implicit lemmas: give explicit terms. Use a have subproof for intermediate results.
Every task is a separate theorem; a theorem proved for one task cannot be named in another.
Use ASCII names for newly introduced variables. Do not provide Lean snippets in the prose as a substitute for commands.
"""

def identifier(s):
    return isinstance(s,str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",s) and s not in {"by","sorry","axiom","theorem","def","let","do","match","import"}

def expression(s, locals_, allowed):
    if not isinstance(s,str) or not s or len(s)>3000:
        raise GateError("invalid expression length")
    if any(x in s for x in ['\n','\r',';','"',"'",'`','/--','/-','--','\\','#','{','}']):
        raise GateError("forbidden expression syntax")
    locals_=set(locals_)
    for bound in re.findall(r'fun\s+([A-Za-z][A-Za-z0-9_]*)\s*(?::|=>)',s):
        if not identifier(bound): raise GateError('invalid lambda binder')
        locals_.add(bound)
    # Complete lexical coverage; an unrecognized character is rejected, never ignored.
    # Permit only these postfix projections after a parenthesized expression.
    # A bare '.' is deliberately not legal: it could expose arbitrary declarations.
    pattern = r"[A-Za-zℝ][A-Za-z0-9_ℝ]*(?:\.[A-Za-z0-9_]+)*|\.(?:symm|trans|mp|mpr|1|2)(?![A-Za-z0-9_])|\d+|[\s_(),\[\]:+*/^=<>|@-]|[∀∈→ₗ↔¬≠≤≥←∧∨]"
    matches=list(re.finditer(pattern,s))
    if ''.join(m.group() for m in matches)!=s:
        raise GateError("unrecognized expression token")
    for m in matches:
        token=m.group()
        if not re.match(r"[A-Za-zℝ]",token): continue
        if token in allowed or token in DEFINITIONS or token in locals_ or token in KEYWORDS: continue
        pieces=token.split('.')
        base='.'.join(pieces[:-1])
        if len(pieces)>1 and pieces[-1] in SAFE_METHODS and (base in locals_ or base in allowed): continue
        raise GateError(f"declaration not allowed: {token}")
    return s

def compile_steps(steps, baseline, locals_, depth=0):
    if depth>8 or not isinstance(steps,list) or not 1<=len(steps)<=60:
        raise GateError("proof must contain 1..60 commands; nesting at most 8")
    names=set(locals_)
    allowed=set(facts(baseline))
    lines=[]
    for step in steps:
        if not isinstance(step,dict): raise GateError("command must be an object")
        op=step.get('op')
        def expr(s): return expression(s,names,allowed)
        if op in {'exact','apply','change'}:
            line=op+' '+expr(step.get('term'))
        elif op in {'ring','rfl','assumption','constructor'}:
            line=op
        elif op in {'rw','simp','norm_num','field_simp','linarith','nlinarith'}:
            terms=step.get('terms',[])
            if not isinstance(terms,list) or len(terms)>30: raise GateError('invalid terms')
            line=('simp only' if op=='simp' else op)+' ['+', '.join(expr(t) for t in terms)+']'
            at=step.get('at','')
            if at:
                if op not in {'rw','simp','norm_num','field_simp'} or at not in names: raise GateError('invalid location')
                line+=' at '+at
        elif op=='intro':
            new=step.get('names')
            if not isinstance(new,list) or not new or any(not identifier(n) or n in names for n in new): raise GateError('invalid introduced names')
            names.update(new); line='intro '+' '.join(new)
        elif op=='have':
            name=step.get('name')
            if not identifier(name) or name in names or name in allowed: raise GateError('invalid local name')
            typ=expr(step.get('type'))
            body=compile_steps(step.get('steps'),baseline,names,depth+1)
            line=f'have {name} : {typ} := by\n'+ '\n'.join('  '+s for s in body.splitlines())
            names.add(name)
        elif op=='all_goals':
            body=compile_steps(step.get('steps'),baseline,names,depth+1)
            line='all_goals\n'+'\n'.join('  '+s for s in body.splitlines())
        else:
            raise GateError(f'unsupported operation: {op}')
        lines.append(line)
    return '\n'.join(lines)
