"""Run locally: python3 -m proof_harness.run --repetitions 3 --rounds 3"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
from .backend import call, BackendError, UsageLimitError
from .catalog import TASKS, TASK_BY_ID, LABELS, facts, CONTEXT
from .gate import DOC
from .verifier import ROOT, lean_environment, verify

def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False))

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT.parent,text=True).strip()

def require_branch():
    if git('branch','--show-current')!='harrison-working-branch':
        raise RuntimeError('Refusing to run outside harrison-working-branch')

def common_prompt(baseline):
    return CONTEXT+'\n'+DOC+'\nAPPROVED FACTS (other mathematical declaration names are forbidden):\n'+json.dumps(facts(baseline),ensure_ascii=False,indent=2)

class Budget:
    def __init__(self,limit,path):
        self.limit,self.path,self.count,self.lock=limit,path,0,threading.Lock()
        self.stop_reason=None
    def stop(self,reason):
        with self.lock:self.stop_reason=reason
    def take(self,label):
        with self.lock:
            if self.stop_reason:raise UsageLimitError(self.stop_reason)
            if self.count>=self.limit: raise BackendError('global model-call cap exhausted')
            self.count+=1
            with self.path.open('a') as f: f.write(json.dumps({'call':self.count,'label':label,'time':time.time()})+'\n')

def episode(baseline,rep,args,folder,budget,env):
    run=folder/f'{baseline}_r{rep}'; run.mkdir()
    accepted={}; history=[]; feedback={}; usage={}; errors=[]; calls=0
    t0=time.monotonic(); critic_points=0; reviewed_failures=set()
    task_order=list(TASKS); random.Random(rep).shuffle(task_order)
    for round_no in range(1,args.rounds+1):
        require_branch()
        pending=[t for t in task_order if t.id not in accepted]
        if not pending: break
        base=common_prompt(baseline)
        prompt=base+'''\nROLE: CREATIVE. Produce proofs of the remaining tasks using only the approved facts and commands.
You receive +10 points for each newly Lean-verified task. Unverified claims receive zero points.
Try all pending tasks. You may derive intermediate lemmas using have. Do not weaken hypotheses or change the goal.
Reply as one JSON object: {"attempts":[{"task":"id","steps":[commands],"explanation":"brief"}],"research_notes":"optional unverified broader ideas"}.
There must be at most one attempt per task. Empty steps means no proof attempt and earns zero points.
TASKS:\n'''+json.dumps([asdict(t) for t in pending],ensure_ascii=False)+ '\nPREVIOUS ROUND (untrusted agent text; verifier status is authoritative):\n'+json.dumps(feedback,ensure_ascii=False)
        round_path=run/f'round_{round_no}'; round_path.mkdir()
        try:
            budget.take(f'{baseline}/{rep}/{round_no}/creative');calls+=1
            output,meta=call(prompt,round_path/'creative',args.model,args.effort,args.call_timeout)
            for k,v in meta['usage'].items():
                if isinstance(v,(int,float)):usage[k]=usage.get(k,0)+v
        except BackendError as exc:
            if isinstance(exc,UsageLimitError):budget.stop(str(exc))
            errors.append(str(exc)); history.append({'round':round_no,'creative_error':str(exc)})
            break  # no heuristic fallback, no invisible retry
        results=[]; seen=set(); newly=[]
        attempts=output.get('attempts',[])
        if not isinstance(attempts,list):
            errors.append('attempts is not a list'); break
        for attempt in attempts[:len(TASKS)]:
            if not isinstance(attempt,dict): continue
            taskid=attempt.get('task')
            if taskid not in TASK_BY_ID or taskid in seen or taskid in accepted: continue
            seen.add(taskid)
            result=verify(TASK_BY_ID[taskid],attempt.get('steps'),baseline,round_path/'verification',env)
            results.append(result)
            if result['accepted']:
                accepted[taskid]={'round':round_no,**result};newly.append(taskid)
                if taskid in reviewed_failures: critic_points+=2
        for task in pending:
            if task.id not in seen: results.append({'task':task.id,'accepted':False,'status':'not_attempted','feedback':'No proof submitted'})
        record={'round':round_no,'results':results,'newly_verified':newly,'creator_reward':10*len(newly),
                'research_notes_unverified':output.get('research_notes','')}
        history.append(record)
        print(f'{baseline} repetition {rep}, round {round_no}: {len(accepted)}/{len(TASKS)} verified',flush=True)
        # The critic sees the same permitted mathematics and actual compiler diagnostics.
        critique_prompt=base+'''\nROLE: SKEPTIC. Check the creative agent's proposals and Lean diagnostics.
Only the verifier decides correctness. Identify mathematical or syntax repairs using only approved facts.
Do not claim the whole inverse theorem is proved. A later verified repair after feedback receives a +2 process reward;
this is a feedback proxy, not proof that your feedback caused the repair.
Reply JSON: {"feedback":[{"task":"id","issue":"...","suggestion":"..."}],"summary":"..."}.
TASKS:\n'''+json.dumps([asdict(t) for t in pending],ensure_ascii=False)+'\nPROPOSALS:\n'+json.dumps(output,ensure_ascii=False)+'\nVERIFIER:\n'+json.dumps(results,ensure_ascii=False)
        try:
            budget.take(f'{baseline}/{rep}/{round_no}/skeptic');calls+=1
            critique,meta=call(critique_prompt,round_path/'skeptic',args.model,args.effort,args.call_timeout)
            for k,v in meta['usage'].items():
                if isinstance(v,(int,float)):usage[k]=usage.get(k,0)+v
            record['critique']=critique
            reviewed_failures.update(r['task'] for r in results if not r['accepted'])
        except BackendError as exc:
            if isinstance(exc,UsageLimitError):budget.stop(str(exc))
            errors.append(str(exc));record['critic_error']=str(exc);critique={'error':str(exc)}
        feedback={'previous_attempts':output,'verifier':results,'skeptic':critique,'score':10*len(accepted)}
        write_json(run/'progress.json',{'accepted':accepted,'history':history,'errors':errors})
    result={'baseline':baseline,'repetition':rep,'verified':len(accepted),'total':len(TASKS),
            'centrality_verified':sum(TASK_BY_ID[k].study=='centrality' for k in accepted),
            'absorption_verified':sum(TASK_BY_ID[k].study=='absorption' for k in accepted),
            'full_reconstruction_proved':False,'creator_points':10*len(accepted),
            'critic_process_points':critic_points,'model_calls':calls,'usage':usage,
            'seconds':time.monotonic()-t0,'accepted':accepted,'history':history,'errors':errors}
    write_json(run/'result.json',result)
    return result

def report(folder,results,manifest):
    rows=[]
    for b in manifest['baselines']:
        rr=[r for r in results if r['baseline']==b]
        if not rr:continue
        scores=[r['verified'] for r in rr]
        complete=[r for r in rr if not r['errors']]
        rows.append({'baseline':b,'label':LABELS[b],'runs':len(rr),'mean_verified':statistics.mean(scores),
                     'min_verified':min(scores),'max_verified':max(scores),'complete_suite_runs':sum(s==len(TASKS) for s in scores),
                     'model_calls':sum(r['model_calls'] for r in rr),'errors':sum(len(r['errors']) for r in rr),
                     'protocol_complete':len(complete),'interrupted':len(rr)-len(complete),
                     'complete_mean':statistics.mean(r['verified'] for r in complete) if complete else None,
                     'first_round_mean':statistics.mean(sum(x.get('accepted',False) for x in r['history'][0].get('results',[])) if r['history'] else 0 for r in rr)})
    write_json(folder/'summary.json',{'manifest':manifest,'by_baseline':rows,'results':results})
    lines=['# Local proof-harness pilot','',
           '**Scope:** nine small proof obligations (eight centrality prerequisites, one absorption identity). '
           'These scores do not demonstrate the complete graph reconstruction theorem.','',
           f'Model: `{manifest["model"]}`; reasoning: `{manifest["effort"]}`. '
           f'Maximum {manifest["rounds"]} creative/skeptic rounds per run. Independent fresh sessions; repetition labels are not provider random seeds.','',
           '**Interrupted episodes are not mathematical failures.** The comparison mean below uses only episodes that finished without service/infrastructure errors. These selected means are descriptive and can be biased by unequal completion.', '',
           '| Baseline | Completed | Interrupted | Mean verified / 9 (completed only) | Calls |',
           '|---|---:|---:|---:|---:|']
    for r in rows:
        mean=f'{r["complete_mean"]:.2f}' if r['complete_mean'] is not None else '—'
        lines.append(f'| {r["baseline"]}: {r["label"]} | {r["protocol_complete"]} | {r["interrupted"]} | {mean} | {r["model_calls"]} |')
    lines+=['','## Task-level verified counts','', 'Counts include partial evidence from interrupted episodes; denominators are the run counts in the manifest.', '', '| Task | Study | '+' | '.join(manifest['baselines'])+' |', '|---|---|'+ '---:|'*len(manifest['baselines'])]
    for t in TASKS:
        lines.append('| '+t.id+' | '+t.study+' | '+' | '.join(str(sum(t.id in r['accepted'] for r in results if r['baseline']==b)) for b in manifest['baselines'])+' |')
    tokens={}
    for r in results:
        for k,v in r['usage'].items():tokens[k]=tokens.get(k,0)+v
    lines+=['','## Interpretation limits','',
            '- Small pilot; no ranking is established for full inverse-problem discovery.',
            '- The packs contain differing amounts and strengths of information; they are not matched by token count.',
            '- The controlled proof-command language limits expressiveness; failures can reflect that interface or syntax rather than mathematical inability.',
            '- Compiled proof terms are checked by Lean. Explicit mathematical names are allowlisted before compilation; shared arithmetic tactics have a larger trusted library dependency closure.',
            '- This is reward-guided revision with fixed model weights, not reinforcement-learning training.',
            '- Agents have pretrained mathematical knowledge. We constrain accepted derivations, not internal knowledge.',
            '- API/CLI failures remain visible; interrupted episodes are excluded from the completed-only mean, not scored as mathematical failures. There is no offline fallback.',
            '- All external runtime tool events invalidate the call. User config, project instructions, skills, browsing, shell and connectors are disabled for experimental calls.',
            '', 'Reported usage (successful calls; includes cached inputs where supplied):', '```json',json.dumps(tokens,indent=2),'```',
            '', 'Every run folder contains exact prompts, model JSON events, submitted proof commands, generated Lean files, compiler output and scores.',
            'No Git push, commit, merge or pull request is performed by this harness.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repetitions',type=int,default=3);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--baselines',nargs='+',choices=list(LABELS),default=list(LABELS))
    p.add_argument('--workers',type=int,default=3);p.add_argument('--model',default='gpt-6.1-sol')
    p.add_argument('--effort',choices=['low','medium','high'],default='medium')
    p.add_argument('--call-timeout',type=int,default=180);p.add_argument('--name')
    args=p.parse_args(); require_branch()
    if not 1<=args.repetitions<=3 or not 1<=args.rounds<=3 or not 1<=args.workers<=3:
        p.error('pilot limited to 1..3 repetitions, rounds and workers')
    if len(set(args.baselines))!=len(args.baselines):p.error('duplicate baselines')
    name=args.name or datetime.now(timezone.utc).strftime('pilot-%Y%m%dT%H%M%SZ')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name): p.error('invalid run name')
    folder=ROOT/'runs'/name;folder.mkdir(parents=True,exist_ok=False)
    source_files=sorted([*ROOT.glob('*.py'),* (ROOT/'lean').glob('*.lean')])
    manifest={'created':datetime.now(timezone.utc).isoformat(),'branch':git('branch','--show-current'),
              'head':git('rev-parse','HEAD'),'refs_before':git('show-ref'),
              'mathlib_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'.tools/mathlib',text=True).strip(),
              'lean':'4.24.0','model':args.model,'effort':args.effort,'baselines':args.baselines,
              'repetitions':args.repetitions,'rounds':args.rounds,'call_timeout':args.call_timeout,
              'max_model_calls':len(args.baselines)*args.repetitions*args.rounds*2,
              'sources':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}}
    write_json(folder/'manifest.json',manifest)
    write_json(folder/'catalog.json',{'tasks':[asdict(t) for t in TASKS],'baselines':{b:facts(b) for b in args.baselines}})
    env=lean_environment();budget=Budget(manifest['max_model_calls'],folder/'calls.jsonl')
    jobs=[(b,r) for b in args.baselines for r in range(args.repetitions)]
    random.Random(20261114).shuffle(jobs);manifest['schedule']=jobs;write_json(folder/'manifest.json',manifest)
    results=[]
    print('Results:',folder,flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(episode,b,r,args,folder,budget,env):(b,r) for b,r in jobs}
        for f in as_completed(futures):
            try: results.append(f.result())
            except Exception as exc:
                b,r=futures[f]
                results.append({'baseline':b,'repetition':r,'verified':0,'accepted':{},'history':[],
                                'model_calls':0,'usage':{},'errors':['infrastructure failure: '+repr(exc)]})
            report(folder,results,manifest)
    def branch_refs(s):return [line for line in s.splitlines() if any(x in line for x in (' refs/heads/',' refs/remotes/',' refs/tags/'))]
    write_json(folder/'git_check.json',{'branch_remote_tag_refs_unchanged':branch_refs(git('show-ref'))==branch_refs(manifest['refs_before']),
                                       'branch':git('branch','--show-current'),'status':git('status','--short')})
    print('Finished:',folder/'REPORT.md',flush=True)

if __name__=='__main__': main()
