"""Codex text-only subprocess adapter with isolated instructions and event audit."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

DISABLED = ['shell_tool','shell_snapshot','apps','plugins','remote_plugin','multi_agent',
            'multi_agent_v2','memories','hooks','browser_use','browser_use_external',
            'browser_use_full_cdp_access','computer_use','in_app_browser','image_generation',
            'view_image','skill_search','skill_mcp_dependency_install','goals','code_mode_host',
            'plugin_discovery_prefetch','recommended_plugins','sleep_tool']

class BackendError(RuntimeError): pass
class UsageLimitError(BackendError): pass

def call(prompt, directory, model='gpt-6.1-sol', effort='medium', timeout=180):
    directory=Path(directory).resolve(); directory.mkdir(parents=True,exist_ok=False)
    exe=shutil.which('codex')
    if not exe: raise BackendError('Codex CLI missing')
    (directory/'prompt.txt').write_text(prompt)
    result=directory/'response.txt'
    instructions=directory/'instructions.txt'
    instructions.write_text('You are a mathematical proof experiment participant. Return only the requested JSON. No external tools are permitted. Use only the supplied mathematical task and allowed declarations. All verification is mediated by the harness.')
    cmd=[exe,'exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check',
         '--sandbox','read-only','--json','--color','never','-m',model,'-C',str(directory),
         '-o',str(result),'-c',f'model_reasoning_effort="{effort}"',
         '-c','web_search="disabled"','-c','project_doc_max_bytes=0',
         '-c','approval_policy="never"', '-c','mcp_servers={}',
         '-c','model_instructions_file='+json.dumps(str(instructions)),
         '-c','suppress_unstable_features_warning=true',
         '--enable','skip_host_skill_discovery','--enable','omit_skill_descriptions']
    for feature in DISABLED: cmd+=['--disable',feature]
    cmd+=['-']
    # Do not read .env or forward API/service secrets from the parent process.
    env={k:v for k,v in os.environ.items() if k in {'HOME','PATH','USER','TMPDIR','LANG','TERM','SHELL','CODEX_HOME'}}
    start=time.monotonic()
    with (directory/'events.jsonl').open('w') as stdout, (directory/'stderr.txt').open('w') as stderr:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,
                              text=True,env=env,cwd=directory,start_new_session=True)
        try: proc.communicate(prompt,timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); proc.wait()
            raise BackendError(f'model call timed out after {timeout}s')
    events=[]
    for line in (directory/'events.jsonl').read_text().splitlines():
        try: events.append(json.loads(line))
        except json.JSONDecodeError: pass
    usage={}
    for event in events:
        message=event.get('message','') or (event.get('error') or {}).get('message','')
        if 'usage limit' in message.lower():
            raise UsageLimitError('Codex subscription usage limit reached; no retry or paid fallback')
        if event.get('type')=='turn.completed': usage=event.get('usage',{})
        item=event.get('item',{})
        if item.get('type')=='error' and 'unrecognized configuration' in item.get('message',''):
            raise BackendError('unrecognized configuration: '+item['message'])
        if event.get('type','').startswith('item.') and item.get('type') not in {'agent_message','reasoning','error',None}:
            raise BackendError('forbidden runtime tool/item: '+str(item.get('type')))
    if proc.returncode or not result.exists():
        raise BackendError(f'Codex failed (exit {proc.returncode}); see {directory.name}/stderr.txt')
    raw=result.read_text().strip()
    if raw.startswith('```'):
        raw='\n'.join(raw.splitlines()[1:-1])
    try: parsed=json.loads(raw)
    except json.JSONDecodeError as exc: raise BackendError('model returned invalid JSON') from exc
    if not isinstance(parsed,dict): raise BackendError('model response must be a JSON object')
    metadata={'seconds':time.monotonic()-start,'model':model,'effort':effort,'usage':usage,
              'tool_events':0,'command':cmd,'returncode':proc.returncode}
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return parsed,metadata
