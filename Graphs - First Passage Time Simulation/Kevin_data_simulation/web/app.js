'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const number = (value, digits=4) => value == null ? '—' : Number(value).toLocaleString(undefined, {maximumFractionDigits:digits});
const percent = (value, digits=2) => value == null ? 'undefined' : `${number(value,digits)}%`;
const probability = value => percent(value*100,2);
let graphs = [], runs = [], result = null, selectedId = null, view = 'pmf', refreshing = false;
const cache = new Map();
const activeStatuses = ['queued','running','cancelling'];

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}
function showError(error) { $('error').textContent = error.message || String(error); $('error').hidden = false; }
function clearError() { $('error').hidden = true; }
function graphConfig() { return graphs.find(graph => graph.id === $('graph').value); }
function configuration() {
  const graph = graphConfig();
  let q = 0;
  if ($('q-mode').value === 'uniform') q = Number($('uniform-q').value);
  if ($('q-mode').value === 'vertex') q = graph.vertices.map(vertex => Number($(`q-${vertex.id}`).value));
  return {name:$('name').value,graph:graph.id,T:Number($('T').value),N:Number($('N').value),q,
    seed:Number($('seed').value),dt:$('dt').value.trim().toLowerCase() === 'auto' ? 'auto' : Number($('dt').value)};
}
function vertexFields(values) {
  const graph = graphConfig();
  $('vertex-fields').innerHTML = graph.vertices.map(vertex => `<div><label for="q-${vertex.id}">Vertex ${vertex.id}</label><input id="q-${vertex.id}" type="number" min="0" max="1000000" step="any" value="${values?.[vertex.id] ?? 0}" required></div>`).join('');
}
function qMode() {
  const mode = $('q-mode').value;
  $('uniform-field').hidden = mode !== 'uniform';
  $('uniform-q').disabled = mode !== 'uniform';
  $('vertex-fields').hidden = mode !== 'vertex';
  $('vertex-fields').querySelectorAll('input').forEach(input => { input.disabled = mode !== 'vertex'; });
  updatePreview();
}
function updatePreview() {
  if (!graphs.length) return;
  const graph = graphConfig(), cfg = configuration();
  $('graph-description').textContent = graph.id === 'paper8' ? '2 × 4 ladder · w = 0.25 · μ = 1 · B = {1, 2, 3}' : '3 × 3 grid · w = 1 · μ = degree · B = {0, 1, 2}';
  const q = Array.isArray(cfg.q) ? cfg.q : Array(graph.vertices.length).fill(cfg.q);
  const rate = Math.max(...graph.vertices.map((vertex,i) => vertex.degree*graph.edge_weight/vertex.mu + q[i]));
  const h = cfg.dt === 'auto' ? Math.min(1,1/rate) : cfg.dt;
  const valid = q.every(value => Number.isFinite(value) && value >= 0 && value <= 1e6) && h > 0 && h <= 1 && h*rate <= 1+1e-12;
  $('kernel-note').textContent = valid ? `h = ${number(h,6)} · Physical cutoff = ${number(cfg.T*h,6)} · ${number(cfg.N*9,0)} total walks` : 'This potential / step size is invalid. Use q ≥ 0 and a sufficiently small h, or choose auto.';
  if (!result) { drawGraph(graph,q); populatePairs(graph.observed); }
}
function populatePairs(observed) {
  const oldSource = $('source').value, oldTarget = $('target').value;
  const options = observed.map(vertex => `<option value="${vertex}">${vertex}</option>`).join('');
  $('source').innerHTML = options; $('target').innerHTML = options;
  $('source').value = observed.includes(Number(oldSource)) && oldSource !== '' ? oldSource : observed[0];
  $('target').value = observed.includes(Number(oldTarget)) && oldTarget !== '' ? oldTarget : observed[1];
}
function drawGraph(graph,q) {
  const width=280,height=170, marginX=34, marginY=28;
  const position = vertex => [marginX+vertex.column*(width-2*marginX)/(graph.columns-1), marginY+vertex.row*(height-2*marginY)/(graph.rows-1)];
  const maxQ = Math.max(...q,0);
  const edges = graph.edges.map(([a,b]) => {const [x1,y1]=position(graph.vertices[a]),[x2,y2]=position(graph.vertices[b]);return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#b9c9b4" stroke-width="1.7"/>`;}).join('');
  const nodes = graph.vertices.map(vertex => {const [x,y]=position(vertex), observed=graph.observed.includes(vertex.id);return `<g><title>Vertex ${vertex.id}; μ=${vertex.mu}; q=${q[vertex.id]}; ${observed?'observed':'interior'}</title>${q[vertex.id]>0?`<circle cx="${x}" cy="${y}" r="14" fill="#dc9c61" opacity="${.12+.3*q[vertex.id]/maxQ}"/>`:''}<circle cx="${x}" cy="${y}" r="7" fill="${observed?'#fff':'#dbe8d6'}" stroke="${observed?'#2a7654':'#9eb294'}" stroke-width="${observed?2:1.2}"/><text x="${x}" y="${y+23}" text-anchor="middle" font-size="10" fill="#7a8e70">${vertex.id}</text></g>`;}).join('');
  $('graph-view').innerHTML = `<svg viewBox="0 0 ${width} ${height+20}" role="img" aria-label="${esc(graph.name)}, observed vertices ${graph.observed.join(', ')}"><title>${esc(graph.name)} with fixed edge weights and vertex measures</title>${edges}${nodes}</svg>`;
  $('graph-caption').textContent = `${graph.name} · ${result?'selected result':'configuration preview'}`;
  $('graph-facts').innerHTML = `<div><span>VERTICES / EDGES</span><strong>${graph.vertices.length} / ${graph.edges.length}</strong></div><div><span>OBSERVED VERTICES</span><strong>{${graph.observed.join(', ')}}</strong></div><div><span>EDGE WEIGHT w</span><strong>${graph.edge_weight}</strong></div><div><span>VERTEX MEASURE μ</span><strong>${graph.id==='paper8'?'Constant · 1':'Degree of each vertex'}</strong></div>`;
  $('potential-caption').textContent = maxQ>0 ? `Amber halos indicate q > 0. q = [${q.map(value=>number(value,3)).join(', ')}].` : 'Zero potential · original paper walk when h = 1.';
}
function selectedPair() {
  if (!result) return null;
  const source=Number($('source').value),target=Number($('target').value);
  const i=result.observed.indexOf(source),j=result.observed.indexOf(target);
  return {source,target,i,j,pair:result.pairs.find(p=>p.source===source && p.target===target)};
}
function renderResult(data) {
  result=data; cache.set(data.id,data); selectedId=data.id;
  $('result-eyebrow').textContent='COMPLETED EXPERIMENT';
  $('result-name').textContent=data.config.name;
  $('result-config').textContent=`${data.graph.name} · T=${data.config.T} steps · N=${number(data.config.N,0)} per pair · seed ${data.config.seed} · h=${number(data.config.dt,6)}`;
  $('metric-frne').textContent=percent(data.metrics.frne_percent,3);
  $('metric-absolute').textContent=number(data.metrics.max_absolute_error,5);
  $('metric-tail').textContent=probability(data.metrics.mean_no_hit_exact);
  $('metric-runtime').textContent=`${number(data.metrics.elapsed_seconds,2)} s`;
  $('exports').hidden=false;
  $('export-csv').href=`/api/runs/${data.id}/csv`; $('export-json').href=`/api/runs/${data.id}/json`;
  populatePairs(data.observed);drawGraph(data.graph,data.config.q);drawPair();drawConvergence();drawKernel();
}
function drawPair() {
  if (!result) return;
  const {source,target,i,j,pair}=selectedPair(),T=result.config.T;
  let exact=result.r_exact.map(row=>row[i][j]),empirical=result.r_empirical.map(row=>row[i][j]);
  const lower=result.ci95_lower.map(row=>row[i][j]),upper=result.ci95_upper.map(row=>row[i][j]);
  if(view==='cdf') {let a=0,b=0;exact=exact.map(value=>a+=value);empirical=empirical.map(value=>b+=value);}
  $('ci-legend').hidden=view==='cdf';
  const width=620,height=280,left=44,right=16,top=20,bottom=43;
  const plotWidth=width-left-right,plotHeight=height-top-bottom;
  const max=Math.min(1,Math.max(.02,...exact,...empirical,...(view==='pmf'?upper:[]))*1.13);
  const x=t=>T===1?left+plotWidth/2:left+(t-1)*plotWidth/(T-1),y=p=>top+plotHeight*(1-p/max);
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${view.toUpperCase()} of first passage from vertex ${source} to ${target}"><title>Exact and Monte Carlo ${view.toUpperCase()}, vertex ${source} to ${target}</title>`;
  for(let k=0;k<=4;k++){const value=max*k/4;svg+=`<line class="grid-line" x1="${left}" x2="${width-right}" y1="${y(value)}" y2="${y(value)}"/><text class="axis-label" x="${left-9}" y="${y(value)+3}" text-anchor="end">${number(value,3)}</text>`;}
  const ticks=[...new Set(Array.from({length:Math.min(7,T)},(_,k)=>Math.round(1+k*(T-1)/Math.max(1,Math.min(7,T)-1))))];
  ticks.forEach(t=>{svg+=`<text class="axis-label" x="${x(t)}" y="${height-bottom+19}" text-anchor="middle">${t}</text>`;});
  const path=values=>values.map((value,index)=>`${index===0?'M':'L'}${x(index+1)},${y(value)}`).join(' ');
  svg+=`<path class="exact-path" d="${path(exact)}"/><path class="empirical-path" d="${path(empirical)}" opacity=".6"/>`;
  const stride=Math.max(1,Math.ceil(T/180));
  for(let t=0;t<T;t+=stride){if(view==='pmf') svg+=`<line class="ci-line" x1="${x(t+1)}" x2="${x(t+1)}" y1="${y(lower[t])}" y2="${y(upper[t])}"/>`;svg+=`<circle class="sample-dot" cx="${x(t+1)}" cy="${y(empirical[t])}" r="${T>100?2:3}"><title>t=${t+1}; exact=${exact[t]}; sampled=${empirical[t]}</title></circle>`;}
  if(T===1) svg+=`<circle cx="${x(1)}" cy="${y(exact[0])}" r="3" fill="#244c5b"/>`;
  svg+=`<text class="axis-label" x="${left+plotWidth/2}" y="${height-5}" text-anchor="middle">Step t · physical time = t × ${number(result.config.dt,5)}</text></svg>`;
  $('distribution-chart').innerHTML=svg;
  $('pair-note').textContent=`${source===target?'First return (strictly t ≥ 1).':'First arrival (strictly t ≥ 1).'} Pair FRNE: ${percent(pair.frne_percent,3)} · TV including no-hit: ${number(pair.total_variation,5)}.${T>180?' Lines use all bins; markers are thinned for readability.':''}`;
  drawOutcomes(pair);drawDataTable(i,j);
}
function drawOutcomes(pair) {
  const N=result.config.N, colors=['#507b5b','#d59961','#cbd4c2'];
  const exact=[pair.hit_exact,pair.killed_exact,pair.censored_exact],emp=[pair.hit_empirical,pair.killed_count/N,pair.censored_count/N];
  const bar=values=>`<div class="outcome-bar">${values.map((value,i)=>`<span style="width:${value*100}%;background:${colors[i]}" title="${probability(value)}"></span>`).join('')}</div>`;
  $('outcomes-caption').textContent=`Vertex ${pair.source} → ${pair.target} · cutoff T = ${result.config.T}`;
  $('outcomes').innerHTML=`<div class="outcome-label"><span>Exact</span><span>Total ${number(exact.reduce((a,b)=>a+b,0),6)}</span></div>${bar(exact)}<div class="outcome-label"><span>Monte Carlo</span><span>${number(N,0)} trials</span></div>${bar(emp)}<div class="outcome-key"><span></span><strong>Exact</strong><strong>Sampled</strong>${['Arrived by T','Killed before arrival','Alive, no arrival by T'].map((label,i)=>`<span><i class="swatch" style="background:${colors[i]}"></i>${label}</span><strong>${probability(exact[i])}</strong><strong>${probability(emp[i])}</strong>`).join('')}</div><p class="small-note">No-hit probability: ${probability(pair.no_hit_exact)} exact / ${probability(pair.no_hit_empirical)} sampled. Truncated live paths may arrive later.</p>`;
}
function drawConvergence() {
  const values=result.convergence.filter(item=>item.frne_percent!==null);
  if(!values.length){$('convergence-chart').innerHTML='<p class="empty-note">FRNE is undefined because the exact finite-bin norm is zero.</p>';return;}
  const w=430,h=155,l=40,r=12,top=15,bottom=34;
  const minN=Math.log10(values[0].N),maxN=Math.log10(values.at(-1).N),maximum=Math.max(.01,...values.flatMap(v=>[v.frne_percent,v.expected_rms_frne_percent]))*1.15;
  const x=N=>l+(maxN===minN?.5:(Math.log10(N)-minN)/(maxN-minN))*(w-l-r),y=v=>top+(h-top-bottom)*(1-v/maximum);
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="FRNE at increasing Monte Carlo sample counts"><title>Observed FRNE and theoretical expected RMS FRNE, logarithmic N axis</title>`;
  for(let k=0;k<=3;k++){const v=k*maximum/3;svg+=`<line class="grid-line" x1="${l}" x2="${w-r}" y1="${y(v)}" y2="${y(v)}"/><text class="axis-label" x="${l-8}" y="${y(v)+3}" text-anchor="end">${number(v,1)}%</text>`;}
  for(const [key,color,dash] of [['frne_percent','#507b5b',''],['expected_rms_frne_percent','#d59961','4 3']]){svg+=`<path d="${values.map((v,i)=>`${i?'L':'M'}${x(v.N)},${y(v[key])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="1.6" stroke-dasharray="${dash}"/>`;values.forEach(v=>{svg+=`<circle cx="${x(v.N)}" cy="${y(v[key])}" r="3" fill="${color}"><title>N=${v.N}: ${key}=${v[key]}%</title></circle>`;});}
  values.forEach(v=>{svg+=`<text class="axis-label" x="${x(v.N)}" y="${h-16}" text-anchor="middle">${v.N>=1e6?number(v.N/1e6,1)+'m':v.N>=1000?number(v.N/1000,1)+'k':v.N}</text>`;});
  $('convergence-chart').innerHTML=svg+'</svg><div class="legend"><span><i class="line" style="background:#507b5b"></i>Observed FRNE</span><span><i class="line" style="background:#d59961"></i>Expected RMS · N⁻¹ᐟ²</span></div><p class="small-note">N axis is logarithmic. Nested samples are correlated; individual errors need not decrease monotonically.</p>';
}
function drawDataTable(i,j) {
  $('data-table').innerHTML=`<table><thead><tr><th>Step t</th><th>Physical time</th><th>Count / N</th><th>Exact r</th><th>Sampled r′</th><th>Absolute error</th><th>95% Wilson CI</th></tr></thead><tbody>${result.times.map((t,k)=>`<tr><td>${t}</td><td>${number(result.physical_times[k],5)}</td><td>${number(result.counts[k][i][j],0)} / ${number(result.config.N,0)}</td><td>${number(result.r_exact[k][i][j],7)}</td><td>${number(result.r_empirical[k][i][j],7)}</td><td>${number(Math.abs(result.r_exact[k][i][j]-result.r_empirical[k][i][j]),7)}</td><td>[${number(result.ci95_lower[k][i][j],6)}, ${number(result.ci95_upper[k][i][j],6)}]</td></tr>`).join('')}</tbody></table>`;
}
function drawKernel() {
  const labels=[...result.graph.vertices.map(v=>v.id),'†'];
  $('kernel-table').innerHTML=`<table><thead><tr><th>From / to</th>${labels.map(label=>`<th>${label}</th>`).join('')}</tr></thead><tbody>${result.augmented_transition.map((row,i)=>`<tr><td>${labels[i]}</td>${row.map(value=>`<td>${number(value,4)}</td>`).join('')}</tr>`).join('')}</tbody></table><p class="hint">† is absorbing death. Rows are current states; columns are next states.</p>`;
}
function potentialLabel(q) {const max=Math.max(...q),min=Math.min(...q);return max===0?'q = 0':min===max?`q = ${number(max,3)}`:`Vertex rates · max ${number(max,3)}`;}
function drawHistory() {
  $('history-body').innerHTML=runs.length ? runs.map(run=>`<tr class="${run.id===selectedId?'selected':''}"><td class="history-name" title="${esc(run.config.name)}">${esc(run.config.name)}<span class="history-date">${esc(new Date(run.created_at).toLocaleString())}</span></td><td>${run.config.graph==='paper8'?'8 · ladder':'9 · grid'}</td><td>${run.config.T} / ${number(run.config.N,0)}</td><td>${potentialLabel(run.config.q)}</td><td>${number(run.config.dt,4)}</td><td>${run.metrics?percent(run.metrics.frne_percent,3):'—'}</td><td><span class="status ${run.status}">${run.status}</span></td><td><div class="history-actions"><button type="button" class="text-button" data-view="${run.id}">View</button><button type="button" class="text-button" data-clone="${run.id}">Reuse</button>${activeStatuses.includes(run.status)?`<button type="button" class="text-button" data-cancel="${run.id}">Cancel</button>`:''}</div></td></tr>`).join(''):'<tr><td colspan="8" class="empty-note">No experiments yet. Start with the paper baseline.</td></tr>';
}
function showProgress(job) {
  $('progress-panel').hidden=!job || !activeStatuses.includes(job.status);
  if(!job)return;
  $('progress-title').textContent=`${job.status==='queued'?'Queued':job.status==='cancelling'?'Cancelling':'Running'} · ${job.config.name}`;
  $('progress-text').textContent=`${percent((job.progress?.fraction||0)*100,1)} · ${number(job.progress?.completed_trials||0,0)} / ${number(job.config.N*9,0)} walks`;
  $('progress-bar').value=job.progress?.fraction||0;
  $('cancel-run').disabled=job.status==='cancelling';
  if(job.status==='failed') showError(job.error || 'Experiment failed.');
}
async function selectRun(id) {
  selectedId=id;
  if(cache.has(id)){renderResult(cache.get(id));showProgress(null);drawHistory();return;}
  const job=await api(`/api/runs/${id}`);
  showProgress(job);
  if(job.result)renderResult(job.result);
  drawHistory();
}
async function refresh() {
  if(refreshing)return;
  refreshing=true;
  try {
    runs=(await api('/api/runs')).runs;
    if(!selectedId && runs.length) selectedId=runs[0].id;
    const selected=runs.find(run=>run.id===selectedId);
    if(selected){showProgress(selected);if(selected.status==='completed' && !cache.has(selectedId))await selectRun(selectedId);}
    drawHistory();
  }finally{refreshing=false;}
}
function hydrate(cfg) {
  $('graph').value=cfg.graph;
  $('name').value=cfg.name || 'Imported experiment';$('T').value=cfg.T ?? 17;$('N').value=cfg.N ?? 10000;$('seed').value=cfg.seed ?? 2026;
  $('dt').value=cfg.dt_mode==='auto'?'auto':cfg.dt ?? 'auto';
  const q=cfg.q ?? 0;
  if(Array.isArray(q)){vertexFields(q);$('q-mode').value=q.every(v=>v===0)?'zero':q.every(v=>v===q[0])?'uniform':'vertex';$('uniform-q').value=q[0];}
  else {vertexFields();$('q-mode').value=q===0?'zero':'uniform';$('uniform-q').value=q;}
  qMode();$('experiment-form').scrollIntoView({behavior:'smooth',block:'start'});
}
function inputConfig(cfg) {
  if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) throw new Error('Import a JSON configuration object.');
  const clean = {...cfg};
  delete clean.dt_mode; delete clean.physical_horizon;
  return clean;
}
async function cancel(id) {clearError();await api(`/api/runs/${id}/cancel`,{});await refresh();}

$('experiment-form').addEventListener('submit',async event=>{
  event.preventDefault();clearError();$('run-button').disabled=true;
  try{const cfg=configuration();await api('/api/validate',cfg);const job=await api('/api/runs',cfg);selectedId=job.id;showProgress(job);await refresh();}
  catch(error){showError(error);}finally{$('run-button').disabled=false;}
});
$('graph').addEventListener('change',()=>{vertexFields();qMode();});
$('q-mode').addEventListener('change',qMode);
$('experiment-form').addEventListener('input',updatePreview);
$('paper-preset').addEventListener('click',()=>{$('T').value=17;$('N').value=1000000;$('q-mode').value='zero';$('dt').value='auto';$('name').value=`Paper baseline · ${$('graph').value==='paper8'?'eight':'nine'} vertices`;qMode();});
$('source').addEventListener('change',drawPair);$('target').addEventListener('change',drawPair);
for(const mode of ['pmf','cdf'])$(`view-${mode}`).addEventListener('click',()=>{view=mode;for(const m of ['pmf','cdf'])$(`view-${m}`).setAttribute('aria-pressed',String(m===view));drawPair();});
$('refresh-history').addEventListener('click',()=>refresh().catch(showError));
$('cancel-run').addEventListener('click',()=>cancel(selectedId).catch(showError));
$('history-body').addEventListener('click',event=>{
  const button=event.target.closest('button');if(!button)return;
  if(button.dataset.view)selectRun(button.dataset.view).catch(showError);
  if(button.dataset.cancel)cancel(button.dataset.cancel).catch(showError);
  if(button.dataset.clone){const run=runs.find(run=>run.id===button.dataset.clone);hydrate(run.config);}
});
$('save-config').addEventListener('click',async()=>{
  try{const cfg=configuration();await api('/api/validate',cfg);const url=URL.createObjectURL(new Blob([JSON.stringify(cfg,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`${cfg.graph}-config.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(error){showError(error);}
});
$('import-config').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  try{if(file.size>65536)throw new Error('Import a configuration file smaller than 64 KB.');const imported=JSON.parse(await file.text());const cfg=inputConfig(imported.config||imported);await api('/api/validate',cfg);hydrate(cfg);clearError();}catch(error){showError(error);}finally{event.target.value='';}
});
async function init(){graphs=(await api('/api/graphs')).graphs;vertexFields();qMode();await refresh();setInterval(()=>{if(runs.some(run=>activeStatuses.includes(run.status)))refresh().catch(showError);},1000);}
init().catch(showError);
