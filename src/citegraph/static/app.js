const $ = id => document.getElementById(id);
let overview, selected, neighborhood, positions = [], busy = false, exportData;
const colors = ['#0d977b','#ca9540','#658ab8'];
const percent = value => `${(value*100).toFixed(1)}%`;

async function api(path, options={}) {
 const headers = {'Content-Type':'application/json',...options.headers};
 if ($('token').value) headers.Authorization = `Bearer ${$('token').value}`;
 const response = await fetch(path,{...options,headers});
 const body = await response.json();
 if(!response.ok) throw Error(typeof body.detail === 'string' ? body.detail : 'Invalid request');
 return body;
}
function message(text=''){ $('message').textContent=text; }
function setBusy(value){ busy=value; $('stress').disabled=value||!selected;
 document.querySelectorAll('.example').forEach(button=>button.disabled=value); }
function renderPrediction(name,data){
 $(name+'-topic').textContent=data.topic;
 $(name+'-confidence').textContent=percent(data.confidence)+' confidence';
 $(name+'-fill').style.width=percent(data.confidence);
 $(name+'-decision').textContent=data.decision;
 $(name+'-decision').className='decision '+data.decision;
}
async function inspect(id){
 if(busy)return;
 setBusy(true);message();
 try{
  const [node,graph]=await Promise.all([api('/api/node/'+encodeURIComponent(id)),api('/api/neighborhood/'+encodeURIComponent(id))]);
  selected=node;neighborhood=graph;
  $('paper-id').value=id;$('selected-id').textContent=id;
  $('node-meta').textContent=`${node.degree} citation connections · ${node.partition} partition`;
  $('truth').textContent=node.observed_topic||'Label withheld';
  renderPrediction('gcn',node.gcn);renderPrediction('mlp',node.mlp);
  $('terms').replaceChildren();
  for(const item of node.terms){const span=document.createElement('span');span.className='term';span.textContent=item.term;span.title=`Input weight: ${item.weight.toFixed(4)}`;$('terms').append(span);}
  $('network-size').textContent=`${graph.nodes.length} nodes · ${graph.edges.length} edges`;
  $('stress-result').textContent='Run a perturbation to compare the frozen model with fewer citation edges.';
  document.querySelectorAll('.example').forEach(button=>button.classList.toggle('selected',button.dataset.id===id));
  exportData={node,neighborhood:graph};$('export').disabled=false;draw();
 }catch(error){message(error.message)}finally{setBusy(false)}
}
function draw(){
 if(!neighborhood)return;
 const canvas=$('network'),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
 canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);
 const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
 const w=rect.width,h=rect.height,cx=w*.5,cy=h*.43;
 const first=neighborhood.nodes.filter(n=>n.hop===1),second=neighborhood.nodes.filter(n=>n.hop===2);
 const place=(items,radius,offset)=>items.map((node,i)=>({...node,x:cx+Math.cos(offset+i/items.length*Math.PI*2)*radius,y:cy+Math.sin(offset+i/items.length*Math.PI*2)*radius*.84}));
 positions=[{...neighborhood.nodes[0],x:cx,y:cy},...place(first,Math.min(w*.25,h*.23),-.6),...place(second,Math.min(w*.42,h*.38),.1)];
 const map=new Map(positions.map(n=>[n.id,n]));
 for(const [left,right] of neighborhood.edges){const a=map.get(left),b=map.get(right);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=a.hop===0||b.hop===0?'#a2c8b9':'#d7e4da';ctx.lineWidth=a.hop===0||b.hop===0?1.3:.7;ctx.stroke();}
 for(const node of positions){const r=node.hop===0?14:node.hop===1?7:4;ctx.beginPath();ctx.arc(node.x,node.y,r+4,0,2*Math.PI);ctx.fillStyle=node.hop===0?'#d8eee4':'#f3f7f2';ctx.fill();ctx.beginPath();ctx.arc(node.x,node.y,r,0,2*Math.PI);ctx.fillStyle=colors[node.topic_id];ctx.fill();if(node.hop===0){ctx.fillStyle='white';ctx.font='bold 11px system-ui';ctx.textAlign='center';ctx.fillText('●',node.x,node.y+4);ctx.fillStyle='#365f51';ctx.font='9px monospace';ctx.fillText(node.id,node.x,node.y+31)}}
}
$('network').addEventListener('click',event=>{if(busy)return;const rect=$('network').getBoundingClientRect();const x=event.clientX-rect.left,y=event.clientY-rect.top;const node=positions.find(n=>Math.hypot(n.x-x,n.y-y)<12);if(node&&node.id!==selected?.id)inspect(node.id);});
window.addEventListener('resize',draw);
$('lookup').addEventListener('submit',event=>{event.preventDefault();const id=$('paper-id').value.trim();if(/^\d{1,20}$/.test(id))inspect(id);else message('Enter a numeric paper ID from this benchmark.');});
$('drop').addEventListener('input',()=>$('drop-value').textContent=$('drop').value+'%');
$('stress').addEventListener('click',async()=>{
 if(!selected||busy)return;setBusy(true);message();$('stress-result').textContent='Running the frozen graph network…';
 try{const result=await api('/api/stress',{method:'POST',body:JSON.stringify({paper_id:selected.id,drop_rate:Number($('drop').value)/100,seed:2026})});
 $('stress-result').textContent=`${result.baseline.topic} (${percent(result.baseline.confidence)}) → ${result.perturbed.topic} (${percent(result.perturbed.confidence)}). ${result.perturbed.decision.toUpperCase()} · ${result.retained_undirected_edges.toLocaleString()} edges retained · ${Math.round(result.latency_ms)} ms.`;
 exportData.stress=result;
 }catch(error){message(error.message);$('stress-result').textContent='Perturbation failed; original prediction is unchanged.'}finally{setBusy(false)}
});
$('export').addEventListener('click',()=>{if(!exportData)return;const url=URL.createObjectURL(new Blob([JSON.stringify(exportData,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`citegraph-${selected.id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
async function initialize(){
 try{
 overview=await api('/api/overview');
 $('node-count').textContent=overview.nodes.toLocaleString();$('edge-count').textContent=overview.edges.toLocaleString();
 $('gcn-score').textContent=percent(overview.summary.gcn.accuracy.mean);$('mlp-score').textContent=percent(overview.summary.mlp.accuracy.mean);
 $('examples').replaceChildren();
 for(const example of overview.examples.slice(0,6)){const button=document.createElement('button');button.className='example';button.dataset.id=example.id;const strong=document.createElement('strong'),small=document.createElement('small');strong.textContent=example.id;small.textContent=example.category;button.append(strong,small);button.addEventListener('click',()=>inspect(example.id));$('examples').append(button);}
 for(const name of ['gcn','mlp']){const row=document.createElement('tr');for(const text of [name==='gcn'?'Graph + words':'Words only',percent(overview.summary[name].accuracy.mean),overview.summary[name].macro_f1.mean.toFixed(3)]){const cell=document.createElement('td');cell.textContent=text;row.append(cell)}$('benchmark-rows').append(row)}
 const majority=document.createElement('tr');for(const text of ['Training-majority',percent(overview.majority.accuracy),overview.majority.macro_f1.toFixed(3)]){const cell=document.createElement('td');cell.textContent=text;majority.append(cell)}$('benchmark-rows').append(majority);
 const initial=overview.examples[0];if(initial)await inspect(initial.id);
 }catch(error){message(error.message+' If token access is enabled, enter the token and reload the workspace.');}
}
$('token').addEventListener('change',()=>{if(!overview){$('benchmark-rows').replaceChildren();initialize();}});
initialize();
