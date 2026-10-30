'use strict';
const $=id=>document.getElementById(id), opaque=()=>{const bytes=new Uint8Array(16);crypto.getRandomValues(bytes);bytes[6]=(bytes[6]&15)|64;bytes[8]=(bytes[8]&63)|128;const hex=Array.from(bytes,byte=>byte.toString(16).padStart(2,'0')).join('');return hex.slice(0,8)+'-'+hex.slice(8,12)+'-'+hex.slice(12,16)+'-'+hex.slice(16,20)+'-'+hex.slice(20);};
const validId=v=>typeof v==='string'&&/^[A-Za-z0-9_-]{24,80}$/.test(v);
const videoLabel=profile=>({fast:'H3 Turbo',base:'H3-Base',pika:'Pika H3 Flex'})[profile]||'MiniMax H3';
const key='local-generator-tab-v6', fragment=new URLSearchParams(location.hash.slice(1));
let saved={};try{saved=JSON.parse(sessionStorage.getItem(key)||'{}');}catch{}
const newTab=fragment.get('new')==='1';
let tabId=newTab?opaque():validId(fragment.get('tab'))?fragment.get('tab'):validId(saved.tab)?saved.tab:opaque();
const pageId=opaque();
let batchId=newTab?opaque():validId(fragment.get('job'))?fragment.get('job'):validId(saved.job)?saved.job:opaque();
let mode='text',sources=[],results=new Map(),cards=new Map(),selected=null,activeBatch=null;
let controller=null,busy=false,disposed=false,heartbeating=false,ticker=null,latestSeconds=0,latestAt=performance.now();
function remember(){const state={tab:tabId,job:batchId};try{sessionStorage.setItem(key,JSON.stringify(state));}catch{}history.replaceState(null,'',location.pathname+location.search+'#'+new URLSearchParams(state));$('job-id').textContent=batchId;}
remember();
function aborted(){return new DOMException('Aborted','AbortError');}
function check(signal){if(disposed||(signal&&signal.aborted))throw aborted();}
function sleep(ms,signal){return new Promise((resolve,reject)=>{check(signal);const finish=()=>{if(signal)signal.removeEventListener('abort',cancel);resolve();};const timer=setTimeout(finish,ms);const cancel=()=>{clearTimeout(timer);reject(aborted());};if(signal)signal.addEventListener('abort',cancel,{once:true});});}
function stage(title,note='Inputs and results stay in memory. Reload to resume this batch.'){
  $('loading').hidden=results.size>0;$('empty-state').hidden=true;
  $('loading').querySelector('h2').textContent=title;$('loading').querySelector('p').textContent=note;
  $('batch-note').textContent=title+' · '+note;
}
function statusText(text){$('connection-status').textContent=text;}
async function once(path,data,signal,{binary=false,timeout=15000,onBytes=null}={}){
  check(signal);const local=new AbortController(),cancel=()=>local.abort();
  if(signal)signal.addEventListener('abort',cancel,{once:true});const timer=setTimeout(cancel,timeout);
  try{
    const response=await fetch(path,{method:'POST',cache:'no-store',credentials:'omit',signal:local.signal,headers:{'Content-Type':'application/json','X-Studio-Tab':tabId,'X-Studio-Page':pageId},body:JSON.stringify(data)});
    if(!response.ok){let message='Request failed ('+response.status+')';try{message=(await response.json()).detail||message;}catch{}throw Object.assign(new Error(message),{status:response.status,transient:response.status===408||response.status>=500});}
    statusText('Connected · 2 GPUs');if(!binary)return await response.json();
    const reader=response.body.getReader(),parts=[];let received=0;
    const total=Number(response.headers.get('Content-Length')||response.headers.get('X-Result-Bytes')||0);
    while(true){const {done,value}=await reader.read();if(done)break;parts.push(value);received+=value.byteLength;if(onBytes)onBytes(received,total);}
    return new Blob(parts,{type:response.headers.get('Content-Type')?.split(';')[0]||'image/webp'});
  }catch(error){check(signal);if(local.signal.aborted||error instanceof TypeError)throw Object.assign(new Error('Connection interrupted'),{transient:true});throw error;}
  finally{clearTimeout(timer);if(signal)signal.removeEventListener('abort',cancel);}
}
async function request(path,data,signal,options={}){let attempts=0;while(true){try{return await once(path,data,signal,options);}catch(error){if(!error.transient)throw error;check(signal);attempts++;const confirmed=!!activeBatch&&activeBatch.id===batchId;statusText(confirmed?'Reconnecting · submitted':'Reconnecting · not submitted');if(busy)stage('Reconnecting',confirmed?'The server accepted batch '+batchId+'; retrying the connection.':'Submission is not confirmed. Inputs remain in this tab.');if(!confirmed&&attempts>=3)throw new Error('Connection failed before the server confirmed the batch. Reconnect and retry.');await sleep(Math.min(500*attempts,5000),signal);}}}
async function parallel(items,fn,signal,limit=4){let cursor=0;await Promise.all(Array.from({length:Math.min(limit,items.length)},async()=>{while(cursor<items.length){check(signal);const index=cursor++;await fn(items[index],index);}}));}
function clearResults(){
  $('result').hidden=true;$('result').removeAttribute('src');for(const result of results.values())URL.revokeObjectURL(result.url);
  $('result-video').pause();$('result-video').hidden=true;$('result-video').removeAttribute('src');$('result-video').load();
  results.clear();cards.clear();selected=null;$('result-list').replaceChildren();$('empty-state').hidden=false;
  $('reuse').disabled=true;$('clear').disabled=true;$('canvas-label').textContent='Output';$('result-meta').textContent='Qwen-Image-2.1';
}
function renderSources(){
  $('source-list').replaceChildren();$('source-count').textContent=sources.length?'Selected '+sources.length+' images · same prompt for each '+(mode==='video'?'video':'image'):'Choose multiple images';
  $('clear-sources').hidden=sources.length===0;$('clear-sources').disabled=busy;$('upload').disabled=busy;
  for(const [index,item]of sources.entries()){
    const card=document.createElement('div');card.className='source-card';
    const img=document.createElement('img');img.src=item.url;img.alt='Input image '+(index+1);
    const name=document.createElement('span');name.textContent=item.name;name.title=item.name;
    const remove=document.createElement('button');remove.type='button';remove.textContent='Remove';remove.disabled=busy;remove.setAttribute('aria-label','Remove input image '+(index+1));
    remove.addEventListener('click',()=>{URL.revokeObjectURL(item.url);sources=sources.filter(other=>other!==item);renderSources();});card.append(img,name,remove);$('source-list').append(card);
  }
  if(!selected)$('result-meta').textContent=mode==='video'?videoLabel($('video-profile').value):'Qwen-Image-2.1';
}
function clearSources(){for(const item of sources)URL.revokeObjectURL(item.url);sources=[];$('upload').value='';renderSources();}
function addFiles(files){
  const incoming=Array.from(files);$('form-error').textContent='';if(!incoming.length)return;
  if(sources.length+incoming.length>20){$('form-error').textContent='A batch supports up to 20 images. Remove some first.';return;}
  if(incoming.some(file=>!['image/png','image/jpeg','image/webp'].includes(file.type)||file.size>12*1024*1024)){$('form-error').textContent='Choose PNG, JPG, or WebP images under 12 MB each.';return;}
  for(const file of incoming)sources.push({blob:file,url:URL.createObjectURL(file),name:file.name||'Result image '+(sources.length+1)});
  $('upload').value='';renderSources();if(mode==='text')setMode('image');
}
function setMode(next){mode=['text','image','video'].includes(next)?next:'image';for(const name of ['text','image','video']){$(name+'-tab').classList.toggle('selected',name===mode);$(name+'-tab').setAttribute('aria-selected',String(name===mode));}$('upload-section').hidden=mode==='text';$('video-settings').hidden=mode!=='video';$('image-size-settings').hidden=mode==='video';$('image-advanced-settings').hidden=mode==='video';$('upload').multiple=true;$('upload-caption').textContent=mode==='video'?'First-frame images':'Reference images';$('upload-action').textContent='Choose or drop images';$('upload-limit').textContent='Up to 20 images, 12 MB each';$('source-hint').textContent=mode==='video'?'Each image becomes the first frame of a separate video. The same prompt and settings are applied to all videos.':'The same prompt is applied to each image.';$('prompt-caption').textContent=mode==='video'?'Video prompt':'Prompt';$('prompt').placeholder=mode==='video'?'Example: The camera moves closer as the person smiles and turns. Soft ambient sound.':'Example: A cabin by the sea after rain, warm lights, cinematic detail.';renderSources();}
$('video-tab').addEventListener('click',()=>setMode('video'));
$('text-tab').addEventListener('click',()=>setMode('text'));$('image-tab').addEventListener('click',()=>setMode('image'));
$('video-profile').addEventListener('change',renderSources);
$('upload').addEventListener('change',event=>addFiles(event.target.files));$('clear-sources').addEventListener('click',clearSources);
$('drop-zone').addEventListener('dragover',event=>{event.preventDefault();$('drop-zone').classList.add('drag');});
$('drop-zone').addEventListener('dragleave',()=>$('drop-zone').classList.remove('drag'));
$('drop-zone').addEventListener('drop',event=>{event.preventDefault();$('drop-zone').classList.remove('drag');if(!busy)addFiles(event.dataTransfer.files);});
function addResultAsSource(id){const result=results.get(id);if(result&&result.job.mode!=='video'){addFiles([result.blob]);$('prompt').focus();}}
$('reuse').addEventListener('click',()=>addResultAsSource(selected));
function chooseResult(id){
  const result=results.get(id);if(!result)return;selected=id;
  const video=result.job.mode==='video';$('result').hidden=video;$('result-video').hidden=!video;if(video){$('result').removeAttribute('src');$('result-video').src=result.url;}else{$('result-video').pause();$('result-video').removeAttribute('src');$('result').src=result.url;}$('empty-state').hidden=true;$('loading').hidden=true;
  $('reuse').disabled=busy||video;$('clear').disabled=false;$('canvas-label').textContent=result.job.label;
  const j=result.job;$('result-meta').textContent=(video?(j.stats.model||'MiniMax H3')+' · '+(j.stats.duration||activeBatch?.spec.duration||5)+' s video · ':'Qwen-Image-2.1 · ')+(j.stats.width||j.width)+' × '+(j.stats.height||j.height)+' · GPU '+j.gpu+' · generated in '+j.stats.seconds+' s · received in '+result.seconds.toFixed(2)+' s · seed '+j.stats.seed;
  for(const [key,card]of cards)card.element.classList.toggle('selected-result',key===id);
}
function setBusy(value){busy=value;$('generate').disabled=value;$('generate-label').textContent=value?'Processing batch':'Add to queue';$('reuse').disabled=value||!selected||results.get(selected)?.job.mode==='video';renderSources();clearInterval(ticker);if(value)ticker=setInterval(()=>{$('elapsed').value=Math.floor(latestSeconds+(performance.now()-latestAt)/1000)+' s';},1000);}
const stages={queued:'Queued',loading:'Loading model',preparing:'Preparing image',encoding:'Encoding input',sampling:'Generating',decoding:'Decoding result',packing:'Encoding result',stopping:'Stopping',done:'Done',error:'Failed'};
let refreshingGlobal=false;
function renderGlobalJobs(jobs){
  $('all-jobs-count').textContent=String(jobs.length);
  const list=$('all-jobs-list');list.replaceChildren();
  if(!jobs.length){const empty=document.createElement('p');empty.className='all-jobs-empty';empty.textContent='No queued or running jobs.';list.append(empty);return;}
  for(const job of jobs){
    const row=document.createElement('div');row.className='all-job';
    const detail=document.createElement('div');detail.className='all-job-detail';
    const title=document.createElement('strong');title.textContent=job.label;
    const status=document.createElement('span');status.textContent=(job.gpu===null?'Queued · ahead: '+job.ahead+' jobs':'GPU '+job.gpu+' · '+(stages[job.stage]||job.stage))+(job.stage==='sampling'&&job.total?' · '+job.step+'/'+job.total+' steps':'')+' · '+Math.floor(job.seconds)+' s';
    const id=document.createElement('small');id.textContent='Job '+job.id+' · batch '+job.batch_id;
    detail.append(title,status,id);
    const stop=document.createElement('button');stop.type='button';stop.className='stop-job';stop.textContent=job.stage==='stopping'?'Stopping':'Stop';stop.disabled=job.stage==='stopping';stop.setAttribute('aria-label','Stop job '+job.label);
    stop.addEventListener('click',async()=>{stop.disabled=true;stop.textContent='Stopping';$('all-jobs-error').textContent='';try{await once('/api/jobs/stop',{id:job.id},null,{timeout:15000});await refreshGlobalJobs();}catch(error){$('all-jobs-error').textContent=error.message;stop.disabled=false;stop.textContent='Stop';}});
    row.append(detail,stop);list.append(row);
  }
}
async function refreshGlobalJobs(){
  if(disposed||refreshingGlobal)return;
  refreshingGlobal=true;
  try{const data=await once('/api/jobs/active',{},null,{timeout:6000});renderGlobalJobs(data.jobs);$('all-jobs-error').textContent='';}
  catch(error){if(!$('all-jobs-panel').hidden)$('all-jobs-error').textContent='Job list is temporarily unavailable: '+error.message;}
  finally{refreshingGlobal=false;}
}
$('all-jobs-toggle').addEventListener('click',()=>{const panel=$('all-jobs-panel');panel.hidden=!panel.hidden;$('all-jobs-toggle').setAttribute('aria-expanded',String(!panel.hidden));if(!panel.hidden)refreshGlobalJobs();});
$('all-jobs-close').addEventListener('click',()=>{$('all-jobs-panel').hidden=true;$('all-jobs-toggle').setAttribute('aria-expanded','false');});
setInterval(refreshGlobalJobs,3000);
function renderBatch(batch){
  $('batch-progress').hidden=false;$('batch-summary').textContent='Completed '+batch.completed+' / '+batch.total+' · running '+batch.running+' · queued '+batch.queued+(batch.failed?' · failed '+batch.failed:'');
  $('batch-note').textContent=batch.stage==='done'?'Batch complete. Select a result below.':batch.spec.mode==='video'?videoLabel(batch.spec.video_profile)+' · '+(batch.spec.duration||5)+' s each · '+batch.total+' video'+(batch.total===1?'':'s')+' · two GPUs take jobs from the queue. Reload to resume this batch.':'Two GPUs take jobs from the same queue. Reload to resume this batch.';
  for(const job of batch.jobs){
    let card=cards.get(job.id);
    if(!card){
      const element=document.createElement('article');element.className='result-card';
      const select=document.createElement('button');select.type='button';select.className='result-select';select.disabled=true;
      const image=document.createElement('img');image.hidden=true;image.alt='Result '+(job.index+1)+' · '+job.label;
      const video=document.createElement('video');video.hidden=true;video.muted=true;video.playsInline=true;video.preload='metadata';
      const placeholder=document.createElement('span');placeholder.className='result-placeholder';placeholder.textContent=String(job.index+1).padStart(2,'0');
      select.append(image,video,placeholder);select.addEventListener('click',()=>chooseResult(job.id));
      const name=document.createElement('strong');name.textContent=(job.index+1)+'. '+job.label;name.title=job.label;
      const status=document.createElement('small');status.className='result-status';
      const reuse=document.createElement('button');reuse.type='button';reuse.className='quiet';reuse.textContent='Use as input';reuse.disabled=true;reuse.addEventListener('click',()=>addResultAsSource(job.id));
      element.append(select,name,status,reuse);$('result-list').append(element);card={element,select,image,video,placeholder,status,reuse};cards.set(job.id,card);
    }
    card.status.textContent=(job.gpu!==null?'GPU '+job.gpu+' · ':'')+(job.stage==='sampling'&&job.total?stages[job.stage]+' · '+job.step+' / '+job.total+' steps':job.stage==='queued'?'Ahead: '+job.ahead+' jobs':job.stage==='error'?job.error:stages[job.stage]||job.stage);
    card.element.classList.toggle('result-error',job.stage==='error');
    const result=results.get(job.id);card.reuse.disabled=!result||busy||job.mode==='video';
    if(result){if(job.mode==='video'){card.video.src=result.url;card.video.hidden=false;}else{card.image.src=result.url;card.image.hidden=false;}card.placeholder.hidden=true;card.select.disabled=false;}
  }
}
async function uploadReference(item,signal,index,totalFiles){
  const image=new Image();image.src=item.url;await image.decode();check(signal);
  const scale=Math.min(1,1536/Math.max(image.naturalWidth,image.naturalHeight)),canvas=document.createElement('canvas');
  canvas.width=Math.round(image.naturalWidth*scale);canvas.height=Math.round(image.naturalHeight*scale);
  const context=canvas.getContext('2d',{alpha:false});context.fillStyle='#fff';context.fillRect(0,0,canvas.width,canvas.height);context.drawImage(image,0,0,canvas.width,canvas.height);
  let encoded=canvas.toDataURL('image/jpeg',0.95).split(',')[1];context.clearRect(0,0,canvas.width,canvas.height);canvas.width=canvas.height=0;image.removeAttribute('src');
  const total=encoded.length*3/4-(encoded.endsWith('==')?2:encoded.endsWith('=')?1:0),id=opaque();
  await request('/api/upload/start',{id,total},signal);
  const chunks=[];for(let offset=0;offset<encoded.length;offset+=65536)chunks.push(encoded.slice(offset,offset+65536));encoded='';
  let completed=0;await parallel(chunks,async(chunk,n)=>{await request('/api/upload/chunk',{id,index:n,chunk},signal);completed++;stage('Uploading input '+(index+1)+' / '+totalFiles+' · '+Math.round(completed/chunks.length*100)+'%');},signal);
  chunks.fill('');return id;
}
async function receiveResult(job,signal){
  const started=performance.now();let blob;
  stage('Receiving '+(job.mode==='video'?'video':'image')+' '+(job.index+1),'Transfer resumes in chunks if the connection drops.');
  try{blob=await once('/api/jobs/result',{id:job.id},signal,{binary:true,timeout:job.mode==='video'?120000:12000,onBytes:(n,total)=>{if(cards.has(job.id))cards.get(job.id).status.textContent='Receiving '+(total?Math.round(n/total*100)+'%':Math.round(n/1024)+' KB');}});}
  catch(error){
    if(!error.transient)throw error;const offsets=[];for(let n=0;n<job.result_bytes;n+=49152)offsets.push(n);const chunks=new Array(offsets.length);let received=0;
    await parallel(offsets,async(offset,index)=>{const part=await request('/api/jobs/result-chunk',{id:job.id,offset},signal);chunks[index]=Uint8Array.from(atob(part.chunk),c=>c.charCodeAt(0));received+=chunks[index].length;if(cards.has(job.id))cards.get(job.id).status.textContent='Resuming '+Math.round(received/job.result_bytes*100)+'%';},signal);
    blob=new Blob(chunks,{type:job.mode==='video'?'video/mp4':'image/webp'});chunks.length=0;
  }
  check(signal);const result={blob,url:URL.createObjectURL(blob),job,seconds:(performance.now()-started)/1000};results.set(job.id,result);
  if(!selected)chooseResult(job.id);
}
async function follow(batch,signal){
  activeBatch=batch;batchId=batch.id;remember();const spec=batch.spec;$('video-profile').value=spec.video_profile||'fast';setMode(spec.mode);$('prompt').value=spec.prompt;$('width').value=String(spec.width);$('height').value=String(spec.height);$('seed').value=String(spec.seed);$('steps').value=String(spec.steps);$('duration').value=String(spec.duration||5);
  while(true){
    check(signal);activeBatch=batch;latestSeconds=batch.seconds;latestAt=performance.now();$('elapsed').value=Math.floor(batch.seconds)+' s';renderBatch(batch);
    const ready=batch.jobs.filter(job=>job.stage==='done'&&!results.has(job.id));await parallel(ready,job=>receiveResult(job,signal),signal,2);renderBatch(batch);
    if(batch.stage==='done'){if(batch.failed)$('form-error').textContent=batch.failed+' item(s) failed. Other results remain available.';return;}
    if(!results.size){const running=batch.jobs.filter(job=>job.gpu!==null&&job.stage!=='done'&&job.stage!=='error');stage(running.length?(batch.spec.mode==='video'?'Generating videos with '+videoLabel(batch.spec.video_profile):'Processing images'):'Queued',running.length?running.map(job=>'GPU '+job.gpu+' · item '+(job.index+1)+' · '+(job.stage==='sampling'?job.step+'/'+job.total+' steps':stages[job.stage])).join('; '):'Batch: '+batch.total+(batch.spec.mode==='video'?' videos':' images')+', waiting for an available GPU.');}
    await sleep(1000,signal);batch=await request('/api/batches/status',{id:batch.id},signal);
  }
}
async function manage(action){
  if(controller)controller.abort();const current=new AbortController();controller=current;setBusy(true);$('form-error').textContent='';
  try{await action(current.signal);}catch(error){current.abort();if(error.name!=='AbortError')$('form-error').textContent=error.message;}
  finally{if(controller===current){controller=null;setBusy(false);$('loading').hidden=true;$('empty-state').hidden=results.size>0;if(activeBatch)renderBatch(activeBatch);}}
}
async function resume(){await manage(async signal=>{stage('Restoring this tab');const state=await request('/api/tab/state',{},signal);if(state.batch)await follow(state.batch,signal);else statusText('Connected · 2 GPUs');});}
$('generate-form').addEventListener('submit',event=>{
  event.preventDefault();if(busy)return;if(mode!=='text'&&!sources.length){$('form-error').textContent='Select an input image first.';return;}
  const picked=sources.slice(),spec={id:activeBatch?opaque():batchId,mode,prompt:$('prompt').value,width:Number($('width').value),height:Number($('height').value),seed:Number($('seed').value),steps:Number($('steps').value),items:[]};
  if(mode==='video'){spec.duration=Number($('duration').value);spec.video_profile=$('video-profile').value;}
  batchId=spec.id;activeBatch=null;remember();clearResults();$('batch-progress').hidden=true;latestSeconds=0;latestAt=performance.now();
  manage(async signal=>{
    stage('Preparing batch');
    if(spec.mode!=='text'){
      spec.items=picked.map(item=>({id:opaque(),label:item.name.slice(0,160)}));
      await parallel(picked,async(item,index)=>{spec.items[index].upload_id=await uploadReference(item,signal,index,picked.length);},signal,2);
    }else spec.items=[{id:opaque(),label:'Generated image'}];
    const batch=await request('/api/batches/start',spec,signal);spec.prompt='';activeBatch=batch;await follow(batch,signal);
  });
});
async function clearPage(){await manage(async signal=>{clearResults();clearSources();$('prompt').value='';$('seed').value='-1';stage('Clearing this tab');await request('/api/tab/clear',{},signal);activeBatch=null;batchId=opaque();remember();$('batch-progress').hidden=true;latestSeconds=0;$('elapsed').value='0 s';});}
$('clear-page').addEventListener('click',clearPage);$('clear').addEventListener('click',clearPage);$('reconnect').addEventListener('click',resume);document.addEventListener('keydown',event=>{if(event.key==='Escape')clearPage();});
let tabChannel=null,identityReady=false;
function rotateTabIdentity(){
  if(controller)controller.abort();
  clearResults();clearSources();activeBatch=null;$('batch-progress').hidden=true;
  tabId=opaque();batchId=opaque();remember();
  if(identityReady)resume();
}
function claimTabIdentity(onReady){
  identityReady=false;
  if(typeof BroadcastChannel!=='function'){identityReady=true;onReady();return;}
  if(tabChannel)tabChannel.close();
  tabChannel=new BroadcastChannel('local-generator-tabs-v1');
  tabChannel.onmessage=event=>{
    const message=event.data;
    if(!message||message.tab!==tabId||message.page===pageId)return;
    if(message.type==='probe')tabChannel.postMessage({type:'occupied',tab:tabId,target:message.page,page:pageId});
    else if(message.type==='occupied'&&message.target===pageId){
      rotateTabIdentity();
      tabChannel.postMessage({type:'probe',tab:tabId,page:pageId});
    }
  };
  tabChannel.postMessage({type:'probe',tab:tabId,page:pageId});
  setTimeout(()=>{if(!disposed){identityReady=true;onReady();}},350);
}
window.addEventListener('pagehide',()=>{disposed=true;if(tabChannel){tabChannel.close();tabChannel=null;}if(controller)controller.abort();clearInterval(ticker);navigator.sendBeacon('/api/tab/detach',new Blob([JSON.stringify({tab_id:tabId,page_id:pageId})],{type:'application/json'}));for(const result of results.values())URL.revokeObjectURL(result.url);for(const item of sources)URL.revokeObjectURL(item.url);});
window.addEventListener('pageshow',event=>{if(event.persisted){disposed=false;clearResults();claimTabIdentity(resume);}});
setInterval(async()=>{if(disposed||heartbeating)return;heartbeating=true;try{await once('/api/tab/state',{},null,{timeout:5000});}catch{statusText(activeBatch?'Disconnected · submitted':'Disconnected · not submitted');}finally{heartbeating=false;}},20000);
renderSources();claimTabIdentity(resume);
