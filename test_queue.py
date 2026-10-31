"""RAM-only regression tests: dual GPU scheduling, atomic batches and refresh recovery."""
import asyncio
import base64
import importlib
import io
import os
import random
import threading
import time
import uuid
import zipfile

import httpx
from PIL import Image
studio = importlib.import_module(os.environ.get('STUDIO_TEST_MODULE', 'app'))


def ident():
    return str(uuid.uuid4())


async def main():
    pixels = random.Random(42).randbytes(512*512*3)
    with io.BytesIO() as buffer:
        Image.frombytes('RGB',(512,512),pixels).save(buffer,format='PNG')
        png=buffer.getvalue()
    gate=threading.Event();guard=threading.Lock();calls=[];active=[0];peak=[0]
    class FakeRunner:
        def __init__(self,gpu):self.gpu=gpu;self.cancelled=threading.Event();self.started=False
        def start(self):pass
        def cancel(self):self.cancelled.set()
        def stop(self):self.cancel()
        def resident(self):return self.started
        def stop_if_idle(self,idle_seconds,now=None):return False
        def run(self,params,source,progress):
            self.started=True
            self.cancelled.clear()
            with guard:
                active[0]+=1;peak[0]=max(peak[0],active[0]);calls.append((self.gpu,params[0],source))
            try:
                progress('sampling',1,2)
                if params[0]=='hold':
                    while not gate.wait(.01):
                        if self.cancelled.is_set():raise RuntimeError('Cancelled')
                        progress('sampling',1,2)
                if source==b'bad':raise ValueError('Test failure')
                progress('sampling',2,2)
                return png,dict(seed=42,seconds=.01,total_seconds=.01,gpu=self.gpu,peak_vram_gib=0)
            finally:
                with guard:active[0]-=1
    studio.RUNNER_FACTORY=FakeRunner
    async with studio.app.router.lifespan_context(studio.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio.app),base_url='http://localhost:7860') as client:
            def headers(tab,page):return {'Origin':'http://localhost:64251','X-Studio-Tab':tab,'X-Studio-Page':page}
            a,b,c,d=ident(),ident(),ident(),ident();pa,pb,pc,pd=ident(),ident(),ident(),ident();ha,hb,hc,hd=headers(a,pa),headers(b,pb),headers(c,pc),headers(d,pd)
            async def post(path,data,h=ha,code=200):
                response=await client.post(path,json=data,headers=h)
                assert response.status_code==code,(path,response.status_code,response.text)
                return response
            async def upload(raw,h):
                uid=ident();await post('/api/upload/start',{'id':uid,'total':len(raw)},h)
                chunks=[raw[n:n+studio.CHUNK_SIZE] for n in range(0,len(raw),studio.CHUNK_SIZE)]
                for n in reversed(range(len(chunks))):
                    data={'id':uid,'index':n,'chunk':base64.b64encode(chunks[n]).decode()}
                    await post('/api/upload/chunk',data,h);await post('/api/upload/chunk',data,h)
                return {'id':ident(),'label':'reference','upload_id':uid}
            async def batch_state(bid,h):return (await post('/api/batches/status',{'id':bid},h)).json()
            async def wait_for(bid,h,predicate):
                for _ in range(500):
                    state=await batch_state(bid,h)
                    if predicate(state):return state
                    await asyncio.sleep(.01)
                raise AssertionError('batch did not reach expected state')
            spec=dict(mode='image',width=512,height=512,steps=2,seed=42)
            index=await client.get('/');assert 'type="password"' not in index.text and 'no-store' in index.headers['cache-control']
            health=(await client.get('/api/health')).json()
            assert len(health['gpus'])==2 and health['idle_unload_seconds']>=30
            refs=[b'a'*100000,b'b'*100000,b'c'*100000]
            items=[await upload(raw,ha) for raw in refs]
            bid=ident();request=dict(spec,id=bid,prompt='hold',items=items)
            await post('/api/batches/start',request,ha)
            running=await wait_for(bid,ha,lambda s:s['running']==2 and s['queued']==1)
            assert {j['gpu'] for j in running['jobs'] if j['gpu'] is not None}=={0,1}
            for _ in range(500):
                if peak[0]==2:break
                await asyncio.sleep(.01)
            assert peak[0]==2
            await post('/api/batches/start',request,ha)
            assert len(studio.JOBS)==3 and studio.QUEUE.qsize()==1
            await post('/api/batches/status',{'id':bid},hb,404)
            await post('/api/jobs/status',{'id':items[0]['id']},hb,404)
            # Refresh keeps the root batch and all child IDs; late close beacons cannot remove it.
            pa=ident();ha=headers(a,pa)
            restored=(await post('/api/tab/state',{},ha)).json()['batch']
            assert restored['id']==bid and [j['id'] for j in restored['jobs']]==[item['id'] for item in items]
            await post('/api/tab/detach',{'tab_id':a,'page_id':pa},ha)
            studio.expire(time.monotonic()+studio.CLOSE_GRACE+1)
            assert a in studio.TABS
            gate.set();finished=await wait_for(bid,ha,lambda s:s['stage']=='done')
            assert finished['completed']==3 and finished['failed']==0
            assert sorted(source for _,_,source in calls)==sorted(refs)
            assert all(prompt=='hold' for _,prompt,_ in calls)
            # All results survive refresh; binary and multiple chunks reproduce exact pixels.
            assert (await post('/api/tab/state',{},ha)).json()['batch']['completed']==3
            job=finished['jobs'][0]
            assert job['retryable'] and job['prompt']=='hold'
            payload=(await post('/api/jobs/result',{'id':job['id']},ha)).content
            with Image.open(io.BytesIO(payload)) as image:assert image.convert('RGB').tobytes()==pixels
            assert len(payload)>studio.CHUNK_SIZE*2
            parts=[]
            for offset in range(0,len(payload),studio.CHUNK_SIZE):
                part=(await post('/api/jobs/result-chunk',{'id':job['id'],'offset':offset},ha)).json();parts.append(base64.b64decode(part['chunk']))
            assert b''.join(parts)==payload
            # A single editing input can be retried in the same batch with its own prompt.
            assert (await post('/api/jobs/source',{'id':job['id']},ha)).content==refs[0]
            retry_id=ident();retry=(await post('/api/jobs/retry',{'id':retry_id,'source_id':job['id'],'prompt':'special retry'},ha)).json()
            assert retry['total']==4 and retry['jobs'][-1]['prompt']=='special retry'
            retried=await wait_for(bid,ha,lambda s:s['stage']=='done' and s['total']==4)
            assert any(prompt=='special retry' and source==refs[0] for _,prompt,source in calls)
            # A retry may replace its source with a browser-cropped upload.
            cropped=await upload(b'cropped-source',ha);cropped_id=ident()
            cropped_batch=(await post('/api/jobs/retry',{'id':cropped_id,'source_id':job['id'],'upload_id':cropped['upload_id']},ha)).json()
            assert cropped_batch['total']==5 and cropped['upload_id'] not in studio.UPLOADS
            retried=await wait_for(bid,ha,lambda s:s['stage']=='done' and s['total']==5)
            assert any(source==b'cropped-source' for _,_,source in calls)
            assert (await post('/api/jobs/source',{'id':cropped_id},ha)).content==b'cropped-source'
            # The complete batch is packaged in RAM as a standard ZIP archive.
            archive=(await post('/api/batches/archive',{'id':bid},ha)).content
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                assert len(bundle.namelist())==5 and all(name.endswith('.webp') for name in bundle.namelist())
                assert all(bundle.read(name)==payload for name in bundle.namelist())
            # The former 20-image cap is gone.
            many=[await upload(bytes([n]),hd) for n in range(21)];many_id=ident()
            await post('/api/batches/start',dict(spec,id=many_id,prompt='many',items=many),hd)
            many_done=await wait_for(many_id,hd,lambda s:s['stage']=='done')
            assert many_done['total']==21 and many_done['completed']==21
            # Atomic validation: one missing image must not consume any of the valid uploads.
            valid=await upload(b'd',ha);bad=ident();count=len(studio.JOBS)
            await post('/api/batches/start',dict(spec,id=bad,prompt='edit',items=[valid,{'id':ident(),'upload_id':ident()}]),ha,400)
            assert len(studio.JOBS)==count and valid['upload_id'] in studio.UPLOADS
            assert (await post('/api/tab/state',{},ha)).json()['batch']['id']==bid
            # Cancel two running GPUs and verify a queued page runs afterwards.
            gate.clear();bi=ident();block=[await upload(b'x',hb),await upload(b'y',hb)]
            await post('/api/batches/start',dict(spec,id=bi,prompt='hold',items=block),hb)
            await wait_for(bi,hb,lambda s:s['running']==2)
            ci=ident();next_item=await upload(b'z',hc)
            await post('/api/batches/start',dict(spec,id=ci,prompt='next',items=[next_item]),hc)
            assert (await batch_state(ci,hc))['queued']==1
            cleared=(await post('/api/tab/clear',{},hb)).json();assert cleared['cancelled']==2
            await wait_for(ci,hc,lambda s:s['stage']=='done')
            await post('/api/batches/status',{'id':bi},hb,404)
            assert peak[0]==2
            # One failed image does not abort the batch or the worker.
            failures=[await upload(b'bad',hb),await upload(b'good',hb)];fi=ident()
            await post('/api/batches/start',dict(spec,id=fi,prompt='mixed',items=failures),hb)
            partial=await wait_for(fi,hb,lambda s:s['stage']=='done')
            assert partial['completed']==1 and partial['failed']==1
            foreign=headers(a,ident());foreign['Origin']='https://example.com'
            await post('/api/tab/clear',{},foreign,403)
            for h in (ha,hb,hc,hd):await post('/api/tab/clear',{},h)
            assert not studio.JOBS and not studio.BATCHES and not studio.UPLOADS
            print('PASS: dual GPU FIFO; unlimited image count; same-prompt batch; per-image retry and prompt override; cropped retry source; RAM ZIP; refresh; isolation; cancellation; failure cleanup; origin protection')

asyncio.run(main())
