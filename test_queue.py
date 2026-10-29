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
        def __init__(self,gpu):self.gpu=gpu;self.cancelled=threading.Event()
        def start(self):pass
        def cancel(self):self.cancelled.set()
        def stop(self):self.cancel()
        def run(self,params,source,progress):
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
            a,b,c=ident(),ident(),ident();pa,pb,pc=ident(),ident(),ident();ha,hb,hc=headers(a,pa),headers(b,pb),headers(c,pc)
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
            assert len((await client.get('/api/health')).json()['gpus'])==2
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
            payload=(await post('/api/jobs/result',{'id':job['id']},ha)).content
            with Image.open(io.BytesIO(payload)) as image:assert image.convert('RGB').tobytes()==pixels
            assert len(payload)>studio.CHUNK_SIZE*2
            parts=[]
            for offset in range(0,len(payload),studio.CHUNK_SIZE):
                part=(await post('/api/jobs/result-chunk',{'id':job['id'],'offset':offset},ha)).json();parts.append(base64.b64decode(part['chunk']))
            assert b''.join(parts)==payload
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
            await post('/api/tab/clear',{},hb)
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
            for h in (ha,hb,hc):await post('/api/tab/clear',{},h)
            assert not studio.JOBS and not studio.BATCHES and not studio.UPLOADS
            print('PASS: two independent GPU slots; FIFO batches; same prompt and separate inputs; idempotency; refresh of all child jobs/results; tab isolation; atomic batch validation; parallel upload; lossless binary/chunk transfer; cancellation on both GPUs; failure isolation; memory cleanup; origin protection')

asyncio.run(main())
