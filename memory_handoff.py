"""One-time upgrade bridge: existing results remain only in protected process RAM."""
import base64,json,os,time
from urllib.parse import urlsplit
from urllib.request import build_opener,ProxyHandler


def restore_from_environment(studio):
    url=os.environ.pop('STUDIO_HANDOFF_URL',None)
    if not url:return
    parts=urlsplit(url)
    if parts.scheme!='http' or parts.hostname!='127.0.0.1' or not parts.port or parts.query or parts.fragment:
        raise RuntimeError('Invalid local memory handoff')
    with build_opener(ProxyHandler({})).open(url,timeout=15) as response:
        snapshot=json.load(response)
    if snapshot is None:return
    entries=snapshot.get('snapshots',[snapshot])
    if not isinstance(entries,list) or len(entries)>128:
        raise RuntimeError('Invalid memory snapshot')
    now=time.monotonic()
    for entry in entries:
        owner,batch=entry['owner'],entry['batch']
        if not studio.valid_id(owner) or not studio.valid_id(batch['id']) or not 1<=len(batch['jobs'])<=studio.MAX_BATCH:
            raise RuntimeError('Invalid memory snapshot')
        ids=[]
        for item in batch['jobs']:
            if not studio.valid_id(item['id']) or item['stage'] not in ('done','error'):
                raise RuntimeError('Cannot migrate an active job')
            delivery=base64.b64decode(item.pop('delivery_b64'),validate=True) if item.get('delivery_b64') else None
            job=dict(id=item['id'],batch_id=batch['id'],owner=owner,index=item['index'],label=item['label'],
                     mode=item.get('mode',batch['spec'].get('mode','image')),
                     stage=item['stage'],step=0,total=0,gpu=item['gpu'],width=item['width'],height=item['height'],
                     created=now,started=now,finished=now,cancelled=False,result=delivery,delivery=delivery,
                     source=None,params=None,stats=item.get('stats'),error=item.get('error'))
            studio.JOBS[job['id']]=job;ids.append(job['id'])
        studio.BATCHES[batch['id']]=dict(id=batch['id'],owner=owner,jobs=ids,spec=batch['spec'],created=now)
        studio.TABS[owner]=dict(job_id=batch['id'],page_id=entry['page'],touched=now,detached=None)
