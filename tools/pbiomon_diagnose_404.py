"""Manual bounded private/public URL diagnostics; never used as a public download fallback."""
import collections
import concurrent.futures
import datetime
import hashlib
import http.client
import ipaddress
import json
from pathlib import Path
import random
import socket
import sqlite3
import ssl
import time
import urllib.parse
ROOT = Path('/mnt/data2/pbiomon_yolo')
HOST = 'resource.pbiomon.cn'
PRIVATE = '10.249.0.110'
publics = sorted({x[4][0] for x in socket.getaddrinfo(HOST, 443, type=socket.SOCK_STREAM) if ipaddress.ip_address(x[4][0]).is_global})
if not publics:
    raise RuntimeError('no_current_public_address_for_bounded_diagnosis')
PUBLIC = publics[0]
CTX = ssl.create_default_context()
class Pinned(http.client.HTTPSConnection):
    def __init__(self, ip):
        super().__init__(HOST, 443, timeout=10, context=CTX)
        self.ip = ip
    def connect(self):
        self.sock = self._context.wrap_socket(socket.create_connection((self.ip, self.port), self.timeout), server_hostname=HOST)
def probe(url, ip):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname != HOST:
        return {'error_type': 'UnexpectedHost'}
    target = parsed.path or '/'
    if parsed.query:
        target += '?' + parsed.query
    c = Pinned(ip)
    start = time.monotonic()
    try:
        c.connect()
        peer = c.sock.getpeername()[0]
        c.request('GET', target, headers={'Host': HOST, 'Range': 'bytes=0-1023', 'Connection': 'close', 'User-Agent': 'Pbiomon-bounded-source-diagnostic/1.0'})
        response = c.getresponse()
        prefix = response.read(1024)
        allowed = ('Server','Content-Type','Content-Length','Content-Range','Accept-Ranges','ETag','Last-Modified','Age','X-Cache')
        return {'http_status': response.status, 'peer_ip': peer, 'tls_verified': True, 'elapsed_s': round(time.monotonic()-start, 4), 'headers': {key: response.getheader(key) for key in allowed if response.getheader(key) is not None}, 'body_prefix_bytes': len(prefix), 'body_prefix_sha256': hashlib.sha256(prefix).hexdigest(), 'body_markers': [name for name, marker in [('nginx_404',b'404 Not Found'),('object_missing',b'NoSuchKey'),('access_denied',b'AccessDenied')] if marker in prefix]}
    except Exception as exc:
        return {'error_type': type(exc).__name__, 'elapsed_s': round(time.monotonic()-start, 4)}
    finally:
        c.close()
connection = sqlite3.connect('file:'+str(ROOT/'state/state.sqlite')+'?mode=ro', uri=True)
connection.row_factory = sqlite3.Row
failures = [dict(x) for x in connection.execute('SELECT resource_type,resource_id,download_url,metadata_json FROM resources WHERE last_error=?', ('resource_http_404',))]
random.Random(20261102).shuffle(failures)
selected = []
seen_groups = set()
for row in failures:
    metadata = json.loads(row['metadata_json'])
    group = metadata.get('monitorUnitId')
    if group not in seen_groups:
        selected.append((row, 'failed_sample'))
        seen_groups.add(group)
    if len(selected) >= 12:
        break
chosen = {row['resource_id'] for row, _ in selected}
for row in failures:
    if len(selected) >= 20: break
    if row['resource_id'] not in chosen:
        selected.append((row, 'failed_sample'))
        chosen.add(row['resource_id'])
for kind, limit in [('PHOTO',4),('VIDEO',2)]:
    for row in connection.execute('SELECT resource_type,resource_id,download_url,metadata_json FROM resources WHERE resource_type=? AND status=? AND source_transport=? ORDER BY completed_at DESC LIMIT ?', (kind,'done','private_https',limit)):
        selected.append((dict(row),'successful_control'))
connection.close()
def diagnose(item):
    row, role = item
    parsed = urllib.parse.urlsplit(row['download_url'])
    metadata = json.loads(row['metadata_json'])
    source_path = metadata.get('resourcePath')
    expected = '/active/profiles/' + source_path.lstrip('/') if isinstance(source_path,str) else None
    a, b = probe(row['download_url'], PRIVATE), probe(row['download_url'], PUBLIC)
    sa, sb = a.get('http_status'), b.get('http_status')
    if sa in (404,410) and sb in (404,410): outcome='both_endpoints_missing'
    elif sa in (404,410) and sb in (200,206): outcome='private_missing_public_available'
    elif sa in (200,206) and sb in (200,206): outcome='both_available'
    elif sa in (200,206) and sb in (404,410): outcome='private_available_public_missing'
    else: outcome='other_or_connection_failure'
    return {'resource_type':row['resource_type'],'resource_id':row['resource_id'],'role':role,'metadata_path_matches_url': None if expected is None else urllib.parse.unquote(parsed.path)==expected,'query_present':bool(parsed.query),'private':a,'public':b,'outcome':outcome}
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(diagnose,item) for item in selected]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())
        if len(results)%5==0: print(json.dumps({'event':'diagnostic_progress','completed':len(results)},ensure_ascii=False),flush=True)
report={'at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'public_ip':PUBLIC,'private_ip':PRIVATE,'failed_pool_count':len(failures),'sample_count':len(results),'failed_sample_outcomes':dict(collections.Counter(x['outcome'] for x in results if x['role']=='failed_sample')),'control_outcomes':dict(collections.Counter(x['outcome'] for x in results if x['role']=='successful_control')),'samples':results,'notes':['Bounded diagnostic requests read at most 1024 body bytes. Production downloads remain private-only.','Both-endpoint HTTP 404 means this URL is unavailable at both tested endpoints, not proof that the original storage file is physically absent.']}
path=ROOT/'reports/404-diagnostic.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'event':'diagnostic_complete','report_file':str(path), **{k:v for k,v in report.items() if k not in ('samples','notes')}},ensure_ascii=False),flush=True)
