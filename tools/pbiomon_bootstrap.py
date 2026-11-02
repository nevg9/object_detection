"""One-time local-cache import and private-network live verification. No URL logging."""
import collections
import getpass
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid

sys.dont_write_bytecode = True
os.umask(0o077)
ROOT = Path('/mnt/data2/pbiomon_yolo')
SOURCE = Path('/home/yuzhong/pbiomon_api_benchmark/20261102_153313')
SPEC = importlib.util.spec_from_file_location('pbiomon_sync', ROOT / 'bin/pbiomon_sync.py')
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)
lock = sync.lock_process(ROOT)
store = sync.Store(ROOT)
log = sync.Log(ROOT / 'reports/bootstrap-events.jsonl')
stop = threading.Event()
downloader = sync.Downloader(store, '10.249.0.110', 100, stop, log)
report = {'started_at': sync.now(), 'requested_full_history_cutoff': '2026-11-01', 'cache_import': {}, 'live_checks': []}


def partial_task(name, start, end, config):
    run_id = name + '-' + uuid.uuid4().hex[:12]
    with store.transaction() as db:
        db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?)', (run_id, start, end, 'validation_only', sync.now(), None, sync.canonical(config)))
        key = store._add_task(db, run_id, start, end)
    return store.task(key)


def accept_records(task, records, origin):
    payload = {'code': 0, 'msg': 'local_import_of_observed_records', 'data': {'records': records, 'count': len(records), 'pageSize': 200, 'hasMore': False, 'nextCursor': None}, 'local_provenance': origin}
    raw = sync.save_page(ROOT, task, payload)
    store.ingest_page(task, payload, raw)
    with store.transaction() as db:
        db.execute("UPDATE tasks SET status='imported_partial' WHERE task_id=?", (task['task_id'],))
    return task


try:
    imported = reused = missing = 0
    manifest = SOURCE / 'downloads.jsonl'
    if manifest.is_file():
        prior = [json.loads(line) for line in manifest.read_text().splitlines()]
        for item in prior:
            if not item.get('ok'):
                continue
            original = Path(item['image_file']).resolve()
            annotation_path = Path(item['annotation_file']).resolve()
            if SOURCE.resolve() not in original.parents or SOURCE.resolve() not in annotation_path.parents:
                raise RuntimeError('cache_path_outside_known_source')
            if not original.is_file() or original.stat().st_size != item['bytes']:
                missing += 1
                continue
            annotations = json.loads(annotation_path.read_text())
            records = annotations['records']
            if not records or any(r['resourceId'] != item['resourceId'] or r['resourceType'] != 'PHOTO' or r['finalIdentification']['identifiedAt'][:10] > '2026-11-01' for r in records):
                raise RuntimeError('cache_identity_or_cutoff_mismatch')
            task = partial_task('cache', None, '2026-11-01', {'purpose': 'import_prior_verified_sample_not_a_complete_catalog'})
            accept_records(task, records, {'kind': 'prior_benchmark_local_cache', 'source_annotation': str(annotation_path)})
            row = dict(store.db.execute('SELECT * FROM resources WHERE resource_type=? AND resource_id=?', ('PHOTO', item['resourceId'])).fetchone())
            if row['status'] == 'done' and row['file_path'] and (ROOT / row['file_path']).is_file():
                reused += 1
                continue
            ext, width, height, media = downloader.media_info(row, original, item.get('content_type', 'image/jpeg'))
            hashed = sync.sha256_file(original)
            target = ROOT / 'raw/objects' / hashed[:2] / (hashed + ext)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temp = ROOT / 'raw/partial' / (row['resource_key'] + '.cache-import.part')
                shutil.copyfile(original, temp)
                if sync.sha256_file(temp) != hashed:
                    raise RuntimeError('cache_copy_checksum_mismatch')
                with temp.open('rb') as handle:
                    os.fsync(handle.fileno())
                try:
                    os.link(temp, target)
                except FileExistsError:
                    pass
                temp.unlink()
            if sync.sha256_file(target) != hashed:
                raise RuntimeError('cache_target_checksum_mismatch')
            receipt = {'schemaVersion': 1, 'resource_type': 'PHOTO', 'resource_id': item['resourceId'], 'file_path': str(target.relative_to(ROOT)), 'sha256': hashed, 'byte_size': target.stat().st_size, 'file_mtime_ns': target.stat().st_mtime_ns, 'content_type': item.get('content_type', 'image/jpeg'), 'width': width, 'height': height, 'media': {**media, 'imported_from_prior_verified_cache': True}, 'transport': 'local_cache', 'original_remote_ip': 'not_recorded', 'completed_at': sync.now()}
            sync.atomic_json(downloader.receipt_path(row), receipt)
            store.completed(row, receipt)
            imported += 1
            if imported % 50 == 0:
                log('cache_import_progress', imported=imported, reused=reused)
        while store.export_annotations(100):
            pass
    report['cache_import'] = {'imported': imported, 'already_present': reused, 'missing_sources': missing, 'network_requests': 0}
    log('cache_import_complete', **report['cache_import'])
    secret = getpass.getpass('API app secret (hidden; memory only): ')
    api = sync.Api('https://open.pbiomon.cn', 'ai_p_20261102', secret, 6.5, stop, log)
    filters = [
        ('VIDEO', '2023-12-21', {'speciesName': '小麂', 'monitorUnitIds': ['4ff553e93abd4f37ac0dd0e35886b308']}),
        ('PHOTO', '2026-10-19', {'speciesName': '小麂', 'monitorUnitIds': ['4ff553e93abd4f37ac0dd0e35886b308'], 'aiConsistency': 1}),
    ]
    for wanted, date, extra in filters:
        body = {'identifiedDateStart': date, 'identifiedDateEnd': date, 'pageSize': 200, **extra}
        found = []
        for page_number in range(1, 4):
            payload = api.query(body)
            records = payload['data']['records']
            task = partial_task('live-probe', date, date, {'purpose': 'filtered_live_verification_not_a_complete_catalog', 'filters': extra})
            # Preserve the real source page including its actual pagination fields.
            raw = sync.save_page(ROOT, task, payload)
            store.ingest_page(task, payload, raw)
            with store.transaction() as db:
                db.execute("UPDATE tasks SET status='validation_only' WHERE task_id=?", (task['task_id'],))
            log('live_probe_catalog_page', wanted=wanted, count=len(records), type_counts=dict(collections.Counter(r['resourceType'] for r in records)), has_more=payload['data']['hasMore'])
            found = [r for r in records if r['resourceType'] == wanted]
            if found or not payload['data']['hasMore']:
                break
            body['cursor'] = payload['data']['nextCursor']
        if not found:
            report['live_checks'].append({'type': wanted, 'verified': False, 'reason': 'no_matching_source_sample_in_bounded_probe'})
            continue
        record = found[0]
        row = dict(store.db.execute('SELECT * FROM resources WHERE resource_type=? AND resource_id=?', (wanted, record['resourceId'])).fetchone())
        receipt, reused_receipt = downloader.download(row)
        # A second call must reuse the durable verified receipt without a request.
        receipt_2, reused_second = downloader.download(row)
        check = {'type': wanted, 'resource_key': row['resource_key'], 'verified': True, 'bytes': receipt['byte_size'], 'sha256': receipt['sha256'], 'private_ip': receipt.get('remote_ip'), 'tls_verified': receipt.get('tls_verified'), 'second_call_reused_receipt': reused_second, 'checksum_stable': receipt_2['sha256'] == receipt['sha256'], 'media': receipt.get('media')}
        report['live_checks'].append(check)
        log('live_probe_verified', **check)
    while store.export_annotations(100):
        pass
    report['finished_at'] = sync.now()
    report['summary'] = store.summary()
    sync.atomic_json(ROOT / 'reports/bootstrap.json', report)
    log('bootstrap_complete', cached=report['cache_import'], checks=[{'type': x['type'], 'verified': x['verified']} for x in report['live_checks']])
finally:
    store.close()
    lock.close()
    secret = None
