"""Behavior checks use disposable files and synthetic source records only."""
import concurrent.futures
import copy
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / 'tools/pbiomon_sync.py'
SPEC = importlib.util.spec_from_file_location('pbiomon_sync', MODULE)
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


def record(rid='A', species='s1', kind='PHOTO', url=None):
    return {
        'resourceId': rid, 'resourceType': kind, 'monitorUnitId': 'unit1',
        'monitorUnitName': 'test', 'fileName': 'same-name.jpg',
        'capturedAt': '2020-01-01 00:00:00', 'resourcePath': None,
        'downloadUrl': url or 'https://resource.pbiomon.cn/test.jpg?token=do-not-log',
        'speciesId': species, 'speciesName': 'species-' + species,
        'finalIdentification': {
            'id': 'final-' + species, 'identifiedAt': '2026-11-01 12:00:00',
            'identifierId': None, 'identifierName': None, 'remarks': [],
            'boxCount': 1, 'boxesComplete': False,
            'boxes': [{'left': 10, 'top': 20, 'width': 30, 'height': 40,
                       'isAI': None, 'frameId': None if kind == 'PHOTO' else 1,
                       'tsMs': None if kind == 'PHOTO' else 40,
                       'coordinateUnit': 'UNKNOWN', 'referenceWidth': None,
                       'referenceHeight': None, 'complete': False}],
        },
        'ai': {'hasResult': False, 'comparisonStatus': 'NO_RESULT', 'species': [], 'isConsistent': None},
    }


def page(records, cursor=None):
    return {'code': 0, 'msg': '成功', 'data': {'records': records, 'count': len(records),
            'pageSize': 200, 'hasMore': cursor is not None, 'nextCursor': cursor}}


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = sync.Store(self.root)
        self.run, _ = self.store.create_or_resume_run('2026-11-02', '2026-11-01', {})
        self.task = self.store.next_task(self.run)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def ingest(self, records, cursor=None, task=None):
        task = task or self.store.task(self.task['task_id'])
        payload = page(records, cursor)
        raw = sync.save_page(self.root, task, payload)
        self.store.ingest_page(task, payload, raw)
        return task, payload, raw

    def resource(self, rid='A', kind='PHOTO'):
        return dict(self.store.db.execute('SELECT * FROM resources WHERE resource_id=? AND resource_type=?', (rid, kind)).fetchone())

    def receipt(self, row):
        path = self.root / 'raw/objects/test.bin'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'complete synthetic binary')
        return {'resource_type': row['resource_type'], 'resource_id': row['resource_id'],
                'file_path': str(path.relative_to(self.root)), 'sha256': sync.sha256_file(path),
                'byte_size': path.stat().st_size, 'file_mtime_ns': path.stat().st_mtime_ns,
                'content_type': 'image/jpeg', 'width': 100, 'height': 80, 'media': {}, 'completed_at': sync.now()}

    def test_cross_page_species_merge_does_not_create_second_download(self):
        self.ingest([record(species='s1')], 'cursor-2')
        claimed = self.store.claim(('PHOTO',), 'worker-1')
        self.assertIsNotNone(claimed)
        self.ingest([record(species='s2')])
        self.assertIsNone(self.store.claim(('PHOTO',), 'worker-2'))
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM resources').fetchone()[0], 1)
        self.store.export_annotations()
        row = self.resource()
        merged = json.loads((self.root / row['annotation_path']).read_text())
        self.assertEqual({a['speciesId'] for a in merged['annotations']}, {'s1', 's2'})
        self.assertFalse(merged['collection']['trainingExported'])

    def test_identical_page_replay_is_idempotent_including_checkpoint(self):
        task, payload, raw = self.ingest([record()], 'cursor-2')
        self.assertEqual(self.store.ingest_page(task, payload, raw), 0)
        saved = self.store.task(task['task_id'])
        self.assertEqual(saved['page_number'], 1)
        self.assertEqual(saved['records_seen'], 1)
        self.assertEqual(saved['cursor'], 'cursor-2')

    def test_done_file_survives_new_species_and_changed_download_url(self):
        self.ingest([record()], 'c2')
        row = self.store.claim(('PHOTO',), 'worker')
        self.store.completed(row, self.receipt(row))
        self.ingest([record(species='s2', url='https://resource.pbiomon.cn/test.jpg?token=new')])
        self.assertEqual(self.resource()['status'], 'done')
        self.assertIsNone(self.store.claim(('PHOTO',), 'another'))
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM annotations').fetchone()[0], 2)

    def test_concurrent_claims_have_single_winner(self):
        self.ingest([record()])
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            claims = list(pool.map(lambda n: self.store.claim(('PHOTO',), str(n)), range(24)))
        self.assertEqual(sum(x is not None for x in claims), 1)
        self.assertEqual(self.resource()['attempts'], 1)

    def test_bad_record_rolls_back_all_rows_and_cursor(self):
        invalid = record('B')
        invalid['finalIdentification']['boxCount'] = 99
        with self.assertRaises(RuntimeError):
            self.ingest([record(), invalid], 'must-not-commit')
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM resources').fetchone()[0], 0)
        saved = self.store.task(self.task['task_id'])
        self.assertIsNone(saved['cursor'])
        self.assertEqual(saved['page_number'], 0)

    def test_out_of_range_server_date_does_not_advance_checkpoint(self):
        data = record()
        data['finalIdentification']['identifiedAt'] = '2024-01-01 00:00:00'
        with self.assertRaises(RuntimeError):
            self.ingest([data])
        self.assertEqual(self.store.task(self.task['task_id'])['page_number'], 0)

    def test_annotation_change_retains_observed_versions(self):
        self.ingest([record()], 'c2')
        self.store.export_annotations()
        original = record()
        original['finalIdentification']['boxes'][0]['left'] = 11
        self.ingest([original])
        self.store.export_annotations()
        current = self.root / self.resource()['annotation_path']
        versions = [x for x in current.parent.glob('*.json') if x.name != 'current.json']
        self.assertEqual(len(versions), 2)
        self.assertEqual(json.loads(current.read_text())['annotations'][0]['finalIdentification']['boxes'][0]['left'], 11)

    def test_crash_after_receipt_recovers_without_network(self):
        self.ingest([record()])
        row = self.store.claim(('PHOTO',), 'old-worker')
        downloader = sync.Downloader(self.store, '10.249.0.110', 1, threading.Event(), lambda *a, **k: None)
        receipt = self.receipt(row)
        sync.atomic_json(downloader.receipt_path(row), receipt)
        self.assertEqual(self.store.recover_claims(), 1)
        claimed = self.store.claim(('PHOTO',), 'new-worker')
        recovered, reused = downloader.download(claimed)
        self.assertTrue(reused)
        self.assertEqual(recovered['sha256'], receipt['sha256'])
        self.assertEqual(self.resource()['status'], 'done')

    def test_deleted_local_file_is_requeued_on_restart(self):
        self.ingest([record()])
        row = self.store.claim(('PHOTO',), 'worker')
        receipt = self.receipt(row)
        self.store.completed(row, receipt)
        (self.root / receipt['file_path']).unlink()
        self.assertEqual(self.store.reconcile_completed(), 1)
        self.assertEqual(self.resource()['status'], 'retry')

    def test_same_size_file_with_changed_mtime_is_rehashed(self):
        self.ingest([record()])
        row = self.store.claim(('PHOTO',), 'worker')
        receipt = self.receipt(row)
        self.store.completed(row, receipt)
        target = self.root / receipt['file_path']
        target.write_bytes(b'x' * receipt['byte_size'])
        # Linux may coalesce two immediate writes into the same timestamp tick.
        # Exercise the documented mtime-triggered restart check deterministically.
        os.utime(target, ns=(receipt['file_mtime_ns'], receipt['file_mtime_ns'] + 1_000_000_000))
        self.assertEqual(self.store.reconcile_completed(), 1)

    def test_video_has_independent_identity_and_keeps_frame_fields(self):
        self.ingest([record(kind='PHOTO'), record(kind='VIDEO')])
        self.assertIsNotNone(self.store.claim(('PHOTO',), 'image-worker'))
        video = self.store.claim(('VIDEO',), 'video-worker')
        self.assertIsNotNone(video)
        self.store.export_annotations()
        data = json.loads((self.root / self.resource(kind='VIDEO')['annotation_path']).read_text())
        box = data['annotations'][0]['finalIdentification']['boxes'][0]
        self.assertEqual((box['frameId'], box['tsMs']), (1, 40))

    def test_curl_is_private_only_and_url_never_in_argv(self):
        self.ingest([record()])
        row = self.resource()
        downloader = sync.Downloader(self.store, '10.249.0.110', 1, threading.Event(), lambda *a, **k: None)
        command, config = downloader.command(row, self.root / 'temp')
        self.assertIn('resource.pbiomon.cn:443:10.249.0.110', command)
        self.assertEqual(command[command.index('--noproxy') + 1], '*')
        self.assertNotIn('--location', command)
        self.assertNotIn('--insecure', command)
        self.assertNotIn('-k', command)
        self.assertIn('--globoff', command)
        self.assertIn('--path-as-is', command)
        self.assertNotIn('do-not-log', ' '.join(command))
        self.assertIn('do-not-log', config)

    def test_other_download_domains_are_blocked_not_public_fallback(self):
        self.ingest([record(url='https://other.example.invalid/image.jpg')])
        downloader = sync.Downloader(self.store, '10.249.0.110', 1, threading.Event(), lambda *a, **k: None)
        with self.assertRaises(sync.DownloadError):
            downloader.command(self.resource(), self.root / 'temp')

    def test_summary_contains_no_download_url_or_access_token(self):
        self.ingest([record()])
        summary = json.dumps(self.store.summary())
        self.assertNotIn('do-not-log', summary)
        self.assertNotIn('downloadUrl', summary)
        self.assertEqual(self.store.summary()['yolo_training_exports'], 0)

    def test_live_catalog_never_removes_unseen_prior_species(self):
        self.ingest([record(species='s1'), record(species='s2')], 'c2')
        self.ingest([record(species='s1')])
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM annotations').fetchone()[0], 2)

    def test_reopen_database_preserves_completed_file(self):
        self.ingest([record()])
        row = self.store.claim(('PHOTO',), 'worker')
        self.store.completed(row, self.receipt(row))
        self.store.close()
        self.store = sync.Store(self.root)
        self.assertEqual(self.store.reconcile_completed(), 0)
        self.assertIsNone(self.store.claim(('PHOTO', 'VIDEO'), 'new-process'))

    def test_unfinished_catalog_cannot_advance_watermark(self):
        with self.assertRaises(RuntimeError):
            self.store.finish_catalog(self.run)
        self.assertNotIn('catalog_end_date', self.store.summary()['coverage'])

    def test_bounded_sample_does_not_establish_history_watermark(self):
        self.ingest([record()])
        self.assertEqual(self.store.finish_catalog(self.run), 'catalog_complete')
        self.assertNotIn('catalog_end_date', self.store.summary()['coverage'])

    def test_completed_history_advances_to_query_cutoff_not_finish_date(self):
        with self.store.transaction() as db:
            db.execute("UPDATE runs SET status='complete' WHERE run_id=?", (self.run,))
        historical, _ = self.store.create_or_resume_run('2026-11-01', None, {})
        with self.store.transaction() as db:
            db.execute("UPDATE tasks SET status='complete' WHERE run_id=?", (historical,))
        self.store.finish_catalog(historical)
        self.assertEqual(self.store.summary()['coverage']['catalog_end_date'], '2026-11-01')

    def test_resume_keeps_checkpoint_and_clears_paused_finish_time(self):
        self.ingest([record()], 'resume-cursor')
        with self.store.transaction() as db:
            db.execute("UPDATE runs SET status='paused',finished_at=? WHERE run_id=?", (sync.now(),self.run))
        run,resumed=self.store.create_or_resume_run('2026-11-02','2026-11-01',{})
        self.assertTrue(resumed)
        self.assertEqual(run,self.run)
        self.assertEqual(self.store.task(self.task['task_id'])['cursor'],'resume-cursor')
        self.assertIsNone(self.store.summary()['run']['finished_at'])

    def test_404_is_held_after_first_attempt_without_losing_annotations(self):
        self.ingest([record()], 'next')
        row = self.store.claim(('PHOTO',), 'worker')
        self.assertEqual(self.store.failed(row, 'resource_http_404'), 'unavailable')
        self.assertEqual(self.resource()['attempts'], 1)
        self.assertEqual(self.resource()['retry_after'], 0)
        self.assertIsNone(self.store.claim(('PHOTO',), 'another'))
        self.ingest([record(species='s2')])
        self.assertEqual(self.resource()['status'], 'unavailable')
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM annotations').fetchone()[0], 2)
        self.assertEqual(self.store.held_count(), 1)
        self.assertEqual(self.store.summary()['failures'][0]['error'], 'resource_http_404')

    def test_changed_url_releases_held_404_for_one_new_attempt(self):
        self.ingest([record()], 'next')
        row = self.store.claim(('PHOTO',), 'worker')
        self.store.failed(row, 'resource_http_404')
        self.ingest([record(url='https://resource.pbiomon.cn/repaired.jpg')])
        self.assertEqual(self.resource()['status'], 'pending')
        self.assertEqual(self.resource()['attempts'], 0)
        self.assertEqual(self.store.claim(('PHOTO',), 'worker')['attempts'], 1)

    def test_old_inflight_url_failure_does_not_block_refreshed_url(self):
        self.ingest([record()], 'next')
        old = self.store.claim(('PHOTO',), 'worker')
        self.ingest([record(url='https://resource.pbiomon.cn/new-link.jpg')])
        self.assertEqual(self.store.failed(old, 'resource_http_404'), 'pending')
        self.assertEqual(self.resource()['attempts'], 0)

    def test_generic_retry_flag_does_not_reactivate_held_missing_urls(self):
        self.ingest([record()])
        row = self.store.claim(('PHOTO',), 'worker')
        self.store.failed(row, 'resource_http_404')
        self.assertEqual(self.store.retry_failed(), 0)
        self.assertEqual(self.resource()['status'], 'unavailable')
        self.assertEqual(self.store.recheck_unavailable(), 1)
        self.assertIsNotNone(self.store.claim(('PHOTO',), 'explicit-recheck'))

    def test_legacy_hot_retry_queue_is_migrated_without_touching_transients(self):
        self.ingest([record('A'),record('B'),record('C')])
        with self.store.transaction() as db:
            db.execute("UPDATE resources SET status='retry',attempts=2,last_error=? WHERE resource_id=?", ('resource_http_404','A'))
            db.execute("UPDATE resources SET status='failed',attempts=5,last_error=? WHERE resource_id=?", ('resource_http_410','B'))
            db.execute("UPDATE resources SET status='retry',attempts=1,last_error=? WHERE resource_id=?", ('resource_http_503','C'))
        self.assertEqual(self.store.settle_held_failures()['unavailable'], 2)
        self.assertEqual(self.resource('A')['status'], 'unavailable')
        self.assertEqual(self.resource('B')['status'], 'unavailable')
        self.assertEqual(self.resource('C')['status'], 'retry')
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM annotations').fetchone()[0], 3)

    def test_fresh_work_has_priority_but_due_retries_do_not_starve(self):
        self.ingest([record('0-retry')]+[record('F{:02d}'.format(i)) for i in range(10)])
        with self.store.transaction() as db:
            db.execute("UPDATE resources SET status='retry',retry_after=0 WHERE resource_id=?", ('0-retry',))
        first = [self.store.claim(('PHOTO',), str(i)) for i in range(9)]
        self.assertTrue(all(row['resource_id'].startswith('F') for row in first))
        self.assertEqual(self.store.claim(('PHOTO',), 'retry-slot')['resource_id'], '0-retry')

    def test_resource_auth_failures_wait_for_refresh_instead_of_hot_retry(self):
        self.ingest([record()])
        row = self.store.claim(('PHOTO',), 'worker')
        self.assertEqual(self.store.failed(row, 'resource_http_403'), 'refresh_required')
        self.assertIsNone(self.store.claim(('PHOTO',), 'worker'))

    def fake_curl(self, status, peer='10.249.0.110', retry_after=None, body=b'synthetic transport body'):
        def start(command, **kwargs):
            output=Path(command[command.index('--output')+1])
            output.write_bytes(body if status=='200' else b'')
            headers='HTTP/1.1 '+status+' Test\r\n'
            if retry_after:
                headers+='Retry-After: '+retry_after+'\r\n'
            Path(command[command.index('--dump-header')+1]).write_text(headers+'\r\n')
            process=mock.Mock()
            process.returncode=0 if status=='200' else 22
            process.communicate.return_value=('\t'.join([status,peer,str(len(body) if status=='200' else 0),'image/jpeg','0']),'')
            return process
        return start

    def test_resource_429_preserves_retry_after_from_real_header_parsing(self):
        self.ingest([record()])
        row=self.store.claim(('PHOTO',),'worker')
        downloader=sync.Downloader(self.store,'10.249.0.110',1,threading.Event(),lambda *a,**k:None)
        with mock.patch.object(sync.subprocess,'Popen',side_effect=self.fake_curl('429',retry_after='120')):
            with self.assertRaises(sync.DownloadError) as caught:
                downloader.download(row)
        self.assertEqual(caught.exception.code,'resource_http_429')
        self.assertEqual(caught.exception.retry_after,120)
        self.store.failed(row,caught.exception.code,server_delay=caught.exception.retry_after)
        self.assertGreaterEqual(self.resource()['retry_after'],sync.time.time()+119)

    def test_failed_http_response_cannot_hide_an_unexpected_public_peer(self):
        self.ingest([record()])
        row=self.store.claim(('PHOTO',),'worker')
        downloader=sync.Downloader(self.store,'10.249.0.110',1,threading.Event(),lambda *a,**k:None)
        with mock.patch.object(sync.subprocess,'Popen',side_effect=self.fake_curl('404',peer='36.110.25.123')):
            with self.assertRaises(sync.DownloadError) as caught:
                downloader.download(row)
        self.assertEqual(caught.exception.code,'private_route_or_tls_mismatch')

    def test_new_receipt_preserves_source_link_and_metadata_hash_without_url(self):
        self.ingest([record()])
        row=self.store.claim(('PHOTO',),'worker')
        downloader=sync.Downloader(self.store,'10.249.0.110',1,threading.Event(),lambda *a,**k:None)
        with mock.patch.object(sync.subprocess,'Popen',side_effect=self.fake_curl('200')):
            with mock.patch.object(downloader,'media_info',return_value=('.jpg',100,80,{'format':'test'})):
                receipt,reused=downloader.download(row)
        self.assertFalse(reused)
        self.assertEqual(receipt['source_url_sha256'],sync.hashlib.sha256(row['download_url'].encode()).hexdigest())
        self.assertEqual(receipt['source_resource_metadata_sha256'],sync.digest(json.loads(row['metadata_json'])))
        self.assertNotIn('do-not-log',json.dumps(receipt))
        self.assertEqual(self.resource()['status'],'done')


class RetryPolicyTests(unittest.TestCase):
    def test_transient_backoff_is_bounded_and_server_delay_is_honored(self):
        with mock.patch.object(sync.random, 'uniform', return_value=0):
            self.assertEqual(sync.retry_policy('resource_http_503', 1), ('retry',30))
            self.assertEqual(sync.retry_policy('curl_exit_28', 2), ('retry',60))
            self.assertEqual(sync.retry_policy('resource_http_429', 1), ('retry',65))
            self.assertEqual(sync.retry_policy('resource_http_503', 1, server_delay=600), ('retry',600))
            self.assertEqual(sync.retry_policy('curl_exit_28', 5), ('failed',0))

    def test_status_specific_failures_are_not_implicitly_retried(self):
        for code,state in [('resource_http_404','unavailable'),('resource_http_410','unavailable'),('resource_http_401','refresh_required'),('resource_http_403','refresh_required'),('resource_http_400','failed'),('curl_exit_3','blocked_url'),('curl_exit_60','blocked_transport')]:
            with self.subTest(code=code):
                self.assertEqual(sync.retry_policy(code,1),(state,0))

    def test_retry_after_accepts_seconds_and_http_date(self):
        self.assertEqual(sync.retry_after_seconds('120'),120)
        self.assertEqual(sync.retry_after_seconds('Thu, 01 Jan 1970 00:02:00 GMT', reference_time=0),120)
        self.assertIsNone(sync.retry_after_seconds('not-a-date'))
        self.assertIsNone(sync.retry_after_seconds('-1'))


class AssociationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('pbiomon_audit',MODULE.with_name('pbiomon_audit.py'))
        cls.audit=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.audit)

    def setUp(self):
        self.annotation=self.audit.normalized_annotation(record())
        self.resource={'resource_type':'PHOTO','resource_id':'A','sha256':'hash-A','file_path':'raw/objects/hash-A.jpg'}
        self.annotations=[{'species_id':'s1','annotation_hash':sync.digest(self.annotation)}]
        self.bundle={'resource':{'resourceType':'PHOTO','resourceId':'A'},'file':{'status':'done','sha256':'hash-A','path':'raw/objects/hash-A.jpg'},'annotations':[self.annotation]}

    def test_correct_source_and_asset_mapping_passes(self):
        self.assertTrue(self.audit.bundle_matches(self.bundle,self.resource,self.annotations))

    def test_swapped_image_identity_or_object_is_detected(self):
        changed=copy.deepcopy(self.bundle)
        changed['resource']['resourceId']='B'
        self.assertFalse(self.audit.bundle_matches(changed,self.resource,self.annotations))
        changed=copy.deepcopy(self.bundle)
        changed['file']['sha256']='hash-B'
        self.assertFalse(self.audit.bundle_matches(changed,self.resource,self.annotations))

    def test_missing_duplicate_or_changed_species_annotation_is_detected(self):
        for entries in [[],[self.annotation,self.annotation]]:
            changed=copy.deepcopy(self.bundle)
            changed['annotations']=entries
            self.assertFalse(self.audit.bundle_matches(changed,self.resource,self.annotations))
        changed=copy.deepcopy(self.bundle)
        changed['annotations'][0]['finalIdentification']['boxes'][0]['left']=99
        self.assertFalse(self.audit.bundle_matches(changed,self.resource,self.annotations))

    def test_receipt_source_hash_must_belong_to_same_observed_record_pair(self):
        pairs={('url-A','metadata-A'),('url-B','metadata-B')}
        self.assertTrue(self.audit.receipt_source_matches({'source_url_sha256':'url-A','source_resource_metadata_sha256':'metadata-A'},pairs))
        self.assertFalse(self.audit.receipt_source_matches({'source_url_sha256':'url-A','source_resource_metadata_sha256':'metadata-B'},pairs))
        self.assertFalse(self.audit.receipt_source_matches({'source_url_sha256':'other-resource-url'},pairs))

    def test_legacy_receipt_is_reported_as_unproven_not_falsely_verified(self):
        self.assertIsNone(self.audit.receipt_source_matches({'sha256':'local-hash'},set()))

    def test_source_pair_is_stable_across_species_but_changes_with_asset_url(self):
        first=record(species='s1')
        second=record(species='s2')
        self.assertEqual(self.audit.source_provenance_pair(first),self.audit.source_provenance_pair(second))
        second['downloadUrl']='https://resource.pbiomon.cn/other.jpg'
        self.assertNotEqual(self.audit.source_provenance_pair(first),self.audit.source_provenance_pair(second))


class RangeTests(unittest.TestCase):
    def test_history_covers_all_dates_without_assuming_oldest_year(self):
        ranges = sync.monthly_ranges('2026-11-02')
        ascending = list(reversed(ranges))
        self.assertIsNone(ascending[0][0])
        self.assertEqual(ascending[-1][1], '2026-11-02')
        for previous, current in zip(ascending, ascending[1:]):
            self.assertEqual(sync.dt.date.fromisoformat(previous[1]) + sync.dt.timedelta(days=1), sync.dt.date.fromisoformat(current[0]))

    def test_adaptive_split_has_no_gap_or_overlap(self):
        ranges = sync.split_range('2026-10-01', '2026-10-31')
        self.assertEqual(ranges, [('2026-10-01', '2026-10-16'), ('2026-10-17', '2026-10-31')])
        self.assertIsNone(sync.split_range('2026-11-02', '2026-11-02'))

    def test_unbounded_split_preserves_old_history(self):
        ranges = sync.split_range(None, '2019-12-31', '2015-01-01')
        self.assertIsNone(ranges[0][0])
        self.assertEqual(ranges[-1][1], '2019-12-31')


if __name__ == '__main__':
    unittest.main()
