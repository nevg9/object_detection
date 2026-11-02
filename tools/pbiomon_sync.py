#!/usr/bin/env python3
"""Durable Pbiomon final-identification catalog and private-network asset download.

The API secret is supplied through a terminal prompt or environment, never saved.
This tool archives observed source annotations; it does not infer their coordinate
system, create YOLO labels, or claim a point-in-time server snapshot.
"""

import argparse
import base64
import concurrent.futures
import contextlib
import datetime as dt
import email.utils
import fcntl
import getpass
import gzip
import hashlib
import ipaddress
import json
import os
import random
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import hmac

VERSION = "1.1.0"
SCHEMA_VERSION = 1
TZ = dt.timezone(dt.timedelta(hours=8))
QUERY_PATH = "/api/infcam/final-identifications/query"
RESOURCE_HOST = "resource.pbiomon.cn"
DEFAULT_ROOT = "/mnt/data2/pbiomon_yolo"
HELD_STATUSES = ('failed', 'blocked_url', 'no_url', 'unavailable', 'refresh_required', 'blocked_transport')


def retry_after_seconds(value, reference_time=None):
    if not value:
        return None
    try:
        seconds = float(value)
        if seconds >= 0 and seconds < float('inf'):
            return seconds
    except (ValueError, TypeError):
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return max(0, parsed.timestamp() - (time.time() if reference_time is None else reference_time))
    except (ValueError, TypeError, OverflowError):
        return None


def retry_policy(code, attempts, permanent=False, maximum=5, server_delay=None):
    """404 is a held URL, not evidence that the storage file is permanently gone."""
    if code.startswith('url_') or code == 'curl_exit_3':
        return 'blocked_url', 0
    if code in ('private_route_or_tls_mismatch', 'curl_exit_60', 'curl_exit_51'):
        return 'blocked_transport', 0
    if code in ('resource_http_404', 'resource_http_410'):
        return 'unavailable', 0
    if code in ('resource_http_401', 'resource_http_403'):
        return 'refresh_required', 0
    if code.startswith('resource_http_'):
        try:
            status = int(code.rsplit('_', 1)[1])
        except ValueError:
            status = 0
        if 400 <= status < 500 and status not in (408, 429):
            return 'failed', 0
    if permanent or attempts >= maximum:
        return 'failed', 0
    delay = min(300, 30 * 2 ** min(max(attempts - 1, 0), 5))
    delay += random.uniform(0, delay * 0.2)
    if code == 'resource_http_429':
        delay = max(65, delay)
    if server_delay is not None:
        delay = max(delay, server_delay)
    return 'retry', delay


def now():
    return dt.datetime.now(TZ).isoformat(timespec="seconds")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def resource_key(kind, resource_id):
    return digest([kind, resource_id])


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def monthly_ranges(end_date, start_date=None):
    """Newest first, with an unbounded old-history bucket: no minimum date assumption."""
    end = dt.date.fromisoformat(end_date)
    if start_date is None:
        floor = dt.date(2020, 1, 1)
        if end < floor:
            return [(None, end.isoformat())]
    else:
        floor = dt.date.fromisoformat(start_date)
        if floor > end:
            raise ValueError("start date is later than end date")
    ranges = []
    current_end = end
    while current_end >= floor:
        first = current_end.replace(day=1)
        current_start = max(first, floor)
        ranges.append((current_start.isoformat(), current_end.isoformat()))
        current_end = first - dt.timedelta(days=1)
    if start_date is None:
        ranges.append((None, (floor - dt.timedelta(days=1)).isoformat()))
    return ranges


def split_range(start, end, first_observed=None):
    end_day = dt.date.fromisoformat(end)
    if start:
        start_day = dt.date.fromisoformat(start)
        if start_day >= end_day:
            return None
        middle = start_day + (end_day - start_day) // 2
    else:
        first = dt.date.fromisoformat(first_observed) if first_observed else end_day - dt.timedelta(days=366)
        middle = first + max(dt.timedelta(), end_day - first) // 2
        middle = min(middle, end_day - dt.timedelta(days=1))
    return [(start, middle.isoformat()), ((middle + dt.timedelta(days=1)).isoformat(), end)]


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def __call__(self, event, **fields):
        # Callers use fixed error codes, never request headers or download URLs.
        line = canonical({"at": now(), "event": event, **fields})
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(line, flush=True)


class Store:
    def __init__(self, root, include=("PHOTO", "VIDEO")):
        self.root = Path(root)
        self.include = set(include)
        for name in ("state", "raw/pages", "raw/objects", "raw/annotations", "raw/receipts", "raw/partial", "reports"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.claims_since_retry = 0
        self.db = sqlite3.connect(self.root / "state/state.sqlite", timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(
          run_id TEXT PRIMARY KEY, start_date TEXT, end_date TEXT NOT NULL,
          status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, config_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks(
          task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, start_date TEXT, end_date TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending', cursor TEXT, page_number INTEGER NOT NULL DEFAULT 0,
          records_seen INTEGER NOT NULL DEFAULT 0, first_started_at REAL, first_observed_date TEXT,
          attempts INTEGER NOT NULL DEFAULT 0, resets INTEGER NOT NULL DEFAULT 0,
          last_error TEXT, parent_id TEXT, updated_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS tasks_run_status ON tasks(run_id,status,end_date);
        CREATE TABLE IF NOT EXISTS resources(
          resource_type TEXT NOT NULL, resource_id TEXT NOT NULL, resource_key TEXT NOT NULL UNIQUE,
          metadata_json TEXT NOT NULL, download_url TEXT, status TEXT NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0, retry_after REAL NOT NULL DEFAULT 0,
          owner TEXT, file_path TEXT, sha256 TEXT, byte_size INTEGER, content_type TEXT, source_transport TEXT,
          width INTEGER, height INTEGER, media_json TEXT, file_mtime_ns INTEGER,
          last_error TEXT, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
          annotation_revision INTEGER NOT NULL DEFAULT 0, saved_revision INTEGER NOT NULL DEFAULT -1,
          annotation_dirty INTEGER NOT NULL DEFAULT 1,
          annotation_path TEXT, completed_at TEXT,
          PRIMARY KEY(resource_type,resource_id));
        CREATE INDEX IF NOT EXISTS resources_queue ON resources(status,resource_type,retry_after);
        CREATE INDEX IF NOT EXISTS resources_work_queue ON resources(status,resource_type,retry_after,first_seen_at,resource_id);
        CREATE INDEX IF NOT EXISTS resources_dirty ON resources(annotation_dirty);
        CREATE TABLE IF NOT EXISTS annotations(
          resource_type TEXT NOT NULL, resource_id TEXT NOT NULL, species_id TEXT NOT NULL,
          species_name TEXT NOT NULL, annotation_json TEXT NOT NULL, annotation_hash TEXT NOT NULL,
          box_count INTEGER NOT NULL, complete_boxes INTEGER NOT NULL, unknown_units INTEGER NOT NULL,
          first_seen_at TEXT NOT NULL, changed_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
          source_page TEXT NOT NULL, source_task TEXT NOT NULL,
          PRIMARY KEY(resource_type,resource_id,species_id));
        CREATE TABLE IF NOT EXISTS pages(
          page_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT NOT NULL,
          request_cursor TEXT, next_cursor TEXT, has_more INTEGER NOT NULL,
          record_count INTEGER NOT NULL, raw_path TEXT NOT NULL, received_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS class_catalog(
          species_id TEXT PRIMARY KEY, species_name TEXT NOT NULL, first_seen_at TEXT NOT NULL);
        """)
        self.db.execute("INSERT OR IGNORE INTO meta VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        self.db.commit()
        version = self.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        if int(version) != SCHEMA_VERSION:
            raise RuntimeError("unsupported_database_schema")

    @contextlib.contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def close(self):
        with self.lock:
            self.db.close()

    def create_or_resume_run(self, end_date, start_date, config, new_scan=False):
        with self.transaction() as db:
            existing = db.execute("SELECT * FROM runs WHERE status IN ('running','paused','catalog_complete','catalog_errors') ORDER BY started_at DESC LIMIT 1").fetchone()
            if existing and not new_scan:
                if existing['end_date'] != end_date or existing['start_date'] != start_date:
                    raise RuntimeError("unfinished_run_has_different_date_range")
                db.execute("UPDATE runs SET status='running',finished_at=NULL WHERE run_id=?", (existing['run_id'],))
                db.execute("UPDATE tasks SET status='pending' WHERE run_id=? AND status='running'", (existing['run_id'],))
                return existing['run_id'], True
            if existing:
                incomplete = db.execute("SELECT COUNT(*) FROM tasks WHERE run_id=? AND status NOT IN ('complete','split')", (existing['run_id'],)).fetchone()[0]
                if incomplete:
                    raise RuntimeError("finish_or_resume_existing_catalog_before_new_scan")
                db.execute("UPDATE runs SET status='catalog_complete_downloads_pending' WHERE run_id=?", (existing['run_id'],))
            run_id = dt.datetime.now(TZ).strftime('%Y%m%d_%H%M%S') + '-' + uuid.uuid4().hex[:8]
            db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?)", (run_id, start_date, end_date, 'running', now(), None, canonical(config)))
            for start, end in monthly_ranges(end_date, start_date):
                self._add_task(db, run_id, start, end)
            return run_id, False

    def _add_task(self, db, run_id, start, end, parent=None):
        key = digest([run_id, start, end])
        db.execute("INSERT OR IGNORE INTO tasks(task_id,run_id,start_date,end_date,status,parent_id,updated_at) VALUES(?,?,?,?,?,?,?)", (key, run_id, start, end, 'pending', parent, now()))
        return key

    def next_task(self, run_id):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM tasks WHERE run_id=? AND status IN ('pending','running') ORDER BY end_date DESC,start_date DESC LIMIT 1", (run_id,)).fetchone()
            if row:
                db.execute("UPDATE tasks SET status='running',first_started_at=COALESCE(first_started_at,?),updated_at=? WHERE task_id=?", (time.time(), now(), row['task_id']))
                return dict(db.execute("SELECT * FROM tasks WHERE task_id=?", (row['task_id'],)).fetchone())
        return None

    def task(self, task_id):
        with self.lock:
            row = self.db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return dict(row) if row else None

    def split_task(self, task):
        ranges = split_range(task['start_date'], task['end_date'], task['first_observed_date'])
        if not ranges:
            return False
        with self.transaction() as db:
            db.execute("UPDATE tasks SET status='split',updated_at=? WHERE task_id=?", (now(), task['task_id']))
            for start, end in ranges:
                self._add_task(db, task['run_id'], start, end, task['task_id'])
        return True

    def task_error(self, task_id, code, reset=False):
        with self.transaction() as db:
            if reset:
                db.execute("UPDATE tasks SET status='pending',cursor=NULL,page_number=0,first_started_at=NULL,resets=resets+1,last_error=?,updated_at=? WHERE task_id=?", (code, now(), task_id))
            else:
                db.execute("UPDATE tasks SET status='error',last_error=?,updated_at=? WHERE task_id=?", (code, now(), task_id))

    def ingest_page(self, task, response, raw_path):
        page = response['data']
        records = page['records']
        if page['count'] != len(records) or len(records) > 200:
            raise RuntimeError('invalid_page_count')
        if type(page['hasMore']) is not bool or (page['hasMore'] and not page.get('nextCursor')):
            raise RuntimeError('invalid_page_cursor')
        if not page['hasMore'] and page.get('nextCursor') is not None:
            raise RuntimeError('unexpected_end_cursor')
        stamp = now()
        relative = str(Path(raw_path).relative_to(self.root))
        page_id = digest([task['task_id'], task['cursor'], digest(response)])
        with self.transaction() as db:
            previous_page = db.execute("SELECT next_cursor,has_more FROM pages WHERE page_id=?", (page_id,)).fetchone()
            if previous_page:
                db.execute("UPDATE tasks SET cursor=?,status=?,page_number=MAX(page_number,?),updated_at=? WHERE task_id=?", (previous_page['next_cursor'], 'running' if previous_page['has_more'] else 'complete', task['page_number'] + 1, stamp, task['task_id']))
                return 0
            for record in records:
                self._ingest_record(db, record, task, relative, stamp)
            dates = [r['finalIdentification']['identifiedAt'][:10] for r in records]
            db.execute("INSERT OR IGNORE INTO pages VALUES(?,?,?,?,?,?,?,?,?)", (page_id, task['run_id'], task['task_id'], task['cursor'], page.get('nextCursor'), int(page['hasMore']), len(records), relative, stamp))
            # Cursor advancement and all rows commit as a single transaction.
            db.execute("UPDATE tasks SET cursor=?,page_number=page_number+1,records_seen=records_seen+?,status=?,first_observed_date=COALESCE(first_observed_date,?),updated_at=? WHERE task_id=?", (page.get('nextCursor'), len(records), 'running' if page['hasMore'] else 'complete', min(dates) if dates else None, stamp, task['task_id']))
        return len(records)

    def _ingest_record(self, db, record, task, source_page, stamp):
        kind, rid, species = record['resourceType'], record['resourceId'], record['speciesId']
        if kind not in ('PHOTO', 'VIDEO') or not all(isinstance(x, str) and x for x in (rid, species)):
            raise RuntimeError('invalid_record_identity')
        final = record['finalIdentification']
        boxes = final['boxes']
        if final['boxCount'] != len(boxes):
            raise RuntimeError('invalid_box_count')
        identified = dt.datetime.strptime(final['identifiedAt'], '%Y-%m-%d %H:%M:%S').date().isoformat()
        if identified > task['end_date'] or (task['start_date'] and identified < task['start_date']):
            raise RuntimeError('server_returned_date_outside_query')
        metadata = {k: v for k, v in record.items() if k not in ('speciesId', 'speciesName', 'finalIdentification', 'ai', 'downloadUrl')}
        url = record.get('downloadUrl')
        desired = 'pending' if kind in self.include and url else ('no_url' if kind in self.include else 'not_requested')
        prior = db.execute("SELECT download_url,status,metadata_json FROM resources WHERE resource_type=? AND resource_id=?", (kind, rid)).fetchone()
        db.execute("INSERT OR IGNORE INTO resources(resource_type,resource_id,resource_key,metadata_json,download_url,status,first_seen_at,last_seen_at) VALUES(?,?,?,?,?,?,?,?)", (kind, rid, resource_key(kind, rid), canonical(metadata), url, desired, stamp, stamp))
        if prior:
            status = prior['status']
            if status == 'not_requested' and kind in self.include:
                status = desired
            elif status == 'no_url' and url:
                status = 'pending'
            elif status in ('failed', 'blocked_url', 'unavailable', 'refresh_required', 'blocked_transport') and url and prior['download_url'] != url:
                status = 'pending'
            db.execute("UPDATE resources SET metadata_json=?,download_url=?,status=?,last_seen_at=? WHERE resource_type=? AND resource_id=?", (canonical(metadata), url, status, stamp, kind, rid))
            if prior['metadata_json'] != canonical(metadata) or prior['status'] != status:
                db.execute("UPDATE resources SET annotation_revision=annotation_revision+1,annotation_dirty=1 WHERE resource_type=? AND resource_id=?", (kind, rid))
            if status == 'pending' and status != prior['status']:
                db.execute("UPDATE resources SET attempts=0,retry_after=0,last_error=NULL WHERE resource_type=? AND resource_id=?", (kind, rid))
        annotation = {'speciesId': species, 'speciesName': record['speciesName'], 'finalIdentification': final, 'ai': record['ai']}
        hashed = digest(annotation)
        previous = db.execute("SELECT annotation_hash FROM annotations WHERE resource_type=? AND resource_id=? AND species_id=?", (kind, rid, species)).fetchone()
        if previous is None:
            db.execute("INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (kind, rid, species, record['speciesName'], canonical(annotation), hashed, len(boxes), sum(b.get('complete') is True for b in boxes), sum(b.get('coordinateUnit') == 'UNKNOWN' for b in boxes), stamp, stamp, stamp, source_page, task['task_id']))
        elif previous['annotation_hash'] != hashed:
            db.execute("UPDATE annotations SET species_name=?,annotation_json=?,annotation_hash=?,box_count=?,complete_boxes=?,unknown_units=?,changed_at=?,last_seen_at=?,source_page=?,source_task=? WHERE resource_type=? AND resource_id=? AND species_id=?", (record['speciesName'], canonical(annotation), hashed, len(boxes), sum(b.get('complete') is True for b in boxes), sum(b.get('coordinateUnit') == 'UNKNOWN' for b in boxes), stamp, stamp, source_page, task['task_id'], kind, rid, species))
        else:
            db.execute("UPDATE annotations SET last_seen_at=? WHERE resource_type=? AND resource_id=? AND species_id=?", (stamp, kind, rid, species))
        if previous is None or previous['annotation_hash'] != hashed:
            db.execute("UPDATE resources SET annotation_revision=annotation_revision+1,annotation_dirty=1 WHERE resource_type=? AND resource_id=?", (kind, rid))
        db.execute("INSERT INTO class_catalog VALUES(?,?,?) ON CONFLICT(species_id) DO UPDATE SET species_name=excluded.species_name", (species, record['speciesName'], stamp))

    def recover_claims(self):
        # The exclusive process lock is held before this is called.
        with self.transaction() as db:
            count = db.execute("UPDATE resources SET status='retry',owner=NULL,retry_after=0,last_error='recovered_interrupted_worker' WHERE status='downloading'").rowcount
        return count

    def reconcile_completed(self):
        """Cheap restart check; rehash files whose recorded mtime has changed."""
        bad = 0
        last_key = ''
        while True:
            with self.lock:
                rows = [dict(r) for r in self.db.execute("SELECT * FROM resources WHERE status='done' AND resource_key>? ORDER BY resource_key LIMIT 500", (last_key,))]
            if not rows:
                return bad
            for row in rows:
                last_key = row['resource_key']
                path = self.root / row['file_path']
                try:
                    stat = path.stat()
                    valid = stat.st_size == row['byte_size']
                    if valid and row['file_mtime_ns'] != stat.st_mtime_ns:
                        valid = sha256_file(path) == row['sha256']
                except OSError:
                    valid = False
                if not valid:
                    with self.transaction() as db:
                        db.execute("UPDATE resources SET status='retry',attempts=0,retry_after=0,last_error='local_file_missing_or_changed',annotation_revision=annotation_revision+1,annotation_dirty=1 WHERE resource_type=? AND resource_id=?", (row['resource_type'], row['resource_id']))
                    bad += 1

    def retry_failed(self):
        with self.transaction() as db:
            return db.execute("UPDATE resources SET status='retry',attempts=0,retry_after=0,last_error=NULL,annotation_revision=annotation_revision+1,annotation_dirty=1 WHERE status='failed' AND download_url IS NOT NULL").rowcount

    def settle_held_failures(self):
        """Migrate the old hot retry queue; only run after taking the process lock."""
        counts = {}
        with self.transaction() as db:
            for codes, state in [
                (('resource_http_404', 'resource_http_410'), 'unavailable'),
                (('resource_http_401', 'resource_http_403'), 'refresh_required'),
            ]:
                counts[state] = db.execute("UPDATE resources SET status=?,owner=NULL,retry_after=0,annotation_revision=annotation_revision+1,annotation_dirty=1 WHERE status IN ('retry','failed','pending') AND last_error IN (?,?)", (state, *codes)).rowcount
            db.execute("INSERT INTO meta VALUES('retry_policy_version','2') ON CONFLICT(key) DO UPDATE SET value=excluded.value")
        return counts

    def recheck_unavailable(self):
        with self.transaction() as db:
            return db.execute("UPDATE resources SET status='pending',attempts=0,retry_after=0,last_error=NULL,annotation_revision=annotation_revision+1,annotation_dirty=1 WHERE status='unavailable' AND download_url IS NOT NULL").rowcount

    def claim(self, preferred, owner):
        with self.transaction() as db:
            # Reserve at most one preferred slot per nine fresh jobs for due
            # transient retries. If no fresh work exists, retries drain normally.
            states = ('retry', 'pending') if self.claims_since_retry >= 9 else ('pending', 'retry')
            for state in states:
                for kind in preferred:
                    if kind not in self.include:
                        continue
                    row = db.execute("SELECT * FROM resources WHERE resource_type=? AND status=? AND retry_after<=? ORDER BY retry_after,first_seen_at,resource_id LIMIT 1", (kind, state, time.time())).fetchone()
                    if row:
                        changed = db.execute("UPDATE resources SET status='downloading',owner=?,attempts=attempts+1 WHERE resource_type=? AND resource_id=? AND status=?", (owner, kind, row['resource_id'], state)).rowcount
                        if changed:
                            self.claims_since_retry = 0 if state == 'retry' else self.claims_since_retry + 1
                            return dict(db.execute("SELECT * FROM resources WHERE resource_type=? AND resource_id=?", (kind, row['resource_id'])).fetchone())
        return None

    def release_for_disk(self, row):
        with self.transaction() as db:
            db.execute("UPDATE resources SET status='retry',owner=NULL,retry_after=?,attempts=MAX(0,attempts-1),last_error='disk_reserve' WHERE resource_type=? AND resource_id=?", (time.time() + 60, row['resource_type'], row['resource_id']))

    def failed(self, row, code, permanent=False, maximum=5, server_delay=None):
        status, delay = retry_policy(code, row['attempts'], permanent, maximum, server_delay)
        with self.transaction() as db:
            current = db.execute("SELECT download_url FROM resources WHERE resource_type=? AND resource_id=?", (row['resource_type'], row['resource_id'])).fetchone()
            if current and current['download_url'] and current['download_url'] != row['download_url']:
                # An old in-flight URL must not block a newly refreshed URL.
                db.execute("UPDATE resources SET status='pending',owner=NULL,attempts=0,retry_after=0,last_error=NULL,annotation_revision=annotation_revision+1,annotation_dirty=1 WHERE resource_type=? AND resource_id=?", (row['resource_type'], row['resource_id']))
                return 'pending'
            db.execute("UPDATE resources SET status=?,owner=NULL,last_error=?,retry_after=?,annotation_revision=annotation_revision+1,annotation_dirty=1 WHERE resource_type=? AND resource_id=?", (status, code, time.time() + delay if status == 'retry' else 0, row['resource_type'], row['resource_id']))
        return status

    def held_count(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM resources WHERE status IN ({})".format(','.join('?' for _ in HELD_STATUSES)), HELD_STATUSES).fetchone()[0]

    def completed(self, row, receipt):
        with self.transaction() as db:
            db.execute("UPDATE resources SET status='done',owner=NULL,file_path=?,sha256=?,byte_size=?,content_type=?,source_transport=?,width=?,height=?,media_json=?,file_mtime_ns=?,completed_at=?,last_error=NULL,retry_after=0,annotation_revision=annotation_revision+1,annotation_dirty=1 WHERE resource_type=? AND resource_id=?", (receipt['file_path'], receipt['sha256'], receipt['byte_size'], receipt['content_type'], receipt.get('transport', 'unknown'), receipt.get('width'), receipt.get('height'), canonical(receipt.get('media', {})), receipt.get('file_mtime_ns'), receipt['completed_at'], row['resource_type'], row['resource_id']))

    def pending_count(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM resources WHERE status IN ('pending','retry','downloading')").fetchone()[0]

    def export_annotations(self, limit=100):
        with self.lock:
            rows = [dict(r) for r in self.db.execute("SELECT * FROM resources WHERE annotation_dirty=1 LIMIT ?", (limit,))]
        for row in rows:
            with self.lock:
                annotations = [dict(r) for r in self.db.execute("SELECT * FROM annotations WHERE resource_type=? AND resource_id=? ORDER BY species_id", (row['resource_type'], row['resource_id']))]
            payload = {
                'schemaVersion': 1,
                'resource': json.loads(row['metadata_json']),
                'file': {'status': row['status'], 'path': row['file_path'], 'sha256': row['sha256'], 'byteSize': row['byte_size'], 'width': row['width'], 'height': row['height']},
                'collection': {'guarantee': 'observed_records_from_live_queries_not_a_server_snapshot', 'trainingExported': False},
                'annotations': [json.loads(a['annotation_json']) for a in annotations],
                'provenance': [{'speciesId': a['species_id'], 'annotationHash': a['annotation_hash'], 'firstSeenAt': a['first_seen_at'], 'changedAt': a['changed_at'], 'sourcePage': a['source_page']} for a in annotations],
            }
            hashed = digest(payload)
            base = self.root / 'raw/annotations' / row['resource_type'] / row['resource_key'][:2] / row['resource_key']
            version_path = base / (hashed + '.json')
            if not version_path.exists():
                atomic_json(version_path, payload)
            current = base / 'current.json'
            atomic_json(current, payload)
            with self.transaction() as db:
                db.execute("UPDATE resources SET saved_revision=?,annotation_path=?,annotation_dirty=0 WHERE resource_type=? AND resource_id=? AND annotation_revision=?", (row['annotation_revision'], str(current.relative_to(self.root)), row['resource_type'], row['resource_id'], row['annotation_revision']))
        return len(rows)

    def finish_catalog(self, run_id):
        with self.transaction() as db:
            counts = dict(db.execute("SELECT status,COUNT(*) FROM tasks WHERE run_id=? GROUP BY status", (run_id,)))
            if counts.get('pending') or counts.get('running'):
                raise RuntimeError('catalog_has_unfinished_tasks')
            status = 'catalog_errors' if counts.get('error', 0) else 'catalog_complete'
            db.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))
            if status == 'catalog_complete':
                run = db.execute("SELECT start_date,end_date FROM runs WHERE run_id=?", (run_id,)).fetchone()
                previous = db.execute("SELECT value FROM meta WHERE key='catalog_end_date'").fetchone()
                # A manually selected bounded range cannot establish complete
                # historical coverage or jump a gap in the existing watermark.
                contiguous = run['start_date'] is None or (previous and run['start_date'] <= (dt.date.fromisoformat(previous[0]) + dt.timedelta(days=1)).isoformat())
                if contiguous and (not previous or run['end_date'] > previous[0]):
                    db.execute("INSERT INTO meta VALUES('catalog_end_date',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (run['end_date'],))
        return status

    def summary(self, run_id=None):
        with self.lock:
            rows = self.db.execute("SELECT resource_type,status,COUNT(*) AS n,COALESCE(SUM(byte_size),0) AS bytes FROM resources GROUP BY resource_type,status").fetchall()
            resources = [{'type': r[0], 'status': r[1], 'count': r[2], 'bytes': r[3]} for r in rows]
            annotations = self.db.execute("SELECT COUNT(*),COALESCE(SUM(box_count),0),COALESCE(SUM(complete_boxes),0),COALESCE(SUM(unknown_units),0) FROM annotations").fetchone()
            latest = self.db.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1" if run_id is None else "SELECT * FROM runs WHERE run_id=?", () if run_id is None else (run_id,)).fetchone()
            tasks = [] if not latest else [{'status': r[0], 'count': r[1], 'pages': r[2], 'records_seen': r[3]} for r in self.db.execute("SELECT status,COUNT(*),SUM(page_number),SUM(records_seen) FROM tasks WHERE run_id=? GROUP BY status", (latest['run_id'],))]
            dirty = self.db.execute("SELECT COUNT(*) FROM resources WHERE annotation_dirty=1").fetchone()[0]
            failures = [{'error': r[0], 'count': r[1]} for r in self.db.execute("SELECT last_error,COUNT(*) FROM resources WHERE status IN ({}) GROUP BY last_error ORDER BY COUNT(*) DESC LIMIT 10".format(','.join('?' for _ in HELD_STATUSES)), HELD_STATUSES)]
            transports = {r[0]: r[1] for r in self.db.execute("SELECT source_transport,COUNT(*) FROM resources WHERE status='done' GROUP BY source_transport")}
            metadata = dict(self.db.execute("SELECT key,value FROM meta WHERE key IN ('initial_requested_end_date','catalog_end_date')"))
        usage = shutil.disk_usage(self.root)
        return {'at': now(), 'hostname': socket.gethostname(), 'root': str(self.root), 'run': dict(latest) if latest else None, 'coverage': metadata, 'tasks': tasks, 'resources': resources, 'completed_by_transport': transports, 'annotation_records': annotations[0], 'final_boxes': annotations[1], 'api_complete_boxes': annotations[2], 'unknown_coordinate_boxes': annotations[3], 'annotation_files_pending': dirty, 'failures': failures, 'disk_free_gib': round(usage.free / 1024 ** 3, 2), 'yolo_training_exports': 0, 'source_snapshot_guaranteed': False}


class ApiError(Exception):
    def __init__(self, http_status, code):
        self.http_status, self.code = http_status, code
        super().__init__('api_http_{}_code_{}'.format(http_status, code))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Api:
    def __init__(self, base, app_id, secret, interval, stop, log):
        self.base, self.app_id, self.secret = base.rstrip('/'), app_id, secret
        self.interval, self.stop, self.log = interval, stop, log
        self.opener = urllib.request.build_opener(NoRedirect())
        self.last_request = -1e20

    def query(self, body):
        for attempt in range(4):
            delay = max(0, self.interval - (time.monotonic() - self.last_request))
            if self.stop.wait(delay):
                raise InterruptedError()
            ts = str(time.time_ns() // 1_000_000)
            message = (self.app_id + '\n' + ts + '\n' + QUERY_PATH).encode('utf-8')
            signature = base64.b64encode(hmac.new(self.secret.encode('utf-8'), message, hashlib.sha256).digest()).decode('ascii')
            headers = {'X-App-Id': self.app_id, 'X-Timestamp': ts, 'X-Signature': signature, 'X-Trace-Id': 'sync-' + uuid.uuid4().hex, 'Content-Type': 'application/json', 'Accept': 'application/json'}
            request = urllib.request.Request(self.base + QUERY_PATH, data=canonical(body).encode('utf-8'), headers=headers, method='POST')
            self.last_request = time.monotonic()
            try:
                try:
                    response = self.opener.open(request, timeout=60)
                except urllib.error.HTTPError as exc:
                    response = exc
                with response:
                    status = response.code
                    raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise ApiError(status, 'response_over_4mib')
                payload = json.loads(raw)
                code = payload.get('code')
                if status == 200 and code == 0:
                    return payload
                if status == 429:
                    self.log('api_throttled', attempt=attempt + 1, wait_seconds=65)
                    if attempt == 3:
                        raise ApiError(status, code)
                    if self.stop.wait(65):
                        raise InterruptedError()
                    continue
                if status not in (500, 503) or attempt == 3:
                    raise ApiError(status, code)
                self.log('api_retry', http_status=status, attempt=attempt + 1)
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                if attempt == 3:
                    raise ApiError(0, type(exc).__name__) from None
                self.log('api_retry', http_status=0, error_type=type(exc).__name__, attempt=attempt + 1)
            if self.stop.wait(2 ** attempt):
                raise InterruptedError()
        raise ApiError(0, 'retry_exhausted')


class DownloadError(Exception):
    def __init__(self, code, permanent=False, retry_after=None):
        self.code, self.permanent, self.retry_after = code, permanent, retry_after
        super().__init__(code)


class Downloader:
    def __init__(self, store, private_ip, reserve_gib, stop, log, max_photo_mib=256, max_video_gib=16):
        self.store, self.root, self.private_ip = store, store.root, private_ip
        self.reserve = int(reserve_gib * 1024 ** 3)
        self.stop, self.log = stop, log
        self.limits = {'PHOTO': int(max_photo_mib * 1024 ** 2), 'VIDEO': int(max_video_gib * 1024 ** 3)}
        self.ffprobe = shutil.which('ffprobe')

    def receipt_path(self, row):
        return self.root / 'raw/receipts' / row['resource_key'][:2] / (row['resource_key'] + '.json')

    def recover_receipt(self, row):
        path = self.receipt_path(row)
        if not path.is_file():
            return None
        try:
            receipt = json.loads(path.read_text())
            target = self.root / receipt['file_path']
            if receipt['resource_type'] != row['resource_type'] or receipt['resource_id'] != row['resource_id']:
                return None
            if target.is_file() and target.stat().st_size == receipt['byte_size'] and sha256_file(target) == receipt['sha256']:
                self.store.completed(row, receipt)
                return receipt
        except (OSError, ValueError, KeyError):
            pass
        return None

    def command(self, row, temp):
        parsed = urllib.parse.urlsplit(row['download_url'] or '')
        if parsed.scheme != 'https' or parsed.hostname != RESOURCE_HOST or parsed.port not in (None, 443) or parsed.username is not None:
            raise DownloadError('url_not_allowed_private_host', True)
        fields = '%{http_code}\t%{remote_ip}\t%{size_download}\t%{content_type}\t%{ssl_verify_result}'
        command = ['curl', '--config', '-', '--silent', '--show-error', '--fail', '--globoff', '--path-as-is', '--proto', '=https', '--connect-timeout', '10', '--max-time', '900' if row['resource_type'] == 'VIDEO' else '120', '--noproxy', '*', '--resolve', RESOURCE_HOST + ':443:' + self.private_ip, '--max-filesize', str(self.limits[row['resource_type']]), '--dump-header', str(temp) + '.headers', '--output', str(temp), '--write-out', fields]
        # No --location: a redirect must never silently escape the private route.
        config = 'url = ' + json.dumps(row['download_url'], ensure_ascii=False) + '\n'
        return command, config

    def media_info(self, row, path, content_type):
        if row['resource_type'] == 'PHOTO':
            try:
                from PIL import Image
                with Image.open(path) as image:
                    media = {'format': image.format, 'exif_orientation': image.getexif().get(274, 1)}
                    width, height = image.size
                    image.verify()
                if width <= 0 or height <= 0:
                    raise ValueError()
                ext = {'JPEG': '.jpg', 'PNG': '.png', 'TIFF': '.tif', 'WEBP': '.webp', 'BMP': '.bmp', 'GIF': '.gif'}.get(media['format'], '.image')
                return ext, width, height, media
            except Exception:
                raise DownloadError('invalid_image', True) from None
        if self.ffprobe:
            result = subprocess.run([self.ffprobe, '-v', 'error', '-show_entries', 'format=format_name,duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames', '-of', 'json', str(path)], capture_output=True, text=True, timeout=45)
            if result.returncode:
                raise DownloadError('invalid_video_container', True)
            try:
                info = json.loads(result.stdout)
                video = next(s for s in info.get('streams', []) if s.get('codec_type') == 'video')
            except (ValueError, StopIteration):
                raise DownloadError('no_video_stream', True) from None
            formats = info.get('format', {}).get('format_name', '')
            ext = '.avi' if 'avi' in formats else ('.mkv' if 'matroska' in formats else '.mp4')
            return ext, video.get('width'), video.get('height'), {'container_verified': True, 'probe': info, 'frames_decoded': False}
        if content_type.startswith('text/') or content_type in ('application/json', 'application/xml'):
            raise DownloadError('invalid_video_content_type', True)
        return '.video', None, None, {'container_verified': False, 'reason': 'ffprobe_unavailable', 'frames_decoded': False}

    def download(self, row):
        receipt = self.recover_receipt(row)
        if receipt:
            return receipt, True
        temp = self.root / 'raw/partial' / (row['resource_key'] + '.part')
        command, config = self.command(row, temp)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            output, _ = process.communicate(config, timeout=1)
        except subprocess.TimeoutExpired:
            while True:
                reason = None
                if self.stop.is_set():
                    reason = 'interrupted'
                elif shutil.disk_usage(self.root).free < self.reserve:
                    reason = 'disk_reserve'
                elif temp.exists() and temp.stat().st_size > self.limits[row['resource_type']]:
                    reason = 'file_size_limit'
                if reason:
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                    raise DownloadError(reason, reason == 'file_size_limit')
                try:
                    output, _ = process.communicate(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        parts = output.strip().split('\t')
        server_delay = None
        header_path = Path(str(temp) + '.headers')
        if header_path.is_file():
            # Only parse Retry-After; never log headers (which may contain tokens).
            headers = header_path.read_bytes()[:65536].decode('iso-8859-1', 'replace')
            for line in headers.splitlines():
                if line.lower().startswith('retry-after:'):
                    server_delay = retry_after_seconds(line.split(':', 1)[1].strip())
            header_path.unlink()
        if len(parts) == 5 and parts[0] != '000' and (parts[1] != self.private_ip or parts[4] != '0'):
            raise DownloadError('private_route_or_tls_mismatch', True)
        if process.returncode:
            if len(parts) == 5 and parts[0] not in ('000', '200'):
                raise DownloadError('resource_http_' + parts[0], retry_after=server_delay)
            raise DownloadError('curl_exit_' + str(process.returncode), process.returncode == 63)
        if len(parts) != 5:
            raise DownloadError('invalid_curl_metrics')
        status, remote_ip, byte_count, content_type, tls = parts
        if status != '200':
            raise DownloadError('resource_http_' + status, status.startswith('3'), server_delay)
        if remote_ip != self.private_ip or tls != '0':
            raise DownloadError('private_route_or_tls_mismatch', True)
        if not temp.is_file() or temp.stat().st_size <= 0 or temp.stat().st_size != int(float(byte_count)):
            raise DownloadError('incomplete_download')
        ext, width, height, media = self.media_info(row, temp, content_type)
        hashed = sha256_file(temp)
        target = self.root / 'raw/objects' / hashed[:2] / (hashed + ext)
        target.parent.mkdir(parents=True, exist_ok=True)
        with temp.open('rb') as handle:
            os.fsync(handle.fileno())
        # Never overwrite an existing content-addressed file. Concurrent equal
        # content links are harmless; an existing object is independently checked.
        try:
            os.link(temp, target)
        except FileExistsError:
            if target.stat().st_size != temp.stat().st_size or sha256_file(target) != hashed:
                raise DownloadError('existing_object_corrupt', True)
        fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        temp.unlink()
        receipt = {'schemaVersion': 1, 'resource_type': row['resource_type'], 'resource_id': row['resource_id'], 'source_url_sha256': hashlib.sha256(row['download_url'].encode('utf-8')).hexdigest(), 'source_resource_metadata_sha256': digest(json.loads(row['metadata_json'])), 'file_path': str(target.relative_to(self.root)), 'sha256': hashed, 'byte_size': target.stat().st_size, 'file_mtime_ns': target.stat().st_mtime_ns, 'content_type': content_type, 'width': width, 'height': height, 'media': media, 'transport': 'private_https', 'remote_ip': remote_ip, 'tls_verified': True, 'completed_at': now()}
        # Receipt precedes the DB completion flag, permitting crash recovery.
        atomic_json(self.receipt_path(row), receipt)
        self.store.completed(row, receipt)
        return receipt, False


def save_page(root, task, payload):
    directory = Path(root) / 'raw/pages' / task['run_id'] / task['task_id']
    path = directory / ('{:06d}-{}.json.gz'.format(task['page_number'] + 1, digest(payload)))
    if not path.exists():
        atomic_bytes(path, gzip.compress((canonical(payload) + '\n').encode('utf-8'), mtime=0))
    return path


class Runner:
    def __init__(self, args, secret):
        self.args, self.secret = args, secret
        self.root = Path(args.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.catalog_done = threading.Event()
        self.failure = None
        self.log = Log(self.root / 'reports/events.jsonl')
        self.store = Store(self.root, args.include)
        self.api = Api(args.api_base, args.app_id, secret, args.api_interval, self.stop, self.log)
        self.downloader = Downloader(self.store, args.private_ip, args.reserve_gib, self.stop, self.log, args.max_photo_mib, args.max_video_gib)
        self.run_id = None
        self.started_downloads = 0
        self.completed_downloads = 0
        self.counter_lock = threading.Lock()
        self.start_time = time.monotonic()
        self.code_sha256 = sha256_file(__file__)

    def collector(self):
        try:
            while not self.stop.is_set():
                task = self.store.next_task(self.run_id)
                if not task:
                    state = self.store.finish_catalog(self.run_id)
                    self.log('catalog_finished', status=state)
                    return
                if task['page_number'] >= self.args.split_pages and self.store.split_task(task):
                    self.log('task_split', task_id=task['task_id'], start=task['start_date'], end=task['end_date'])
                    continue
                if task['first_started_at'] and time.time() - task['first_started_at'] > 20 * 3600:
                    if not self.store.split_task(task):
                        if task['page_number'] < 10000 and task['resets'] < 2:
                            self.store.task_error(task['task_id'], 'old_cursor_after_pause', reset=True)
                        else:
                            self.store.task_error(task['task_id'], 'single_day_exceeds_cursor_budget')
                    continue
                body = {'pageSize': 200, 'aiConsistency': -1, 'identifiedDateEnd': task['end_date']}
                if task['start_date']:
                    body['identifiedDateStart'] = task['start_date']
                if task['cursor']:
                    body['cursor'] = task['cursor']
                try:
                    payload = self.api.query(body)
                    page_path = save_page(self.root, task, payload)
                    count = self.store.ingest_page(task, payload, page_path)
                    self.log('page_saved', task_id=task['task_id'], start=task['start_date'], end=task['end_date'], page=task['page_number'] + 1, records=count, has_more=payload['data']['hasMore'])
                except ApiError as exc:
                    code = str(exc)
                    if exc.code in (40002, 40004) and task['resets'] < 2:
                        if not self.store.split_task(task):
                            self.store.task_error(task['task_id'], code, reset=True)
                        self.log('cursor_restart', task_id=task['task_id'], error=code)
                        continue
                    self.store.task_error(task['task_id'], code)
                    self.log('task_failed', task_id=task['task_id'], error=code)
                    if exc.http_status in (401, 429) or exc.http_status == 0:
                        self.failure = code
                        self.stop.set()
                        return
                except InterruptedError:
                    return
                except Exception as exc:
                    self.store.task_error(task['task_id'], 'ingest_' + type(exc).__name__)
                    self.failure = 'collector_' + type(exc).__name__
                    self.log('collector_failed', error=self.failure)
                    self.stop.set()
                    return
        finally:
            self.catalog_done.set()

    def worker(self, number):
        preferred = ('VIDEO', 'PHOTO') if number == 0 else ('PHOTO', 'VIDEO')
        owner = '{}:{}'.format(os.getpid(), number)
        while not self.stop.is_set():
            if shutil.disk_usage(self.root).free < self.downloader.reserve:
                if self.stop.wait(10):
                    break
                continue
            if self.args.max_downloads:
                with self.counter_lock:
                    if self.started_downloads >= self.args.max_downloads:
                        return
            row = self.store.claim(preferred, owner)
            if row is None:
                if self.catalog_done.is_set() and self.store.pending_count() == 0:
                    return
                self.stop.wait(1)
                continue
            with self.counter_lock:
                if self.args.max_downloads and self.started_downloads >= self.args.max_downloads:
                    self.store.release_for_disk(row)
                    return
                self.started_downloads += 1
            try:
                receipt, reused = self.downloader.download(row)
                with self.counter_lock:
                    self.completed_downloads += 1
                    completed = self.completed_downloads
                if completed <= 5 or completed % 100 == 0:
                    self.log('download_progress', completed_this_process=completed, resource_type=row['resource_type'], resource_key=row['resource_key'], bytes=receipt['byte_size'], private_ip=self.args.private_ip, reused_receipt=reused)
            except DownloadError as exc:
                if exc.code in ('disk_reserve', 'interrupted'):
                    self.store.release_for_disk(row)
                    self.log('download_deferred', resource_key=row['resource_key'], error=exc.code)
                else:
                    state = self.store.failed(row, exc.code, exc.permanent, self.args.max_attempts, exc.retry_after)
                    self.log('download_failed', resource_type=row['resource_type'], resource_key=row['resource_key'], error=exc.code, status=state)
            except Exception as exc:
                code = 'worker_' + type(exc).__name__
                self.store.failed(row, code, maximum=self.args.max_attempts)
                self.log('download_failed', resource_key=row['resource_key'], error=code)

    def write_summary(self):
        summary = self.store.summary(self.run_id)
        summary.update({'pid': os.getpid(), 'private_resource_ip': self.args.private_ip, 'download_workers': self.args.workers, 'elapsed_seconds_this_process': round(time.monotonic() - self.start_time, 1), 'completed_downloads_this_process': self.completed_downloads, 'fatal_error': self.failure, 'disk_downloads_paused': shutil.disk_usage(self.root).free < self.downloader.reserve, 'code_version': VERSION, 'code_sha256': self.code_sha256})
        atomic_json(self.root / 'reports/status.json', summary)
        return summary

    def run(self, ready_fd=None):
        config = {k: v for k, v in vars(self.args).items() if k not in ('func',)}
        config['code_sha256'] = self.code_sha256
        self.run_id, resumed = self.store.create_or_resume_run(self.args.end_date, self.args.start_date, config, self.args.new_scan)
        with self.store.transaction() as db:
            db.execute("INSERT OR IGNORE INTO meta VALUES('initial_requested_end_date',?)", (self.args.end_date,))
            initial_cutoff = db.execute("SELECT value FROM meta WHERE key='initial_requested_end_date'").fetchone()[0]
        atomic_json(self.root / 'state/sync_policy.json', {
            'initialRequestedEndDate': initial_cutoff,
            'currentRequestedEndDate': self.args.end_date,
            'timezone': 'Asia/Shanghai',
            'dateBasis': 'finalIdentification.identifiedAt_first_completion',
            'nextRunMode': 'incremental_after_completed_catalog',
            'overlapDays': self.args.overlap_days,
            'resourceTypes': self.args.include,
            'resourceHost': RESOURCE_HOST,
            'privateResourceIp': self.args.private_ip,
            'sourceSnapshotGuaranteed': False,
            'retryPolicyVersion': 2,
            'notFoundPolicy': 'hold_until_url_changes_or_explicit_recheck',
        })
        recovered = self.store.recover_claims()
        settled = self.store.settle_held_failures()
        damaged = self.store.reconcile_completed()
        retried = self.store.retry_failed() if self.args.retry_failed else 0
        rechecked = self.store.recheck_unavailable() if self.args.recheck_unavailable else 0
        atomic_json(self.root / 'state/process.json', {'pid': os.getpid(), 'run_id': self.run_id, 'started_at': now(), 'hostname': socket.gethostname()})
        self.log('run_started', run_id=self.run_id, resumed=resumed, recovered_claims=recovered, settled_held_failures=settled, unavailable_explicitly_rechecked=rechecked, files_needing_repair=damaged, failed_tasks_retried=retried, include=self.args.include, private_ip=self.args.private_ip)
        self.write_summary()
        if ready_fd is not None:
            os.write(ready_fd, (canonical({'pid': os.getpid(), 'run_id': self.run_id, 'root': str(self.root)}) + '\n').encode())
            os.close(ready_fd)
        collector = threading.Thread(target=self.collector, name='collector', daemon=True)
        collector.start()
        workers = [threading.Thread(target=self.worker, args=(n,), name='download-{}'.format(n), daemon=True) for n in range(self.args.workers)]
        for worker in workers:
            worker.start()
        report_at = 0
        try:
            while not self.stop.is_set():
                self.store.export_annotations(100)
                if time.monotonic() >= report_at:
                    summary = self.write_summary()
                    self.log('heartbeat', completed_this_process=self.completed_downloads, annotation_records=summary['annotation_records'], disk_free_gib=summary['disk_free_gib'], catalog_done=self.catalog_done.is_set())
                    report_at = time.monotonic() + self.args.report_interval
                if self.args.max_seconds and time.monotonic() - self.start_time >= self.args.max_seconds:
                    self.log('bounded_run_limit', kind='seconds')
                    self.stop.set()
                    break
                if self.args.max_downloads and self.started_downloads >= self.args.max_downloads and not any(w.is_alive() for w in workers):
                    self.log('bounded_run_limit', kind='downloads')
                    self.stop.set()
                    break
                if self.catalog_done.is_set() and not any(w.is_alive() for w in workers):
                    break
                self.stop.wait(0.5)
        finally:
            self.stop.set()
            collector.join(timeout=70)
            for worker in workers:
                worker.join(timeout=70)
            while self.store.export_annotations(100):
                pass
            with self.store.transaction() as db:
                if not self.catalog_done.is_set() or not all(not w.is_alive() for w in workers):
                    status = 'paused'
                else:
                    counts = dict(db.execute("SELECT status,COUNT(*) FROM tasks WHERE run_id=? GROUP BY status", (self.run_id,)))
                    pending = db.execute("SELECT COUNT(*) FROM resources WHERE status IN ('pending','retry','downloading')").fetchone()[0]
                    failed = db.execute("SELECT COUNT(*) FROM resources WHERE status IN ({})".format(','.join('?' for _ in HELD_STATUSES)), HELD_STATUSES).fetchone()[0]
                    status = 'paused' if pending or counts.get('pending') or counts.get('running') else ('completed_with_errors' if failed or counts.get('error') else 'complete')
                db.execute("UPDATE runs SET status=?,finished_at=? WHERE run_id=?", (status, now(), self.run_id))
            summary = self.write_summary()
            self.log('run_stopped', status=summary['run']['status'], completed_this_process=self.completed_downloads, fatal_error=self.failure)
        return 1 if self.failure else 0


def lock_process(root):
    directory = Path(root) / 'state'
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / 'process.lock').open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('another_downloader_is_running') from None
    return handle


def run_command(args):
    if not ipaddress.ip_address(args.private_ip).is_private:
        raise RuntimeError('resource_ip_must_be_private')
    if args.workers < 1 or args.api_interval < 6.1 or args.reserve_gib < 1 or args.overlap_days < 1:
        raise RuntimeError('invalid_worker_rate_or_disk_configuration')
    if not shutil.which('curl'):
        raise RuntimeError('curl_required')
    if 'PHOTO' in args.include:
        import PIL  # noqa: F401
    lock = lock_process(args.root)
    # Resolve incremental dates before asking for a credential. Reuse an
    # unfinished scan's exact range; otherwise replay a persisted overlap window.
    database = Path(args.root) / 'state/state.sqlite'
    if database.exists():
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            active = connection.execute("SELECT * FROM runs WHERE status IN ('running','paused','catalog_complete','catalog_errors') ORDER BY started_at DESC LIMIT 1").fetchone()
            watermark = connection.execute("SELECT value FROM meta WHERE key='catalog_end_date'").fetchone()
            if args.mode == 'auto' and not args.new_scan and active and active['end_date'] == args.end_date and args.start_date is None:
                args.start_date = active['start_date']
            elif args.start_date is None and args.mode != 'history':
                if watermark:
                    args.start_date = (dt.date.fromisoformat(watermark[0]) - dt.timedelta(days=args.overlap_days - 1)).isoformat()
                    args.new_scan = True
                elif args.mode == 'incremental':
                    raise RuntimeError('no_completed_history_watermark_for_incremental')
        finally:
            connection.close()
    elif args.mode == 'incremental':
        raise RuntimeError('no_completed_history_watermark_for_incremental')
    if args.start_date and args.start_date > args.end_date:
        raise RuntimeError('incremental_start_is_after_end_date')
    secret = os.environ.pop('API_APP_SECRET', None) or getpass.getpass('API app secret (hidden; memory only): ')
    ready_fd = None
    if args.detach:
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid:
            os.close(write_fd)
            # Child has its own inherited flock descriptor. Do not explicitly
            # unlock the parent's descriptor, which would unlock the child too.
            lock.close()
            with os.fdopen(read_fd, 'r') as handle:
                line = handle.readline()
            if not line:
                raise RuntimeError('daemon_start_failed_see_reports_daemon_log')
            print(line.strip(), flush=True)
            return 0
        os.close(read_fd)
        os.setsid()
        os.umask(0o077)
        directory = Path(args.root) / 'reports'
        directory.mkdir(parents=True, exist_ok=True)
        output = os.open(str(directory / 'daemon.log'), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        input_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(input_fd, 0)
        os.dup2(output, 1)
        os.dup2(output, 2)
        os.close(input_fd)
        os.close(output)
        ready_fd = write_fd
    runner = None
    try:
        runner = Runner(args, secret)
        signal.signal(signal.SIGTERM, lambda *_: runner.stop.set())
        signal.signal(signal.SIGINT, lambda *_: runner.stop.set())
        return runner.run(ready_fd)
    except Exception as exc:
        # Do not stringify exceptions that might contain credentials or URLs.
        print(canonical({'at': now(), 'event': 'fatal', 'error_type': type(exc).__name__, 'safe_code': str(exc) if isinstance(exc, RuntimeError) and str(exc).replace('_', '').isalnum() else 'see_state'}), flush=True)
        return 1
    finally:
        secret = None
        if runner:
            runner.secret = None
            runner.api.secret = None
            runner.store.close()
        lock.close()


def status_command(args):
    path = Path(args.root) / 'reports/status.json'
    if not path.exists():
        raise RuntimeError('status_not_found')
    status = json.loads(path.read_text())
    pid = status.get('pid')
    try:
        os.kill(pid, 0)
        running = True
    except (ProcessLookupError, TypeError):
        running = False
    status['process_alive'] = running
    if status.get('run'):
        status['run'].pop('config_json', None)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def stop_command(args):
    process = json.loads((Path(args.root) / 'state/process.json').read_text())
    pid = process['pid']
    # PID reuse must not signal an unrelated process.
    command = Path('/proc/{}/cmdline'.format(pid))
    if not command.exists():
        print(canonical({'stopped': True, 'reason': 'process_already_exited'}))
        return 0
    cmdline = command.read_bytes().replace(b'\x00', b' ').decode('utf-8', 'replace')
    if 'pbiomon_sync.py' not in cmdline or str(Path(args.root)) not in cmdline:
        raise RuntimeError('pid_identity_mismatch')
    os.kill(pid, signal.SIGTERM)
    print(canonical({'signal_sent': 'SIGTERM', 'pid': pid}))
    return 0


def verify_command(args):
    store = Store(args.root)
    bad, checked = 0, 0
    try:
        with store.lock:
            rows = [dict(row) for row in store.db.execute("SELECT resource_type,resource_id,file_path,byte_size,sha256 FROM resources WHERE status='done' LIMIT ?", (args.limit,))]
        for row in rows:
            path = Path(args.root) / row['file_path']
            valid = path.is_file() and path.stat().st_size == row['byte_size']
            if valid and args.hash:
                valid = sha256_file(path) == row['sha256']
            checked += 1
            bad += not valid
        print(canonical({'checked': checked, 'invalid': bad, 'sha256_checked': args.hash, 'modified': False}))
        return 1 if bad else 0
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--root', default=DEFAULT_ROOT)
    run.add_argument('--api-base', default='https://open.pbiomon.cn')
    run.add_argument('--app-id', default=os.environ.get('API_APP_ID', 'ai_p_20261102'))
    run.add_argument('--end-date', default=dt.datetime.now(TZ).date().isoformat())
    run.add_argument('--start-date')
    run.add_argument('--mode', choices=['auto', 'history', 'incremental'], default='auto')
    run.add_argument('--overlap-days', type=int, default=7)
    run.add_argument('--include', nargs='+', choices=['PHOTO', 'VIDEO'], default=['PHOTO', 'VIDEO'])
    run.add_argument('--private-ip', default='10.249.0.110')
    run.add_argument('--workers', type=int, default=4)
    run.add_argument('--api-interval', type=float, default=6.5)
    run.add_argument('--reserve-gib', type=float, default=100)
    run.add_argument('--max-photo-mib', type=float, default=256)
    run.add_argument('--max-video-gib', type=float, default=16)
    run.add_argument('--max-attempts', type=int, default=5)
    run.add_argument('--split-pages', type=int, default=1000)
    run.add_argument('--report-interval', type=float, default=30)
    run.add_argument('--max-downloads', type=int, default=0, help='bounded verification run; 0 means unlimited')
    run.add_argument('--max-seconds', type=float, default=0, help='bounded verification run; 0 means unlimited')
    run.add_argument('--new-scan', action='store_true')
    run.add_argument('--retry-failed', action='store_true')
    run.add_argument('--recheck-unavailable', action='store_true', help='explicit one-off recheck after source missing URLs have been repaired')
    run.add_argument('--detach', action='store_true')
    run.set_defaults(func=run_command)
    for name, function in [('status', status_command), ('stop', stop_command), ('verify', verify_command)]:
        command = sub.add_parser(name)
        command.add_argument('--root', default=DEFAULT_ROOT)
        if name == 'verify':
            command.add_argument('--limit', type=int, default=1000)
            command.add_argument('--hash', action='store_true')
        command.set_defaults(func=function)
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(canonical({'event': 'command_failed', 'error_type': type(exc).__name__, 'safe_code': str(exc) if isinstance(exc, RuntimeError) and str(exc).replace('_', '').isalnum() else 'see_state'}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
