#!/usr/bin/env python3
"""Audit archived record/file associations without changing the source ledger.

Works from a consistent local SQLite backup, preserves evidence, never logs URLs,
and distinguishes API/file bookkeeping from visual annotation correctness.
"""
import argparse
import collections
import concurrent.futures
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import types

SPEC = importlib.util.spec_from_file_location('pbiomon_sync', Path(__file__).with_name('pbiomon_sync.py'))
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


def emit(event, **fields):
    print(sync.canonical({'at': sync.now(), 'event': event, **fields}), flush=True)


def normalized_annotation(record):
    return {key: record[key] for key in ('speciesId', 'speciesName', 'finalIdentification', 'ai')}


def identity(record):
    return record['resourceType'], record['resourceId'], record['speciesId']


def source_provenance_pair(record):
    url=record.get('downloadUrl')
    if not url:
        return None
    metadata={k:v for k,v in record.items() if k not in ('speciesId','speciesName','finalIdentification','ai','downloadUrl')}
    return hashlib.sha256(url.encode('utf-8')).hexdigest(),sync.digest(metadata)


def receipt_source_matches(receipt, observed_pairs):
    url=receipt.get('source_url_sha256')
    if not url:
        return None  # Legacy receipts cannot retroactively prove the request URL.
    metadata=receipt.get('source_resource_metadata_sha256')
    if metadata:
        return (url,metadata) in observed_pairs
    return any(pair[0]==url for pair in observed_pairs)


def bundle_matches(bundle, resource, annotations):
    """Independent checks of keys, the species set, annotation values and asset."""
    base = bundle.get('resource', {})
    if (base.get('resourceType'), base.get('resourceId')) != (resource['resource_type'], resource['resource_id']):
        return False
    saved = bundle.get('file', {})
    if saved.get('sha256') != resource['sha256'] or saved.get('path') != resource['file_path'] or saved.get('status') != 'done':
        return False
    entries = bundle.get('annotations', [])
    expected = {a['species_id']: a['annotation_hash'] for a in annotations}
    if len(entries) != len(expected) or {a.get('speciesId') for a in entries} != set(expected):
        return False
    return all(sync.digest(a) == expected[a['speciesId']] for a in entries)


def refetch_private(root, resource, output_dir):
    """Bounded independent fetch; does not use receipts or alter resource state."""
    client = sync.Downloader(types.SimpleNamespace(root=root), '10.249.0.110', 1, threading.Event(), lambda *a, **k: None)
    result = {'resourceType': resource['resource_type'], 'resourceId': resource['resource_id'], 'verified': False}
    try:
        with tempfile.TemporaryDirectory(prefix='refetch-', dir=output_dir) as directory:
            target = Path(directory) / 'asset.bin'
            command, config = client.command(resource, target)
            command[command.index('--max-time') + 1] = '90'
            command[command.index('--max-filesize') + 1] = str(min(resource['byte_size'] + 65536, 128 * 1024**2))
            process = subprocess.run(command, input=config, capture_output=True, text=True, timeout=100)
            parts = process.stdout.strip().split('\t')
            result['curl_exit_code'] = process.returncode
            if len(parts) == 5:
                result.update({'httpStatus': int(parts[0]), 'peerIp': parts[1], 'bytes': int(float(parts[2])), 'tlsVerified': parts[4] == '0'})
            if process.returncode == 0 and len(parts) == 5 and parts[0] == '200' and parts[1] == '10.249.0.110' and parts[4] == '0':
                result['sha256MatchesArchivedFile'] = sync.sha256_file(target) == resource['sha256']
                result['verified'] = result['sha256MatchesArchivedFile']
    except Exception as exc:
        result['errorType'] = type(exc).__name__
    return result


def create_visual_samples(root, output, resources, annotations, count):
    from PIL import Image, ImageDraw, ImageFont
    directory = output / 'visuals'
    directory.mkdir()
    font_paths = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    font = next((ImageFont.truetype(path, 20) for path in font_paths if Path(path).is_file()), ImageFont.load_default())
    chosen = []
    candidates = [r for r in resources if r['resource_type'] == 'PHOTO' and 0 < (r['byte_size'] or 0) <= 10*1024**2]
    random.Random(20261102).shuffle(candidates)
    seen = set()
    # Always include multi-species resources when available, rather than relying
    # on a uniform sample to catch rare cross-species joins.
    for row in candidates:
        values = annotations[(row['resource_type'], row['resource_id'])]
        if len(values) > 1:
            chosen.append(row)
            seen.add(tuple(sorted(a['species_name'] for a in values)))
        if len(chosen) >= min(2, count):
            break
    for row in candidates:
        if len(chosen) >= count:
            break
        values = annotations[(row['resource_type'], row['resource_id'])]
        species = tuple(sorted(a['species_name'] for a in values))
        if species in seen or not any(a['box_count'] for a in values):
            continue
        chosen.append(row)
        seen.add(species)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        selected = {r['resource_key'] for r in chosen}
        chosen.extend(r for r in candidates if r['resource_key'] not in selected)
        chosen = chosen[:count]
    manifest = []
    cell_width, cell_height = 640, 520
    sheet = Image.new('RGB', (cell_width*3, cell_height*((len(chosen)+2)//3)), 'white')
    for index, row in enumerate(chosen):
        values = annotations[(row['resource_type'], row['resource_id'])]
        names = [a['species_name'] for a in values]
        with Image.open(root / row['file_path']) as original:
            image = original.convert('RGB')
            width, height = image.size
        image.thumbnail((cell_width, 440))
        top = (index//3)*cell_height
        left = (index%3)*cell_width
        sheet.paste(image, (left, top+65))
        draw = ImageDraw.Draw(sheet)
        draw.text((left+8,top+4), '{} {}'.format(index+1, ', '.join(names)), fill='black', font=font)
        draw.text((left+8,top+29), row['resource_id'][:24], fill='black', font=font)
        # Copy pixels for independent inspection; no coordinate assumption here.
        image.save(directory / ('photo-{:02d}.jpg'.format(index+1)), quality=94)
        manifest.append({'index': index+1, 'kind': 'PHOTO', 'resourceId': row['resource_id'], 'species': names, 'originalWidth': width, 'originalHeight': height, 'file': 'photo-{:02d}.jpg'.format(index+1), 'annotations': [json.loads(a['annotation_json']) for a in values]})
    if chosen:
        sheet.save(directory / 'photos.jpg', quality=94)
    videos = [r for r in resources if r['resource_type'] == 'VIDEO' and 0 < (r['byte_size'] or 0) <= 128*1024**2]
    random.Random(20261102).shuffle(videos)
    video_chosen = []
    video_species = set()
    for row in videos:
        names = tuple(sorted(a['species_name'] for a in annotations[(row['resource_type'], row['resource_id'])]))
        if names in video_species and video_chosen:
            continue
        video_chosen.append(row)
        video_species.add(names)
        if len(video_chosen) == 3:
            break
    if shutil.which('ffmpeg'):
        for index, row in enumerate(video_chosen):
            values = annotations[(row['resource_type'], row['resource_id'])]
            names = [a['species_name'] for a in values]
            media = json.loads(row['media_json'] or '{}')
            duration = float(media.get('probe', {}).get('format', {}).get('duration', 0))
            if duration <= 0:
                continue
            strip = Image.new('RGB', (640*4, 430*2), 'white')
            times = [duration*(i+0.25)/8 for i in range(8)]
            frames = []
            for frame, second in enumerate(times):
                filename = directory / ('video-{:02d}-frame-{:02d}.jpg'.format(index+1,frame+1))
                process = subprocess.run(['ffmpeg','-v','error','-i',str(root/row['file_path']),'-ss',str(second),'-frames:v','1','-vf','scale=640:-2','-q:v','3','-y',str(filename)], capture_output=True, timeout=45)
                if process.returncode or not filename.exists():
                    continue
                with Image.open(filename) as image:
                    image = image.convert('RGB')
                    image.thumbnail((640,360))
                    x,y = (frame%4)*640, (frame//4)*430
                    strip.paste(image,(x,y+60))
                draw = ImageDraw.Draw(strip)
                draw.text((x+5,y+3), '{} t={:.2f}s'.format(', '.join(names),second), fill='black', font=font)
                draw.text((x+5,y+30), row['resource_id'][:24], fill='black', font=font)
                frames.append({'seconds':second,'file':filename.name})
            strip_name='video-{:02d}-timeline.jpg'.format(index+1)
            strip.save(directory/strip_name, quality=94)
            manifest.append({'index':index+1,'kind':'VIDEO','resourceId':row['resource_id'],'species':names,'durationSeconds':duration,'frames':frames,'file':strip_name,'sampling':'evenly_spaced_content_samples_not_claimed_annotation_frames','annotations':[json.loads(a['annotation_json']) for a in values]})
    sync.atomic_json(directory / 'manifest.json', manifest)
    return chosen, video_chosen, manifest


def audit(args):
    os.umask(0o077)
    root = Path(args.root).resolve()
    output = root / 'reports' / ('association-audit-' + datetime.datetime.now(sync.TZ).strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True)
    source = sqlite3.connect('file:'+str(root/'state/state.sqlite')+'?mode=ro', uri=True)
    destination = sqlite3.connect(output/'state_snapshot.sqlite')
    source.backup(destination, pages=1000)
    destination.close()
    source.close()
    db = sqlite3.connect('file:'+str(output/'state_snapshot.sqlite')+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    report={'startedAt':sync.now(),'root':str(root),'output':str(output),'scope':'consistent_local_ledger_snapshot','sourceApiSnapshotGuaranteed':False,'visualCorrectnessOfAllAnnotationsVerified':False}
    issues=collections.Counter()
    examples=[]
    def issue(code, rid=None):
        issues[code]+=1
        if len(examples)<100:
            examples.append({'code':code,'resourceId':rid})
    rows=[dict(r) for r in db.execute('SELECT * FROM resources')]
    all_annotations=[dict(r) for r in db.execute('SELECT * FROM annotations')]
    by_resource=collections.defaultdict(list)
    by_page=collections.defaultdict(list)
    by_key={(r['resource_type'],r['resource_id']):r for r in rows}
    for annotation in all_annotations:
        key=(annotation['resource_type'],annotation['resource_id'])
        by_resource[key].append(annotation)
        by_page[annotation['source_page']].append(annotation)
        if key not in by_key:
            issue('annotation_without_resource',annotation['resource_id'])
        if sync.digest(json.loads(annotation['annotation_json']))!=annotation['annotation_hash']:
            issue('annotation_hash_mismatch',annotation['resource_id'])
    emit('audit_snapshot_ready', resources=len(rows), annotation_records=len(all_annotations), source_pages=len(by_page))
    original_benchmark={}
    legacy_root=Path(args.benchmark_root)
    for path in (legacy_root/'responses').glob('annotations-page-*.json'):
        payload=json.loads(path.read_text())
        for record in payload['data']['records']:
            original_benchmark[identity(record)]=sync.digest(normalized_annotation(record))
    source_verified=0
    source_pairs=collections.defaultdict(set)
    source_paths=[r[0] for r in db.execute('SELECT DISTINCT raw_path FROM pages')]
    for page_index,relative in enumerate(source_paths):
        annotations=by_page.get(relative,[])
        path=root/relative
        try:
            with gzip.open(path,'rt',encoding='utf-8') as handle:
                payload=json.load(handle)
            index={identity(r):r for r in payload['data']['records']}
            for record in payload['data']['records']:
                pair=source_provenance_pair(record)
                if pair:
                    source_pairs[(record['resourceType'],record['resourceId'])].add(pair)
            imported=payload.get('local_provenance',{}).get('kind')=='prior_benchmark_local_cache'
            for annotation in annotations:
                key=(annotation['resource_type'],annotation['resource_id'],annotation['species_id'])
                record=index.get(key)
                if not record:
                    issue('annotation_missing_in_source_page',key[1])
                elif sync.digest(normalized_annotation(record))!=annotation['annotation_hash']:
                    issue('annotation_disagrees_with_source_page',key[1])
                elif imported and original_benchmark.get(key)!=annotation['annotation_hash']:
                    issue('cache_import_not_verified_against_original_api_response',key[1])
                else:
                    source_verified+=1
        except (OSError,ValueError,KeyError):
            issue('source_page_unreadable',relative)
        if (page_index+1)%100==0:
            emit('audit_source_progress', pages=page_index+1, verified_annotations=source_verified)
    completed=[r for r in rows if r['status']=='done']
    files={}
    aggregate_verified=aggregate_pending=0
    receipts_without_url_hash=0
    for index,row in enumerate(completed):
        rid=row['resource_id']
        key=(row['resource_type'],rid)
        metadata=json.loads(row['metadata_json'])
        if (metadata.get('resourceType'),metadata.get('resourceId'))!=key or sync.resource_key(*key)!=row['resource_key']:
            issue('resource_identity_mismatch',rid)
        path=root/row['file_path']
        if not path.is_file():
            issue('completed_file_missing',rid)
        elif path.stat().st_size!=row['byte_size']:
            issue('completed_file_size_mismatch',rid)
        else:
            if str(path) in files and files[str(path)]!=row['sha256']:
                issue('same_object_path_with_conflicting_expected_hashes',rid)
            files[str(path)]=row['sha256']
        receipt_path=root/'raw/receipts'/row['resource_key'][:2]/(row['resource_key']+'.json')
        try:
            receipt=json.loads(receipt_path.read_text())
            if (receipt.get('resource_type'),receipt.get('resource_id'))!=key:
                issue('receipt_resource_identity_mismatch',rid)
            if any(receipt.get(field)!=row[field] for field in ('file_path','sha256','byte_size')):
                issue('receipt_file_reference_mismatch',rid)
            if receipt.get('transport')=='private_https' and (receipt.get('remote_ip')!='10.249.0.110' or receipt.get('tls_verified') is not True):
                issue('receipt_private_route_mismatch',rid)
            provenance=receipt_source_matches(receipt,source_pairs[key])
            if provenance is None:
                receipts_without_url_hash+=1
            elif not provenance:
                issue('receipt_source_link_not_observed_for_same_resource',rid)
        except (OSError,ValueError):
            issue('receipt_unreadable',rid)
        if row['annotation_dirty'] or row['saved_revision']!=row['annotation_revision'] or not row['annotation_path']:
            aggregate_pending+=1
        else:
            current=root/row['annotation_path']
            candidates=[current]+list(current.parent.glob('*.json'))
            matched=False
            for candidate in candidates:
                try:
                    if bundle_matches(json.loads(candidate.read_text()),row,by_resource[key]):
                        matched=True
                        break
                except (OSError,ValueError,KeyError):
                    continue
            if matched:
                aggregate_verified+=1
            else:
                issue('aggregate_species_or_asset_mismatch',rid)
        if (index+1)%1000==0:
            emit('audit_association_progress', completed_resources=index+1, issues=sum(issues.values()))
    hashed=0
    hashed_bytes=0
    def check_file(item):
        path,expected=item
        try:
            return path,sync.sha256_file(path)==expected,Path(path).stat().st_size
        except OSError:
            return path,False,0
    emit('audit_full_hash_start', unique_objects=len(files), bytes=sum(Path(p).stat().st_size for p in files if Path(p).is_file()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.hash_workers) as pool:
        for path,valid,size in pool.map(check_file,files.items()):
            hashed+=1
            hashed_bytes+=size
            if not valid:
                issue('object_sha256_mismatch',Path(path).name)
            if hashed%500==0:
                emit('audit_hash_progress', objects=hashed, bytes=hashed_bytes, issues=sum(issues.values()))
    report.update({'resourceCount':len(rows),'annotationRecordCount':len(all_annotations),'annotationsVerifiedAgainstSourcePages':source_verified,'completedResources':len(completed),'uniqueObjectsFullyHashed':hashed,'bytesFullyHashed':hashed_bytes,'aggregateAnnotationBundlesVerified':aggregate_verified,'aggregateBundlesPendingAtSnapshot':aggregate_pending,'legacyReceiptsWithoutSourceUrlHash':receipts_without_url_hash,'issues':dict(issues),'issueExamples':examples,'bookkeepingAssociationPassed':not issues})
    sync.atomic_json(output/'association-summary.json',report)
    if args.visual_samples:
        photos,videos,manifest=create_visual_samples(root,output,completed,by_resource,args.visual_samples)
        report['visualSamplePhotos']=len(photos)
        report['visualSampleVideos']=len(videos)
        selected=photos[:args.refetch_photos]+videos[:args.refetch_videos]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            checks=list(pool.map(lambda r:refetch_private(root,r,output),selected))
        report['independentPrivateRefetchChecks']=checks
        report['independentPrivateRefetchMatches']=sum(c['verified'] for c in checks)
        report['visualEvidenceManifest']=str(output/'visuals/manifest.json')
    report['finishedAt']=sync.now()
    sync.atomic_json(output/'association-summary.json',report)
    emit('audit_complete', report_file=str(output/'association-summary.json'), **{k:v for k,v in report.items() if k not in ('issueExamples','independentPrivateRefetchChecks')})
    db.close()
    return 1 if issues else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='/mnt/data2/pbiomon_yolo')
    parser.add_argument('--benchmark-root',default='/home/yuzhong/pbiomon_api_benchmark/20261102_153313')
    parser.add_argument('--hash-workers',type=int,default=2)
    parser.add_argument('--visual-samples',type=int,default=12)
    parser.add_argument('--refetch-photos',type=int,default=8)
    parser.add_argument('--refetch-videos',type=int,default=2)
    return audit(parser.parse_args())


if __name__=='__main__':
    sys.exit(main())
