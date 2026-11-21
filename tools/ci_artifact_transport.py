#!/usr/bin/env python3
"""Transfer the complete official artifact ZIP in independently checked chunks.

The original uploaded artifact remains authoritative. Each chunk worker reads
the complete official ZIP, verifies its API and upload-action digest, retains
one byte interval and publishes the complete interval inventory. No archive
member is selected, unpacked, recompressed or replaced by this helper.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request


CHUNK_BYTES = 200 * 1024 * 1024
SCHEMA = 'heptatorch.official-artifact-zip-transport.v1'
ARTIFACT_NAME = 'current-runtime-native-evidence'
API = 'https://api.github.com'


class TransportError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise TransportError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256(value):
    value = value.removeprefix('sha256:')
    need(re.fullmatch('[0-9a-f]{64}', value) is not None, 'complete SHA256 required')
    return value


def dump(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        return None


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        need(parsed.scheme == 'https' and bool(parsed.hostname)
             and parsed.username is None and parsed.password is None,
             'payload redirect must retain anonymous HTTPS')
        need(not request.has_header('Authorization'), 'payload requests carry no API Authorization')
        return super().redirect_request(request, response, code, message, headers, newurl)


def network_error(error, label):
    # HTTPError text may contain a signed URL. Never include that text or headers.
    suffix = ' HTTP ' + str(error.code) if isinstance(error, urllib.error.HTTPError) else ''
    return TransportError(label + ' failed' + suffix)


class GitHub:
    def __init__(self):
        token = os.environ.get('GITHUB_TOKEN')
        need(bool(token), 'GITHUB_TOKEN is required for artifact metadata')
        self.headers = {'Authorization': 'Bearer ' + token,
                        'Accept': 'application/vnd.github+json',
                        'X-GitHub-Api-Version': '2022-11-28',
                        'User-Agent': 'HeptaTorch-official-artifact-transport'}
        self.authenticated = urllib.request.build_opener(NoRedirect())
        self.anonymous = urllib.request.build_opener(HTTPSRedirect())

    def request(self, path):
        need(path.startswith('/repos/') and '?' not in path, 'fixed repository API path required')
        return urllib.request.Request(API + path, headers=self.headers)

    def document(self, path):
        try:
            with self.authenticated.open(self.request(path), timeout=60) as response:
                need(response.status == 200, 'artifact metadata HTTP200 required')
                raw = response.read(1024 * 1024 + 1)
                need(len(raw) <= 1024 * 1024, 'bounded artifact metadata required')
        except (urllib.error.URLError, TimeoutError) as error:
            raise network_error(error, 'artifact metadata request') from None
        value = json.loads(raw)
        need(isinstance(value, dict), 'artifact metadata object required')
        return value

    def payload(self, path):
        try:
            response = self.authenticated.open(self.request(path), timeout=60)
        except urllib.error.HTTPError as error:
            if error.code != 302:
                raise network_error(error, 'artifact archive redirect') from None
            location = error.headers.get('Location', '')
            error.close()
        except (urllib.error.URLError, TimeoutError) as error:
            raise network_error(error, 'artifact archive redirect') from None
        else:
            response.close()
            raise TransportError('official artifact archive must return its HTTP302 download location')
        parsed = urllib.parse.urlsplit(location)
        need(parsed.scheme == 'https' and bool(parsed.hostname)
             and parsed.username is None and parsed.password is None,
             'official artifact payload requires an HTTPS location')
        # This new request has no Authorization header. Neither the signed URL
        # nor the API token is written to reports, outputs or logs.
        request = urllib.request.Request(location, headers={'Accept-Encoding': 'identity'})
        try:
            response = self.anonymous.open(request, timeout=120)
            need(response.status == 200, 'complete artifact payload HTTP200 required')
            return response
        except (urllib.error.URLError, TimeoutError) as error:
            raise network_error(error, 'artifact payload download') from None


def identity(args, github):
    need(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repository) is not None
         and all(part not in ('.', '..') for part in args.repository.split('/')),
         'owner/repository identity required')
    need(args.artifact_id > 0 and args.run_id > 0 and args.run_attempt > 0, 'positive GitHub identities required')
    need(re.fullmatch('[0-9a-f]{40}', args.checkout_sha) is not None, 'exact checkout commit required')
    need(not args.pull_request_head_sha or re.fullmatch('[0-9a-f]{40}', args.pull_request_head_sha) is not None,
         'exact pull-request head identity required when present')
    artifact = github.document('/repos/' + args.repository + '/actions/artifacts/' + str(args.artifact_id))
    # A failed transport job can be retried after the producer has succeeded.
    # Read the producer's immutable attempt, not the latest run attempt.
    run = github.document('/repos/' + args.repository + '/actions/runs/'
                          + str(args.run_id) + '/attempts/' + str(args.run_attempt))
    expected_digest = sha256(args.artifact_digest)
    need(artifact['id'] == args.artifact_id and artifact['name'] == ARTIFACT_NAME
         and artifact['expired'] is False and type(artifact['size_in_bytes']) is int
         and artifact['size_in_bytes'] > 0 and sha256(artifact['digest']) == expected_digest,
         'the original upload id, name, extent and digest must match the REST artifact')
    need(run['id'] == args.run_id and run['run_attempt'] == args.run_attempt
         and run['repository']['full_name'] == args.repository and run['event'] == args.event_name,
         'the artifact must belong to this exact workflow run and attempt')
    need(artifact['workflow_run']['id'] == run['id']
         and artifact['workflow_run']['head_sha'] == run['head_sha']
         and re.fullmatch('[0-9a-f]{40}', run['head_sha']) is not None,
         'artifact and run REST head identities must agree')
    if args.expected_run_head_sha:
        need(run['head_sha'] == args.expected_run_head_sha, 'the original planned REST run head changed')
    if args.artifact_bytes is not None:
        need(artifact['size_in_bytes'] == args.artifact_bytes, 'the original planned ZIP extent changed')
    helper_sha = digest(Path(__file__).read_bytes())
    if args.helper_sha256:
        need(helper_sha == sha256(args.helper_sha256), 'the checked-out transport helper differs from the original job')
    checkout = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=Path(__file__).resolve().parents[1], text=True).strip()
    need(checkout == args.checkout_sha, 'transport uses the same exact tested checkout')
    # REST run head, event PR head and tested merge checkout are distinct facts.
    # Their relationships are recorded without assuming the three are equal.
    return dict(repository=args.repository, run_id=run['id'], run_attempt=run['run_attempt'],
        event_name=run['event'], checkout_sha=checkout,
        pull_request_head_sha=args.pull_request_head_sha or None,
        rest_run_head_sha=run['head_sha'], artifact_workflow_head_sha=artifact['workflow_run']['head_sha'],
        head_relationships=dict(artifact_matches_rest_run=True,
            checkout_matches_rest_run=checkout == run['head_sha'],
            rest_run_matches_pr_head=(run['head_sha'] == args.pull_request_head_sha) if args.pull_request_head_sha else None,
            checkout_matches_pr_head=(checkout == args.pull_request_head_sha) if args.pull_request_head_sha else None),
        original_artifact=dict(id=artifact['id'], name=artifact['name'], bytes=artifact['size_in_bytes'],
            sha256=expected_digest, api_digest=artifact['digest'],
            created_at=artifact['created_at'], expires_at=artifact['expires_at'],
            workflow_run=artifact['workflow_run']), transport_helper_sha256=helper_sha)


def part_name(index):
    return 'original.zip.part-' + format(index, '04d')


def chunk_stream(stream, expected_bytes, expected_sha, selected, directory, chunk_bytes=CHUNK_BYTES):
    need(type(expected_bytes) is int and expected_bytes > 0, 'positive complete ZIP extent required')
    need(type(chunk_bytes) is int and 0 < chunk_bytes <= CHUNK_BYTES, 'bounded chunk extent required')
    count = (expected_bytes + chunk_bytes - 1) // chunk_bytes
    need(type(selected) is int and 0 <= selected < count, 'selected chunk must be inside the complete ZIP')
    expected_sha = sha256(expected_sha)
    final = directory / part_name(selected)
    partial = directory / (part_name(selected) + '.partial')
    need(not final.exists() and not final.is_symlink(), 'fresh chunk output required')
    whole, current, total, extent, rows = hashlib.sha256(), hashlib.sha256(), 0, 0, []
    with partial.open('xb') as target:
        while True:
            try:
                raw = stream.read(min(2 * 1024 * 1024, chunk_bytes - extent))
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                raise network_error(error, 'artifact payload read') from None
            if not raw:
                break
            need(total + len(raw) <= expected_bytes, 'official ZIP exceeds the exact API extent')
            whole.update(raw)
            current.update(raw)
            if len(rows) == selected:
                target.write(raw)
            total += len(raw)
            extent += len(raw)
            if extent == chunk_bytes:
                rows.append(dict(index=len(rows), offset=total - extent, bytes=extent,
                                 sha256=current.hexdigest(), path=part_name(len(rows))))
                current, extent = hashlib.sha256(), 0
        if extent:
            rows.append(dict(index=len(rows), offset=total - extent, bytes=extent,
                             sha256=current.hexdigest(), path=part_name(len(rows))))
    need(total == expected_bytes and len(rows) == count, 'complete official ZIP and final short chunk required')
    need(whole.hexdigest() == expected_sha, 'complete downloaded ZIP digest differs from the original upload')
    need(partial.stat().st_size == rows[selected]['bytes'], 'selected chunk extent differs')
    os.link(partial, final)
    partial.unlink()
    return rows


def verify(manifest, directory, output, artifact_id, artifact_digest, artifact_bytes):
    need(manifest.get('schema') == SCHEMA and manifest.get('status') == 'PASS_OFFICIAL_ZIP_CHUNK',
         'successful official ZIP transport inventory required')
    original, rows = manifest['source']['original_artifact'], manifest['chunks']
    need(original['id'] == artifact_id and original['bytes'] == artifact_bytes
         and sha256(original['sha256']) == sha256(artifact_digest),
         'inventory must match the independently supplied original artifact identity')
    need(manifest.get('complete_original_zip_digest_verified') is True
         and manifest.get('original_artifact_modified') is False
         and manifest.get('original_zip_members_selected_or_recompressed') is False,
         'unmodified complete original ZIP transport required')
    chunk_bytes = manifest['chunk_bytes']
    need(type(chunk_bytes) is int and 0 < chunk_bytes <= CHUNK_BYTES
         and type(original['bytes']) is int and original['bytes'] > 0,
         'valid original ZIP and chunk extents required')
    need(isinstance(rows, list) and len(rows) == (original['bytes'] + chunk_bytes - 1) // chunk_bytes,
         'every original ZIP interval is required')
    need(not output.exists() and not output.is_symlink(), 'fresh reconstructed ZIP destination required')
    partial = output.with_name(output.name + '.partial')
    total, whole = 0, hashlib.sha256()
    with partial.open('xb') as target:
        for index, row in enumerate(rows):
            expected_size = min(chunk_bytes, original['bytes'] - total)
            need(row['index'] == index and row['offset'] == total and row['path'] == part_name(index)
                 and row['bytes'] == expected_size, 'complete contiguous original ZIP intervals required')
            path = directory / row['path']
            need(path.is_file() and not path.is_symlink(), 'regular original ZIP chunk required')
            current, extent = hashlib.sha256(), 0
            with path.open('rb') as source:
                while raw := source.read(2 * 1024 * 1024):
                    extent += len(raw)
                    need(extent <= expected_size, 'chunk exceeds its original byte interval')
                    current.update(raw)
                    whole.update(raw)
                    target.write(raw)
            need(extent == expected_size and current.hexdigest() == sha256(row['sha256']),
                 'complete original chunk SHA and extent must match')
            total += extent
    need(total == original['bytes'] and whole.hexdigest() == sha256(original['sha256']),
         'reassembled ZIP must equal the original upload digest and extent')
    os.link(partial, output)
    partial.unlink()
    return dict(status='PASS_REASSEMBLED_ORIGINAL_ZIP', bytes=total, sha256=whole.hexdigest(), chunks=len(rows))


def outputs(path, values):
    with path.open('a', encoding='utf-8') as stream:
        for key, value in values.items():
            value = str(value)
            need('\n' not in value and '\r' not in value, 'single-line workflow output required')
            stream.write(key + '=' + value + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('plan', 'chunk'):
        sub = commands.add_parser(command)
        for name in ('repository', 'checkout-sha', 'artifact-digest', 'event-name'):
            sub.add_argument('--' + name, required=True)
        for name in ('run-id', 'run-attempt', 'artifact-id'):
            sub.add_argument('--' + name, type=int, required=True)
        sub.add_argument('--pull-request-head-sha', default='')
        sub.add_argument('--expected-run-head-sha', default='')
        sub.add_argument('--helper-sha256', default='')
        sub.add_argument('--artifact-bytes', type=int)
        sub.add_argument('--output-directory', type=Path, required=True)
        sub.add_argument('--github-output', type=Path, required=True)
        if command == 'chunk':
            sub.add_argument('--index', type=int, required=True)
            sub.add_argument('--transport-attempt', type=int, required=True)
    sub = commands.add_parser('verify')
    sub.add_argument('--manifest', type=Path, required=True)
    sub.add_argument('--chunks-directory', type=Path, required=True)
    sub.add_argument('--output', type=Path, required=True)
    sub.add_argument('--artifact-id', type=int, required=True)
    sub.add_argument('--artifact-digest', required=True)
    sub.add_argument('--artifact-bytes', type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify':
            result = verify(json.loads(args.manifest.read_bytes()), args.chunks_directory, args.output,
                            args.artifact_id, args.artifact_digest, args.artifact_bytes)
        else:
            github = GitHub()
            source = identity(args, github)
            original = source['original_artifact']
            count = (original['bytes'] + CHUNK_BYTES - 1) // CHUNK_BYTES
            args.output_directory.mkdir(parents=True, exist_ok=False)
            if args.command == 'plan':
                result = dict(schema=SCHEMA, status='PLANNED_OFFICIAL_ZIP_TRANSFER',
                              source=source, chunk_bytes=CHUNK_BYTES, parts=list(range(count)))
                dump(args.output_directory / 'plan.json', result)
                outputs(args.github_output, {'parts': json.dumps(result['parts']),
                    'artifact-bytes': original['bytes'], 'rest-head-sha': source['rest_run_head_sha'],
                    'helper-sha256': source['transport_helper_sha256'], 'run-attempt': source['run_attempt']})
            else:
                need(args.transport_attempt >= source['run_attempt'], 'transport attempt must follow the original producer')
                with github.payload('/repos/' + args.repository + '/actions/artifacts/'
                                    + str(args.artifact_id) + '/zip') as stream:
                    rows = chunk_stream(stream, original['bytes'], original['sha256'], args.index, args.output_directory)
                for row in rows:
                    row['artifact_name'] = 'current-runtime-evidence-zip-' + str(args.artifact_id) + '-part-' + format(row['index'], '04d')
                result = dict(schema=SCHEMA, status='PASS_OFFICIAL_ZIP_CHUNK', source=source,
                    chunk_bytes=CHUNK_BYTES, selected_part=args.index, chunks=rows,
                    transport_run_attempt=args.transport_attempt,
                    complete_original_zip_digest_verified=True, original_artifact_modified=False,
                    original_zip_members_selected_or_recompressed=False)
                dump(args.output_directory / 'transport-manifest.json', result)
                outputs(args.github_output, {'artifact-name': rows[args.index]['artifact_name']})
        print(json.dumps({key: result[key] for key in ('status', 'bytes', 'sha256', 'selected_part') if key in result}, sort_keys=True))
        return 0
    except Exception as error:
        # Controlled checks use fixed messages. Unexpected errors are identified
        # by class only so tokens and signed locations cannot enter CI logs.
        message = str(error) if isinstance(error, TransportError) else type(error).__name__
        print(json.dumps(dict(status='FAIL_ARTIFACT_TRANSPORT', error=message), sort_keys=True), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
