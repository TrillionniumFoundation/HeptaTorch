"""Network-free checks for lossless official-artifact transport failures."""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.request


HELPER = Path(__file__).resolve().parents[1] / 'tools/ci_artifact_transport.py'
SPEC = importlib.util.spec_from_file_location('ci_artifact_transport', HELPER)
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


class ShortReads(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3))


class MetadataResponse(io.BytesIO):
    status = 200


class MetadataClock:
    def __init__(self):
        self.now = 0
        self.waits = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def parts(self, raw, chunk_bytes=7):
        digest = hashlib.sha256(raw).hexdigest()
        count = (len(raw) + chunk_bytes - 1) // chunk_bytes
        inventory = None
        for index in range(count):
            rows = transport.chunk_stream(ShortReads(raw), len(raw), digest, index,
                                          self.root, chunk_bytes)
            if inventory is not None:
                self.assertEqual(rows, inventory)
            inventory = rows
        manifest = dict(schema=transport.SCHEMA, status='PASS_OFFICIAL_ZIP_CHUNK',
            source=dict(original_artifact=dict(id=81, bytes=len(raw), sha256=digest)),
            chunk_bytes=chunk_bytes, chunks=inventory,
            complete_original_zip_digest_verified=True, original_artifact_modified=False,
            original_zip_members_selected_or_recompressed=False)
        return manifest

    def verify(self, manifest, raw):
        return transport.verify(manifest, self.root, self.root / 'restored.zip',
                                81, hashlib.sha256(raw).hexdigest(), len(raw))

    def test_short_reads_and_last_short_chunk_reassemble_original_bytes(self):
        raw = bytes(range(31))
        manifest = self.parts(raw)
        self.assertEqual([row['bytes'] for row in manifest['chunks']], [7, 7, 7, 7, 3])
        result = self.verify(manifest, raw)
        self.assertEqual((self.root / 'restored.zip').read_bytes(), raw)
        self.assertEqual(result['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_exact_boundary_and_one_byte_original(self):
        for index, raw in enumerate((b'x', bytes(range(21)))):
            with self.subTest(bytes=len(raw)):
                directory = self.root / str(index)
                directory.mkdir()
                self.root = directory
                manifest = self.parts(raw)
                self.assertEqual(self.verify(manifest, raw)['bytes'], len(raw))

    def test_wrong_complete_digest_is_never_published(self):
        with self.assertRaisesRegex(transport.TransportError, 'digest differs'):
            transport.chunk_stream(io.BytesIO(b'actual'), 6, '0' * 64, 0, self.root, 4)
        self.assertFalse((self.root / transport.part_name(0)).exists())

    def test_truncated_or_extra_original_is_never_published(self):
        for index, raw in enumerate((b'ab', b'abcd')):
            with self.subTest(raw=raw):
                directory = self.root / str(index)
                directory.mkdir()
                with self.assertRaises(transport.TransportError):
                    transport.chunk_stream(io.BytesIO(raw), 3,
                        hashlib.sha256(b'abc').hexdigest(), 0, directory, 2)
                self.assertFalse((directory / transport.part_name(0)).exists())

    def test_corrupted_chunk_rejects_reassembly(self):
        raw = bytes(range(17))
        manifest = self.parts(raw)
        (self.root / transport.part_name(1)).write_bytes(b'changed')
        with self.assertRaisesRegex(transport.TransportError, 'chunk SHA'):
            self.verify(manifest, raw)
        self.assertFalse((self.root / 'restored.zip').exists())

    def test_missing_or_reordered_inventory_rejects_reassembly(self):
        raw = bytes(range(17))
        manifest = self.parts(raw)
        missing = copy.deepcopy(manifest)
        missing['chunks'].pop()
        with self.assertRaisesRegex(transport.TransportError, 'every original'):
            self.verify(missing, raw)
        reordered = copy.deepcopy(manifest)
        reordered['chunks'].reverse()
        with self.assertRaisesRegex(transport.TransportError, 'contiguous'):
            self.verify(reordered, raw)
        self.assertFalse((self.root / 'restored.zip').exists())

    def test_independent_original_identity_is_required(self):
        raw = b'actual ZIP bytes'
        manifest = self.parts(raw)
        with self.assertRaisesRegex(transport.TransportError, 'independently supplied'):
            transport.verify(manifest, self.root, self.root / 'restored.zip', 81, '0' * 64, len(raw))
        self.assertFalse((self.root / 'restored.zip.partial').exists())

    def test_symlink_chunk_and_existing_destination_are_rejected(self):
        raw = b'original bytes'
        manifest = self.parts(raw)
        first = self.root / transport.part_name(0)
        first.rename(self.root / 'foreign')
        first.symlink_to(self.root / 'foreign')
        with self.assertRaisesRegex(transport.TransportError, 'regular original'):
            self.verify(manifest, raw)
        (self.root / 'restored.zip').write_bytes(b'keep')
        with self.assertRaisesRegex(transport.TransportError, 'fresh reconstructed'):
            self.verify(manifest, raw)
        self.assertEqual((self.root / 'restored.zip').read_bytes(), b'keep')

    def test_payload_redirect_never_forwards_api_authorization(self):
        location = 'https://storage.example.invalid/original.zip?secret=signed-value'
        with mock.patch.dict(os.environ, {'GITHUB_TOKEN': 'private-test-token'}):
            github = transport.GitHub()
        github.authenticated = mock.Mock()
        github.authenticated.open.side_effect = urllib.error.HTTPError(
            'https://api.github.com/repos/o/r/actions/artifacts/81/zip', 302,
            'Found', {'Location': location}, None)
        response = mock.Mock(status=200)
        github.anonymous = mock.Mock()
        github.anonymous.open.return_value = response
        self.assertIs(github.payload('/repos/o/r/actions/artifacts/81/zip'), response)
        first = github.authenticated.open.call_args.args[0]
        second = github.anonymous.open.call_args.args[0]
        self.assertEqual(first.get_header('Authorization'), 'Bearer private-test-token')
        self.assertFalse(second.has_header('Authorization'))
        self.assertEqual(second.full_url, location)

    def test_payload_redirect_downgrade_is_rejected(self):
        request = urllib.request.Request('https://storage.example.invalid/original.zip')
        with self.assertRaisesRegex(transport.TransportError, 'anonymous HTTPS'):
            transport.HTTPSRedirect().redirect_request(request, None, 302, '', {}, 'http://other.invalid/file')

    def test_network_error_messages_never_include_credentials_or_signed_url(self):
        secret = 'https://storage.invalid/file?signature=private-test-token'
        error = urllib.error.HTTPError(secret, 403, secret, {'Authorization': secret}, None)
        self.assertEqual(str(transport.network_error(error, 'artifact payload download')),
                         'artifact payload download failed HTTP 403')
        stderr = io.StringIO()
        args = ['plan', '--repository', 'o/r', '--checkout-sha', 'a' * 40,
                '--artifact-digest', 'b' * 64, '--event-name', 'pull_request',
                '--run-id', '71', '--run-attempt', '1', '--artifact-id', '81',
                '--output-directory', str(self.root / 'out'),
                '--github-output', str(self.root / 'outputs')]
        with mock.patch.object(transport, 'GitHub', side_effect=error), contextlib.redirect_stderr(stderr):
            self.assertEqual(transport.main(args), 1)
        self.assertEqual(json.loads(stderr.getvalue())['error'], 'HTTPError')
        self.assertNotIn('private-test-token', stderr.getvalue())
        self.assertNotIn('https:', stderr.getvalue())

    def test_exact_producer_attempt_and_distinct_git_heads_are_recorded(self):
        merge, head = 'a' * 40, 'c' * 40
        args = argparse.Namespace(repository='o/r', run_id=71, run_attempt=1, artifact_id=81,
            checkout_sha=merge, pull_request_head_sha=head, artifact_digest='b' * 64,
            event_name='pull_request', expected_run_head_sha=head, artifact_bytes=123,
            helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest())
        artifact = dict(id=81, name=transport.ARTIFACT_NAME, expired=False,
            size_in_bytes=123, digest='sha256:' + 'b' * 64, created_at='created', expires_at='expires',
            workflow_run=dict(id=71, head_sha=head))
        run = dict(id=71, run_attempt=1, repository=dict(full_name='o/r'), event='pull_request', head_sha=head)
        github = mock.Mock()
        github.document.side_effect = [artifact, run]
        with mock.patch.object(transport.subprocess, 'check_output', return_value=merge + '\n'):
            result = transport.identity(args, github)
        self.assertEqual(github.document.call_args_list[1].args[0], '/repos/o/r/actions/runs/71/attempts/1')
        self.assertEqual(result['run_attempt'], 1)
        self.assertFalse(result['head_relationships']['checkout_matches_pr_head'])
        self.assertTrue(result['head_relationships']['rest_run_matches_pr_head'])

    def metadata_client(self):
        with mock.patch.dict(os.environ, {'GITHUB_TOKEN': 'private-test-token'}):
            github = transport.GitHub()
        github.authenticated = mock.Mock()
        clock, stderr = MetadataClock(), io.StringIO()
        self.enterContext(mock.patch.object(transport.time, 'monotonic', clock.monotonic))
        self.enterContext(mock.patch.object(transport.time, 'sleep', clock.sleep))
        self.enterContext(contextlib.redirect_stderr(stderr))
        return github, clock, stderr

    def metadata_error(self, code=404, headers=None):
        # Deliberately put private material in every untrusted error field.
        secret = 'https://storage.invalid/file?signature=private-test-token'
        return urllib.error.HTTPError(secret, code, secret, headers or {}, io.BytesIO(secret.encode()))

    def test_new_artifact_metadata_retries_same_identity_after_404(self):
        github, clock, stderr = self.metadata_client()
        errors = [self.metadata_error(), self.metadata_error()]
        github.authenticated.open.side_effect = errors + [MetadataResponse(b'{"id": 81}')]
        path = '/repos/o/r/actions/artifacts/81'
        self.assertEqual(github.document(path), {'id': 81})
        self.assertEqual(clock.waits, [1, 2])
        self.assertTrue(all(error.closed for error in errors))
        for call in github.authenticated.open.call_args_list:
            request = call.args[0]
            self.assertEqual(request.full_url, transport.API + path)
            self.assertEqual(request.get_method(), 'GET')
            self.assertEqual(request.get_header('Authorization'), 'Bearer private-test-token')
        notices = [json.loads(line) for line in stderr.getvalue().splitlines()]
        self.assertEqual([notice['http_status'] for notice in notices], [404, 404])
        self.assertTrue(all(notice['endpoint'] == 'artifact' for notice in notices))
        self.assertNotIn('private-test-token', stderr.getvalue())
        self.assertNotIn('https:', stderr.getvalue())

    def test_exact_producer_attempt_retries_transient_server_errors(self):
        for code in (500, 502, 503, 504):
            with self.subTest(code=code):
                github, clock, stderr = self.metadata_client()
                github.authenticated.open.side_effect = [self.metadata_error(code), MetadataResponse(b'{"id": 71}')]
                path = '/repos/o/r/actions/runs/71/attempts/1'
                self.assertEqual(github.document(path), {'id': 71})
                self.assertEqual(clock.waits, [1])
                self.assertEqual(json.loads(stderr.getvalue())['endpoint'], 'producer run-attempt')
                self.assertTrue(all(call.args[0].full_url == transport.API + path
                                    for call in github.authenticated.open.call_args_list))

    def test_persistent_metadata_404_has_finite_attempts_and_keeps_error(self):
        github, clock, stderr = self.metadata_client()
        github.authenticated.open.side_effect = lambda *args, **kwargs: (_ for _ in ()).throw(self.metadata_error())
        with self.assertRaisesRegex(transport.TransportError, 'artifact metadata request failed HTTP 404 after 7 attempt'):
            github.document('/repos/o/r/actions/artifacts/81')
        self.assertEqual(github.authenticated.open.call_count, 7)
        self.assertEqual(clock.waits, [1, 2, 4, 8, 16, 30])
        self.assertLess(clock.now, transport.METADATA_RETRY_DEADLINE_SECONDS)
        self.assertNotIn('private-test-token', stderr.getvalue())

    def test_metadata_deadline_caps_request_timeout_and_prevents_more_attempts(self):
        github, clock, stderr = self.metadata_client()

        def slow_failure(*args, **kwargs):
            clock.now += kwargs['timeout']
            raise self.metadata_error()

        github.authenticated.open.side_effect = slow_failure
        with self.assertRaisesRegex(transport.TransportError, 'metadata retry deadline exhausted'):
            github.document('/repos/o/r/actions/artifacts/81')
        self.assertEqual(clock.waits, [1])
        self.assertEqual([call.kwargs['timeout'] for call in github.authenticated.open.call_args_list], [60, 29])
        self.assertEqual(clock.now, transport.METADATA_RETRY_DEADLINE_SECONDS)

    def test_metadata_received_after_deadline_is_not_accepted(self):
        github, clock, stderr = self.metadata_client()

        def late_success(*args, **kwargs):
            clock.now = transport.METADATA_RETRY_DEADLINE_SECONDS + 1
            return MetadataResponse(b'{"id": 81}')

        github.authenticated.open.side_effect = late_success
        with self.assertRaisesRegex(transport.TransportError, 'exceeded the metadata retry deadline'):
            github.document('/repos/o/r/actions/artifacts/81')
        self.assertEqual(github.authenticated.open.call_count, 1)
        self.assertEqual(clock.waits, [])

    def test_metadata_retry_after_is_respected_or_fails_without_early_retry(self):
        github, clock, stderr = self.metadata_client()
        github.authenticated.open.side_effect = [self.metadata_error(503, {'Retry-After': '9'}),
                                                 MetadataResponse(b'{"id": 81}')]
        self.assertEqual(github.document('/repos/o/r/actions/artifacts/81'), {'id': 81})
        self.assertEqual(clock.waits, [9])
        for delay in ('91', 'Wed, 01 Jan 2031 00:00:00 GMT', 'private-test-token'):
            with self.subTest(delay=delay):
                github, clock, stderr = self.metadata_client()
                github.authenticated.open.side_effect = self.metadata_error(503, {'Retry-After': delay})
                with self.assertRaises(transport.TransportError) as caught:
                    github.document('/repos/o/r/actions/artifacts/81')
                self.assertEqual(github.authenticated.open.call_count, 1)
                self.assertEqual(clock.waits, [])
                self.assertNotIn(delay, str(caught.exception))

    def test_metadata_authentication_rate_limit_and_other_errors_are_not_retried(self):
        for code in (301, 302, 400, 401, 403, 410, 422, 429):
            with self.subTest(code=code):
                github, clock, stderr = self.metadata_client()
                error = self.metadata_error(code)
                github.authenticated.open.side_effect = error
                with self.assertRaisesRegex(transport.TransportError, 'failed HTTP ' + str(code)):
                    github.document('/repos/o/r/actions/artifacts/81')
                self.assertEqual(github.authenticated.open.call_count, 1)
                self.assertEqual(clock.waits, [])
                self.assertTrue(error.closed)

    def test_invalid_metadata_is_not_retried_until_a_better_document_arrives(self):
        for raw in (b'not JSON', b'[]', b' ' * (1024 * 1024 + 1)):
            with self.subTest(bytes=len(raw)):
                github, clock, stderr = self.metadata_client()
                github.authenticated.open.return_value = MetadataResponse(raw)
                with self.assertRaises((transport.TransportError, ValueError)):
                    github.document('/repos/o/r/actions/artifacts/81')
                self.assertEqual(github.authenticated.open.call_count, 1)
                self.assertEqual(clock.waits, [])

    def test_other_repository_metadata_is_not_given_publication_retries(self):
        github, clock, stderr = self.metadata_client()
        github.authenticated.open.side_effect = self.metadata_error()
        with self.assertRaisesRegex(transport.TransportError, 'repository metadata request failed HTTP 404 after 1 attempt'):
            github.document('/repos/o/r/commits/main')
        self.assertEqual(github.authenticated.open.call_count, 1)
        self.assertEqual(clock.waits, [])

    def test_transport_network_errors_remain_redacted_and_fail_without_retries(self):
        for error in (urllib.error.URLError('private-test-token'), TimeoutError('private-test-token')):
            with self.subTest(error=type(error).__name__):
                github, clock, stderr = self.metadata_client()
                github.authenticated.open.side_effect = error
                with self.assertRaisesRegex(transport.TransportError, '^artifact metadata request failed$'):
                    github.document('/repos/o/r/actions/artifacts/81')
                self.assertEqual(github.authenticated.open.call_count, 1)
                self.assertEqual(clock.waits, [])

    def test_identity_mismatch_after_retry_still_rejects_the_artifact(self):
        merge, head = 'a' * 40, 'c' * 40
        args = argparse.Namespace(repository='o/r', run_id=71, run_attempt=1, artifact_id=81,
            checkout_sha=merge, pull_request_head_sha=head, artifact_digest='b' * 64,
            event_name='pull_request', expected_run_head_sha=head, artifact_bytes=123,
            helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest())
        artifact = dict(id=81, name=transport.ARTIFACT_NAME, expired=False,
            size_in_bytes=123, digest='sha256:' + 'b' * 64, created_at='created', expires_at='expires',
            workflow_run=dict(id=71, head_sha=head))
        run = dict(id=71, run_attempt=1, repository=dict(full_name='o/r'), event='pull_request', head_sha=head)
        for changed in ({'id': 82}, {'name': 'another-artifact'}, {'expired': True},
                        {'digest': 'sha256:' + 'd' * 64}, {'size_in_bytes': 124},
                        {'workflow_run': dict(id=72, head_sha=head)}):
            with self.subTest(changed=changed):
                github, clock, stderr = self.metadata_client()
                wrong = dict(artifact, **changed)
                github.authenticated.open.side_effect = [self.metadata_error(),
                    MetadataResponse(json.dumps(wrong).encode()), MetadataResponse(json.dumps(run).encode())]
                with self.assertRaises(transport.TransportError):
                    transport.identity(args, github)
                self.assertEqual(github.authenticated.open.call_count, 3)
                self.assertEqual(clock.waits, [1])


if __name__ == '__main__':
    unittest.main()
