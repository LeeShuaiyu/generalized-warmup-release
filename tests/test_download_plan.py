import json
from pathlib import Path
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gw.download_plan import experiments, plan
from gw.public_downloads import HTTPSRedirect, check
import io
import urllib.request
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class DownloadPlan(unittest.TestCase):
    def test_results_only_is_not_all_weights(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            result = plan(ROOT, results_only=True)
        self.assertEqual([r['file'] for r in result['files']], ['gw-results.zip'])
        self.assertEqual(result['archive_bytes'], 305223459)
        self.assertEqual(result['selected_models'], 0)

    def test_every_experiment_only_includes_intersecting_parts(self):
        manifest = json.loads((ROOT / 'artifacts/manifest.json').read_text())['checkpoints']
        parts = json.loads((ROOT / 'artifacts/model_packages.json').read_text())['parts']
        for experiment in experiments(ROOT):
            with self.subTest(experiment=experiment):
                result = plan(ROOT, experiment)
                selected = {r['destination'] for k, r in manifest.items() if k.split('/')[0] == experiment}
                expected = {r['file'] for r in parts
                            if any(m['destination'] in selected for m in r['models'])}
                self.assertEqual({r['file'] for r in result['files']}, expected)
                self.assertEqual(result['archive_bytes'], sum(r['bytes'] for r in parts if r['file'] in expected))
                self.assertEqual(result['selected_models'], len(selected))
                self.assertLess(len(expected), len(parts))
                for part in parts:
                    if part['file'] in expected:
                        self.assertTrue({m['destination'] for m in part['models']}.issubset(selected))
                urls = json.loads((ROOT/'artifacts/model_urls.json').read_text())
                self.assertTrue(all(r['url'] == urls.get(r['file']) for r in result['files']))

    def test_combined_plan_deduplicates_result_and_models(self):
        result = plan(ROOT, 's4_family', include_results=True)
        self.assertEqual(result['files'][0]['file'], 'gw-results.zip')
        self.assertEqual(len(result['files']), len({r['file'] for r in result['files']}))

    def test_explicit_scope_required(self):
        with self.assertRaises(ValueError):
            plan(ROOT)
        with self.assertRaises(ValueError):
            plan(ROOT, 'not_an_experiment')


class PublicGate(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='gw-public-gate-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copytree(ROOT/'artifacts', self.root/'artifacts')
        catalog = json.loads((self.root/'artifacts/downloads.json').read_text())
        parts = json.loads((self.root/'artifacts/model_packages.json').read_text())['parts']
        host = catalog['hosting']
        base = f"https://huggingface.co/{host['repository']}/resolve/{host['revision']}/"
        (self.root/'artifacts/model_urls.json').write_text(json.dumps({r['file']:base+r['file'] for r in parts}))

    def response(self, body=b'{}', headers=None):
        stream = io.BytesIO(body)
        stream.headers = headers or {}
        stream.status = 200
        return stream

    def test_private_metadata_blocks_without_checking_files(self):
        with patch('gw.public_downloads.open_anonymous', return_value=self.response(b'{"private":true}')) as call:
            with self.assertRaises(ValueError):
                check(self.root)
        self.assertEqual(call.call_count, 1)

    def test_wrong_revision_blocks(self):
        with patch('gw.public_downloads.open_anonymous', return_value=self.response(b'{"private":false,"sha":"wrong"}')):
            with self.assertRaises(ValueError):
                check(self.root)

    def test_all_precise_sizes_and_hashes_required(self):
        catalog = json.loads((ROOT / 'artifacts/downloads.json').read_text())
        parts = json.loads((ROOT / 'artifacts/model_packages.json').read_text())['parts']
        expected = [catalog['results'], *parts]
        revision = catalog['hosting']['revision']
        def response(url, method='GET'):
            if method == 'GET':
                return self.response(json.dumps(dict(private=False, sha=revision)).encode())
            row = next(r for r in expected if url.endswith('/' + r['file']))
            return self.response(headers={'Content-Length': str(row['bytes']), 'ETag': '"'+row['sha256']+'"'})
        with patch('gw.public_downloads.open_anonymous', side_effect=response):
            result = check(self.root)
        self.assertEqual(result['formal_files'], len(expected))
        self.assertEqual(result['formal_bytes'], sum(r['bytes'] for r in expected))
        self.assertFalse(result['full_download_restore_tested'])

    def test_missing_hash_is_not_success(self):
        hosting = json.loads((ROOT / 'artifacts/downloads.json').read_text())['hosting']
        replies = [self.response(json.dumps(dict(private=False, sha=hosting['revision'])).encode()),
                   self.response(headers={'Content-Length': '305223459'})]
        with patch('gw.public_downloads.open_anonymous', side_effect=replies):
            with self.assertRaises(ValueError):
                check(self.root)

    def test_redirect_cannot_downgrade_tls(self):
        request = urllib.request.Request('https://huggingface.co/example/resolve/revision/file')
        with self.assertRaises(ValueError):
            HTTPSRedirect().redirect_request(request, None, 302, '', {}, 'http://example.org/file')
