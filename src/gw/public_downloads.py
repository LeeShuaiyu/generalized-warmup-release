"""Read-only anonymous HF release gate. Never uploads or changes visibility."""
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != 'https':
            raise ValueError('Refusing a non-HTTPS redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_anonymous(url, method='GET'):
    if urllib.parse.urlsplit(url).scheme != 'https':
        raise ValueError('Anonymous checks require HTTPS')
    # Fresh opener: no cookie jar, account configuration or Authorization header.
    opener = urllib.request.build_opener(HTTPSRedirect())
    try:
        return opener.open(urllib.request.Request(url, method=method), timeout=30)
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        raise RuntimeError(f'Anonymous HF check denied (HTTP {code})') from None
    except urllib.error.URLError:
        raise RuntimeError('Anonymous HF connection failed') from None


def check(root):
    root = Path(root)
    catalog = json.loads((root / 'artifacts/downloads.json').read_text())
    hosting = catalog.get('hosting', {})
    repo, revision = hosting.get('repository', ''), hosting.get('revision', '')
    if (hosting.get('provider') != 'huggingface' or
            not re.fullmatch(r'[\w.-]+/[\w.-]+', repo) or
            not re.fullmatch(r'[0-9a-f]{40}', revision)):
        raise ValueError('Expected fixed HF repository and full commit revision')
    with open_anonymous(f'https://huggingface.co/api/models/{repo}/revision/{revision}') as response:
        metadata = json.load(response)
    if metadata.get('private') is not False or metadata.get('sha') != revision:
        raise ValueError('Expected public repository metadata at the registered revision')
    parts = json.loads((root / 'artifacts/model_packages.json').read_text())['parts']
    urls = json.loads((root / 'artifacts/model_urls.json').read_text())
    if set(urls) != {row['file'] for row in parts}:
        raise ValueError('Model URL catalog has missing or unexpected filenames')
    expected = [catalog['results'], *parts]
    base = f'https://huggingface.co/{repo}/resolve/{revision}/'
    verified = []
    for row in expected:
        url = row.get('url') if row is catalog['results'] else urls.get(row['file'])
        if url != base + row['file']:
            raise ValueError('Asset URL differs from fixed audited revision')
        with open_anonymous(url, 'HEAD') as response:
            headers = response.headers
            size = headers.get('X-Linked-Size', headers.get('Content-Length', ''))
            digest = headers.get('X-Linked-Etag', headers.get('ETag', '')).strip('"')
            if (response.status != 200 or str(size) != str(row['bytes']) or
                    digest != row['sha256']):
                raise ValueError('Anonymous remote size/SHA256 not established: ' + row['file'])
        verified.append(dict(file=row['file'], bytes=row['bytes'], sha256=row['sha256']))
    return dict(repository=repo, revision=revision, private=False,
                anonymous_head_verified=True, formal_files=len(verified),
                formal_bytes=sum(row['bytes'] for row in verified), files=verified,
                full_download_restore_tested=False)
