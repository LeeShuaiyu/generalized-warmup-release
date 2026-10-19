"""Check anonymous HF access and file identities without changing release state."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gw.public_downloads import check


if __name__ == '__main__':
    try:
        print(json.dumps(check(ROOT), indent=2))
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
