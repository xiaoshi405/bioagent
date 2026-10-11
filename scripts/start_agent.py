"""Launch the existing agent service without confusing app.py with app/."""
from pathlib import Path
import argparse
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--check-only', action='store_true',
                        help='Load the existing ASGI app without starting a server or calling a model.')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    service = runpy.run_path(str(ROOT / 'app.py'), run_name='bioagent_release_service')
    if args.check_only:
        print('Agent ASGI entry point loaded; no model request was made.')
        return 0
    import uvicorn
    uvicorn.run(service['app'], host=args.host,
                port=args.port if args.port is not None else service['_cfg'].SERVER_PORT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
