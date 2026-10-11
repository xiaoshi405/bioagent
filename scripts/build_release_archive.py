"""Build a verified source ZIP from exactly the release manifest's public files."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_release_assets import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/BioAgent-source.zip')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT)
    if 'outputs' not in output.relative_to(ROOT).parts or output.suffix.lower() != '.zip':
        parser.error('Write the archive under an outputs directory inside this repository.')
    if output.exists():
        parser.error('Output already exists; choose a new archive path.')
    report = verify()
    if report['status'] != 'PASS':
        print(json.dumps(report, indent=2))
        return 1
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    paths = sorted(set(manifest['files']) | {'RELEASE_MANIFEST.json'})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for relative in paths:
            info = zipfile.ZipInfo('bioagent/' + relative, date_time=(2026, 10, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(
        digest + '  ' + output.name + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'archive': str(output), 'sha256': digest,
                      'files': len(paths), 'bytes': output.stat().st_size}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
