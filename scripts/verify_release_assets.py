"""Verify copied release bytes, historical source/input hashes and official cores."""
from pathlib import Path
import hashlib,json,re,sys,subprocess

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
    'provider_token':rb'\bsk-[A-Za-z0-9_-]{24,}\b',
    'github_token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b',
    'aws_id':rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'credential_url':rb'https?://[^\s/:]+:[^\s/@]{8,}@',
}

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def resolve(relative):
    path=(ROOT/relative.replace('\\','/')).resolve();path.relative_to(ROOT.resolve());return path

def verify():
    manifest=read(ROOT/'RELEASE_MANIFEST.json');errors=[]
    expected_paths=set(manifest['files'])|{'RELEASE_MANIFEST.json'}
    tracked_paths=None
    if (ROOT/'.git').exists():
        try:
            result=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,check=True)
            tracked_paths=set(result.stdout.decode('utf-8').rstrip('\0').split('\0'))
            for relative in sorted(tracked_paths-expected_paths):errors.append('Tracked file outside manifest: '+relative)
            for relative in sorted(expected_paths-tracked_paths):errors.append('Manifested file not tracked: '+relative)
        except (OSError,subprocess.CalledProcessError,UnicodeError):
            errors.append('Cannot inspect checkout file coverage')
    def check_hash(relative,expected,label):
        try:
            p=resolve(relative)
            if not p.is_file() or sha(p)!=expected:errors.append(label+': '+relative)
        except (OSError,ValueError):errors.append(label+': '+relative)
    for relative,entry in manifest['files'].items():
        check_hash(relative,entry['sha256'],'Release checksum')
    frozen=read(ROOT/'evaluation/biotooleval/external_baselines_freeze_v3_1_resource.json')
    checked=0
    for relative,expected in frozen['code_sha256'].items():
        check_hash(relative,expected,'Frozen code')
        checked+=1
    data=frozen['dataset']
    for relative,expected in [(data['path'],data['raw_sha256']),(data['metadata_path'],data['metadata_sha256'])]+[(f['path'],f['raw_sha256']) for f in data['fixtures']]:
        check_hash(relative,expected,'Frozen benchmark')
        checked+=1
    official=0
    for index in (ROOT/'third_party').glob('*/UPSTREAM.json'):
        info=read(index)
        for file in info['files']:
            check_hash((index.parent/file['local_path']).relative_to(ROOT).as_posix(),file['sha256'],'Official core')
            official+=1
    findings=[]
    for relative in sorted(expected_paths|(tracked_paths or set())):
        try:data=resolve(relative).read_bytes()
        except (OSError,ValueError):
            errors.append('Cannot scan public asset: '+relative);continue
        for kind,pattern in PATTERNS.items():
            for match in re.finditer(pattern,data):
                if any(x in match.group(0).lower() for x in [b'placeholder',b'your-',b'example',b'xxxx',b'dummy',b'test-',b'fake-']):continue
                findings.append({'file':relative,'kind':kind})
    forbidden=['node_modules','.codex_tmp','docs/reviews','docs/submission','Cover_Letter','Title_Page','ORCID.docx','Author Agreement']
    for relative in expected_paths|(tracked_paths or set()):
        parts=Path(relative.replace('\\','/')).parts
        if any(term in relative for term in forbidden) or '.git' in parts or '__pycache__' in parts or any(part.startswith('.env') for part in parts):errors.append('Excluded material: '+relative)
    if findings:errors.append('Credential-pattern matches: '+json.dumps(findings))
    return {'status':'PASS' if not errors else 'FAIL','release_files':len(expected_paths),'historical_hash_checks':checked,'official_source_files':official,'credential_pattern_matches':len(findings),'git_manifest_coverage_checked':tracked_paths is not None,'errors':errors}

def main():
    try:report=verify()
    except (OSError,ValueError,KeyError) as exc:report={'status':'FAIL','errors':['Cannot read release evidence: '+type(exc).__name__]}
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
