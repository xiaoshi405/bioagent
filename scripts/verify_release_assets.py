"""Verify copied release bytes, historical source/input hashes and official cores."""
from pathlib import Path
import hashlib,json,re,sys

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

def main():
    manifest=read(ROOT/'RELEASE_MANIFEST.json');errors=[]
    for relative,entry in manifest['files'].items():
        p=resolve(relative)
        if not p.is_file() or sha(p)!=entry['sha256']:errors.append('Release checksum: '+relative)
    frozen=read(ROOT/'evaluation/biotooleval/external_baselines_freeze_v3_1_resource.json')
    checked=0
    for relative,expected in frozen['code_sha256'].items():
        if sha(resolve(relative))!=expected:errors.append('Frozen code: '+relative)
        checked+=1
    data=frozen['dataset']
    for relative,expected in [(data['path'],data['raw_sha256']),(data['metadata_path'],data['metadata_sha256'])]+[(f['path'],f['raw_sha256']) for f in data['fixtures']]:
        if sha(resolve(relative))!=expected:errors.append('Frozen benchmark: '+relative)
        checked+=1
    official=0
    for index in (ROOT/'third_party').glob('*/UPSTREAM.json'):
        info=read(index)
        for file in info['files']:
            if sha(index.parent/file['local_path'])!=file['sha256']:errors.append('Official core: '+str(index.parent/file['local_path']))
            official+=1
    findings=[]
    for relative in manifest['files']:
        p=resolve(relative);data=p.read_bytes()
        for kind,pattern in PATTERNS.items():
            for match in re.finditer(pattern,data):
                if any(x in match.group(0).lower() for x in [b'placeholder',b'your-',b'example',b'xxxx',b'dummy',b'test-',b'fake-']):continue
                findings.append({'file':relative,'kind':kind})
    if (ROOT/'.env').exists():errors.append('A local .env is present; not a release asset')
    forbidden=['node_modules','.codex_tmp','docs/reviews','docs/submission','Cover_Letter','Title_Page','ORCID.docx','Author Agreement']
    for relative in manifest['files']:
        if any(term in relative for term in forbidden):errors.append('Excluded material: '+relative)
    if findings:errors.append('Credential-pattern matches: '+json.dumps(findings))
    print(json.dumps({'status':'PASS' if not errors else 'FAIL','release_files':len(manifest['files']),'historical_hash_checks':checked,'official_source_files':official,'credential_pattern_matches':len(findings),'errors':errors},indent=2))
    return 1 if errors else 0

if __name__=='__main__':raise SystemExit(main())
