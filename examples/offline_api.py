"""Check the existing protein API using the archived software-test FASTA fixture."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from ToolsAgent.ToolsFuns.Biology.main import app
    fixture = ROOT / 'tests/fixtures/fasta/multi_protein.fasta'
    if not fixture.is_file():
        raise FileNotFoundError(fixture)
    with TestClient(app) as client:
        response = client.post('/api/parse_multi_protein_fasta',
                               json={'file_path': str(fixture)})
        response.raise_for_status()
        parsed = response.json()
        if parsed.get('status') != 'success':
            raise RuntimeError('FASTA API did not return a successful collection')
        selected_response = client.post('/api/select_fasta_record', json={
            'collection': parsed['collection'], 'record_id': 'protein_b'})
        selected_response.raise_for_status()
        selected = selected_response.json()
        if selected.get('status') != 'success' or selected.get('record_id') != 'protein_b':
            raise RuntimeError('FASTA API did not preserve the requested record identity')
        terminal_response = client.post('/api/compute_gravy',
                                        json={'sequence': selected['sequence']})
        terminal_response.raise_for_status()
        terminal = terminal_response.json()
        if terminal.get('status') != 'success':
            raise RuntimeError('Sequence-level API did not return a successful result')
        print(json.dumps({'status': 'PASS', 'fixture': str(fixture.relative_to(ROOT)),
                          'record_id': selected['record_id'], 'response': terminal,
                          'external_model_calls': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
