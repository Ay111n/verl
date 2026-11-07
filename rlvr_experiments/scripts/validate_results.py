"""Validate explicitly supplied future results; old reward results were deleted."""
import argparse,json,csv
from pathlib import Path
def validate(root):
    root=Path(root)
    if not root.is_dir():raise FileNotFoundError(f'Results not present: {root}')
    result={}
    for name in ('R0',):
        run=root/name
        if not (run/'status.json').is_file():raise FileNotFoundError(f'Missing result: {run}')
        if json.loads((run/'status.json').read_text())['status']!='complete':raise ValueError(f'Incomplete: {run}')
        updates=[json.loads(s) for s in (run/'updates.jsonl').read_text().splitlines() if s.strip()]
        if len(updates)!=50 or any(not u['finite_update'] or u['optimizer_steps_by_rank']!={'0':1,'1':1} for u in updates):raise ValueError('Invalid optimizer exposure')
        with (run/'train_exposure_manifest.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        if len(rows)!=1600 or len({r['sample_id'] for r in rows})!=1600 or any(int(r['generation_count'])!=8 for r in rows):raise ValueError('Invalid sample exposure')
        result[name]={'updates':50,'prompts':1600,'generation_count':8}
    return {'scope':'status and exposure only; use reward/report audits for scientific conclusions','experiments':result}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-root',type=Path,required=True)
    print(json.dumps(validate(p.parse_args().results_root),ensure_ascii=False,indent=2))
