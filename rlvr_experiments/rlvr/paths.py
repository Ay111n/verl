"""Explicit local/server paths. No data, models or contracts are generated here."""
from pathlib import Path,PurePosixPath
import json,os,posixpath
PROJECT_ROOT=Path(__file__).resolve().parent.parent
def load_paths(profile=None,config_file=None):
    profile=profile or os.environ.get('RLVR_PATH_PROFILE') or ('local' if os.name=='nt' else 'server')
    file=Path(config_file or os.environ.get('RLVR_PATHS_FILE',PROJECT_ROOT/'configs/paths.json'))
    doc=json.loads(file.read_text(encoding='utf-8'))
    if doc.get('schema')!='rlvr-paths-v1' or profile not in doc['profiles']:
        raise ValueError(f'Unknown path profile: {profile}')
    raw=doc['profiles'][profile]
    remote_preview=profile=='server' and os.name=='nt'
    root=PurePosixPath('/home/nlp/data/lyl/verl/rlvr_experiments') if remote_preview else PROJECT_ROOT
    result={'profile':profile,'project_dir':root}
    for key,value in raw.items():
        if key=='evaluation_status':
            if value not in ('active','retired'):raise ValueError(f'Invalid evaluation status: {value}')
            result[key]=value
            continue
        if value is None:result[key]=None;continue
        for previous,resolved in result.items():
            value=value.replace('${'+previous+'}',str(resolved))
        if '${' in value:raise ValueError(f'Unresolved path: {key}')
        if remote_preview:
            p=PurePosixPath(value)
            result[key]=PurePosixPath(posixpath.normpath(str(p if p.is_absolute() else root/p)))
            continue
        if key=='python':result[key]=Path(value);continue
        p=Path(value);result[key]=p.resolve() if p.is_absolute() else (PROJECT_ROOT/p).resolve()
    return result
def require_path(paths,key):
    p=paths.get(key)
    if p is None or not p.exists():raise FileNotFoundError(f'Missing configured {key}: {p}; configure paths.json or RLVR_PATHS_FILE')
    return p
