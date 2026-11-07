"""Stop only the process tree of a verified launcher owned by this experiment."""
import argparse
import json
from pathlib import Path
import signal
import time
import psutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('pid', type=int)
parser.add_argument('run_suffix')
args = parser.parse_args()
process = psutil.Process(args.pid)
argv = process.cmdline()
if 'rlvr.launch' not in argv or args.run_suffix not in argv:
    raise ValueError('PID does not match the explicitly named RLVR launcher')
owned = process.children(recursive=True)
# Children are reached strictly through the verified driver ancestry.
for child in reversed(owned):
    try:
        child.send_signal(signal.SIGTERM)
    except psutil.NoSuchProcess:
        pass
process.send_signal(signal.SIGTERM)
gone, alive = psutil.wait_procs([*owned, process], timeout=10)
for child in alive:
    # All remaining PIDs were obtained from the same verified process tree.
    try:
        child.kill()
    except psutil.NoSuchProcess:
        pass
print(json.dumps({'launcher_pid': args.pid, 'owned_children': len(owned), 'terminated': len(gone)}))
