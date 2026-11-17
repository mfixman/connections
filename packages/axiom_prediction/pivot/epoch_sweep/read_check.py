# Reads every file the given problems include, with a timeout per file, and prints those that hang and their Lustre targets.
import csv
import re
import subprocess
import sys
import time

tptp = '/rds/user/msf43/hpc-work/TPTP/'
rows = list(csv.DictReader(open(sys.argv[1])))
names = [r['problem'] for r in rows if r.get('error') == 'Timeout']
files = []
for name in names:
    path = tptp + 'Problems/' + name[:3] + '/' + name
    files.append(path)
    files += [tptp + m for m in re.findall(r"include\('([^']+)'", open(path).read())]

hung = []
for path in dict.fromkeys(files):
    target = subprocess.run(['lfs', 'getstripe', '-i', path], capture_output = True, text = True).stdout.strip()
    print(f'reading ost {target} {path}', flush = True)
    started = time.monotonic()
    done = subprocess.run(['timeout', '30', 'cat', path], stdout = subprocess.DEVNULL)
    seconds = time.monotonic() - started
    if done.returncode != 0 or seconds > 5:
        hung.append(path)
        print(f'SLOW {seconds:.1f}s exit {done.returncode} ost {target} {path}', flush = True)

print(f'{len(dict.fromkeys(files))} files read, {len(hung)} slow or hung', flush = True)
