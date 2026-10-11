import argparse
import json
from pathlib import Path
import shutil
import sys

repo = Path('/home/msf43/tmp/connections-mizar-weekend-40336e3')
sys.path[:0] = [str(repo / 'src'), str(repo / 'packages')]

from axiom_prediction.choices import ProverPolicy
from axiom_prediction.dataset import collect_axiom_dataset_shard, problem_key
from axiom_prediction.split import ProblemSplit

results = Path('/home/msf43/connections/results/weekend-mizar')
source_dir = results / 'data/mizar40/problems_small_consist'
dataset_dir = results / 'Mizar40-2m/dataset'
old_cache = results / 'Mizar40/dataset/.cache/Mizar40'
timeout_s = 120
step_limit = 1_000_000_000
tptp_root = '/rds/user/msf43/hpc-work/TPTP'
num_workers = 76


def problem_list():
    entries = [line.strip() for line in (source_dir / '00all.p').read_text().splitlines() if line.strip()]
    return [str(source_dir / f'{entry}.p') for entry in entries]


def seed_cache():
    if dataset_dir.exists():
        raise FileExistsError(f'refusing to replace existing output directory {dataset_dir}')
    dataset_dir.mkdir(parents = True)
    archive = dataset_dir / '.seeded-900s-cache'
    shutil.copytree(old_cache, archive)
    problems = set(problem_list())
    split = ProblemSplit(4, [0])
    copied_failures = [0] * 4
    rechecked_examples = [0] * 4
    for source_kind, category in (
        ('failures', 'failures'),
        ('partial/failures', 'partial/failures'),
        ('examples', None),
        ('partial/examples', None),
    ):
        source = old_cache / source_kind
        if not source.is_dir():
            continue
        for record_path in source.glob('*.json'):
            record = json.loads(record_path.read_text())
            problem = record.get('problem_path') or record.get('problem')
            if problem not in problems:
                continue
            part = ProblemSplit(4, [0]).part(problem)
            if category is None:
                rechecked_examples[part] += 1
                continue
            target_dir = dataset_dir / '.cache' / f'Mizar40-2m-part{part}' / category
            target_dir.mkdir(parents = True, exist_ok = True)
            shutil.copy2(record_path, target_dir / f'{problem_key(problem)}.json')
            copied_failures[part] += 1
    manifest = {
        'source_cache': str(old_cache),
        'source_timeout_seconds': 900,
        'target_timeout_seconds': timeout_s,
        'copied_failure_records_by_part': copied_failures,
        'rechecked_success_records_by_part': rechecked_examples,
        'seeded_cache_archive': str(archive),
    }
    (dataset_dir / 'seed-manifest.json').write_text(json.dumps(manifest, indent = 2) + '\n')
    print(json.dumps(manifest, sort_keys = True))


def collect_part(part):
    problems = ProblemSplit(4, [part]).select(problem_list())
    shard, summary = collect_axiom_dataset_shard(
        problems,
        output_dir = dataset_dir,
        shard_name = f'Mizar40-2m-part{part}',
        tptp_root = tptp_root,
        step_limit = step_limit,
        timeout_s = timeout_s,
        sat_policy = ProverPolicy.SatResetCoP,
        num_workers = num_workers,
    )
    print(json.dumps({**summary, 'dataset_shard': str(shard)}, sort_keys = True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices = ('prepare', 'collect'))
    parser.add_argument('--part', type = int, choices = range(4))
    args = parser.parse_args()
    if args.action == 'prepare':
        seed_cache()
    elif args.part is None:
        parser.error('collect requires --part 0, 1, 2, or 3')
    else:
        collect_part(args.part)


if __name__ == '__main__':
    main()
