# Uploads the epoch-sweep schedule results (rows.csv) into each model's W&B training run, one point per epoch.
# usage: epoch_sweep_upload.py ROWS_CSV MODEL=RUN_ID...
from collections import defaultdict
import csv
from pathlib import Path
import sys

from axiom_prediction.wandb_tracking import WandbConfig

def read_rows(path):
    epochs = defaultdict(lambda: defaultdict(dict))
    for row in csv.DictReader(open(path)):
        epochs[row['model']][int(row['epoch'])][(row['prover'], row['schedule'])] = row

    return epochs

def epoch_metrics(rows):
    metrics = {}
    for (prover, schedule), row in rows.items():
        files, plain = int(row['files']), int(rows[(prover, 'Plain')]['files'])
        name = f'epoch_sweep/{prover}/{schedule}'
        metrics[f'{name}/files'] = files
        metrics[f'{name}/families'] = int(row['families'])
        metrics[f'{name}/gain_pct'] = 100 * (files - plain) / plain

    return metrics

def upload(run_id, epochs, source):
    import wandb

    config = WandbConfig()
    wandb.login(key = config.key_file.expanduser().read_text(encoding = 'utf-8').strip())
    run = wandb.init(entity = config.entity, project = config.project, id = run_id, resume = 'must', dir = '/dev/shm')
    run.define_metric('epoch')
    run.define_metric('epoch_sweep/*', step_metric = 'epoch')
    for epoch, rows in sorted(epochs.items()):
        run.log({'epoch': epoch, **epoch_metrics(rows)})

    run.config.update({'epoch_sweep_backfill': {'source': source, 'epochs': sorted(epochs)}}, allow_val_change = True)
    run.finish()
    print(f'{run_id}: uploaded {len(epochs)} epochs', flush = True)

def main():
    source, targets = sys.argv[1], dict(target.split('=') for target in sys.argv[2:])
    epochs = read_rows(source)
    for model, run_id in targets.items():
        upload(run_id, epochs[model], source)

if __name__ == '__main__':
    main()
