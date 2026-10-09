"""
Последовательно обучает одну модель с несколькими сидами.

Берёт configs/<dataset>/<model>.yaml, для каждого сида пишет временную копию конфига
с подменёнными seed и output.run_name (= <model>_<seed>) и запускает src/train.py.
Прогоны разных датасетов не пересекаются: у каждого датасета свой output.dir в конфигах.

    python scripts/run_seeds.py --dataset toys --model dreamrec --n-seeds 5
    python scripts/run_seeds.py --dataset toys --model dreamrec_text --n-seeds 3 --start-seed 10
    python scripts/run_seeds.py --dataset atg --model sasrec --seeds 1 7 42
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description='Обучение модели с несколькими сидами.')
    datasets = sorted(p.name for p in (ROOT / 'configs').iterdir() if p.is_dir())
    parser.add_argument('--dataset', required=True, choices=datasets, help='папка конфигов: configs/<dataset>/')
    parser.add_argument('--model', required=True, help='имя конфига: configs/<dataset>/<model>.yaml')
    parser.add_argument('--n-seeds', type=int, default=None, help='сколько сидов прогнать')
    parser.add_argument('--start-seed', type=int, default=0, help='первый сид при --n-seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='явный список сидов')
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = args.seeds
    elif args.n_seeds is not None:
        seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    else:
        parser.error('укажите --n-seeds или --seeds')

    config_path = ROOT / 'configs' / args.dataset / f'{args.model}.yaml'
    base_cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))

    failed = []
    for seed in seeds:
        cfg = dict(base_cfg, seed=seed)
        cfg['output'] = dict(base_cfg['output'], run_name=f'{args.model}_{seed}')

        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            tmp_config = f.name

        print(f'\n===== {args.dataset}/{args.model}: seed {seed} -> run_name {cfg["output"]["run_name"]} =====', flush=True)
        try:
            result = subprocess.run([sys.executable, 'src/train.py', '--config', tmp_config], cwd=ROOT)
        finally:
            Path(tmp_config).unlink(missing_ok=True)
        if result.returncode != 0:
            failed.append(seed)
            print(f'!!! seed {seed} завершился с кодом {result.returncode}', flush=True)

    print(f'\nГотово. Успешно: {[s for s in seeds if s not in failed]}, упали: {failed}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
