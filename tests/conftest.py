import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ('src', 'scripts'):
    sys.path.insert(0, str(ROOT / sub))


def pytest_configure(config):
    config.addinivalue_line('markers', 'raw: сверка с сырыми JSONL (один проход по ~10 ГБ); '
                                       'пропустить — pytest -m "not raw"')
