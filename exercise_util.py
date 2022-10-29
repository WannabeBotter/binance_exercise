import os
from pathlib import Path

import joblib
import contextlib

from tqdm.auto import tqdm

target_symbols = {
    'BTCUSDT': (2019, 9, 8),
    'ETHUSDT': (2019, 11, 27),
    'XRPUSDT': (2020, 1, 6),
    'BNBUSDT': (2020, 2, 10),
    'ADAUSDT': (2020, 1, 31),
    'SOLUSDT': (2020, 9, 14),
    'DOGEUSDT': (2020, 7, 10),
    'MATICUSDT': (2020, 10, 22),
    'AVAXUSDT': (2020, 9, 23),
    '1000SHIBUSDT': (2021, 5, 10),
    'ATOMUSDT': (2020, 2, 7),
}


# データ保存ディレクトリの中のデータファイル一覧を返すユーティリティ関数
def identify_datafiles(datadir: str = None, datatype: str = None, symbol: str = None, interval: int = None, incomplete: bool = False):
    assert datadir is not None
    assert datatype is not None
    assert symbol is not None
    
    if interval is None:
        _p = Path(f'{datadir}/{datatype}/{symbol}')
    else:
        _p = Path(f'{datadir}/{datatype}/{symbol}/{interval}')
    
    if incomplete == True:
        _target_pattern = f'incomplete-{symbol}-{datatype}-*'
        _result_list = [_ for _ in _p.glob(_target_pattern)]
        _target_pattern = f'temp-{symbol}-{datatype}-*'
        _result_list = _result_list + [_ for _ in _p.glob(_target_pattern)]
    else:
        _target_pattern = f'{symbol}-{datatype}-*'
        _result_list = [_ for _ in _p.glob(_target_pattern)]

    return sorted([_ for _ in _p.glob(_target_pattern)])

# joblibの並列処理のプログレスバーを表示するためのユーティリティ関数
# https://blog.ysk.im/x/joblib-with-progress-bar
@contextlib.contextmanager
def tqdm_joblib(total: int = None, **kwargs):

    pbar = tqdm(total = total, miniters = 1, smoothing = 0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n = self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()