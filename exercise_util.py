import os
from pathlib import Path

import joblib
import contextlib

from tqdm.auto import tqdm

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
    else:
        _target_pattern = f'{symbol}-{datatype}-*'

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