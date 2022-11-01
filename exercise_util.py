import os
from pathlib import Path
import joblib
import contextlib
from tqdm.auto import tqdm
import pandas as pd
from arch.unitroot import ADF
import datetime
import re

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
        _target_pattern = f'incomplete-{symbol}-{datatype}-{interval}sec*'
        _result_list = [_ for _ in _p.glob(_target_pattern)]
        _target_pattern = f'temp-{symbol}-{datatype}-*'
        _result_list = _result_list + [_ for _ in _p.glob(_target_pattern)]
    else:
        _target_pattern = f'{symbol}-{datatype}-*'
        _result_list = [_ for _ in _p.glob(_target_pattern)]

    return sorted(_result_list)

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

# タイムバーファイルをロードしてすべて結合する関数
def concat_timebar_files(symbol: str = None, interval: int = None, from_str:str = None):
    assert symbol is not None
    assert interval is not None

    _d_today = datetime.date.today()
    
    if from_str is None:
        _p = Path(f'data/binance/timebar/{symbol}/{interval}')
        _list_trades_file = sorted(_p.glob(f'{symbol}-timebar-*'))
    else:
        _m = re.match('(\d{4})-(\d{2})-(\d{2})', from_str)
        _year = int(_m.group(1))
        _month = int(_m.group(2))
        _day = int(_m.group(3))

        _d_cursor = datetime.date(year = _year, month = _month, day = _day)
    
        _list_trades_file = []
        while _d_cursor < _d_today:
            _filename = f'data/binance/timebar/{symbol}/{interval}/{symbol}-timebar-{interval}sec-{_d_cursor.year:04}-{_d_cursor.month:02}-{_d_cursor.day:02}.pkl.gz'
            if Path(_filename).exists() == True:
                _list_trades_file.append(_filename)
            _d_cursor = _d_cursor + datetime.timedelta(days = 1)
        _list_trades_file = sorted(_list_trades_file)        
    
    def read_timebar(idx, filename):
        _df = pd.read_pickle(filename)
        return (idx, _df)
    
    with tqdm_joblib(total = len(_list_trades_file)):
        results = joblib.Parallel(n_jobs = -2, timeout = 60*60*24)([joblib.delayed(read_timebar)(_idx, _filename) for _idx, _filename in enumerate(_list_trades_file)])

    results.sort(key = lambda x: x[0])

    _list_timebar_df = []
    for _result in results:
        _list_timebar_df.append(_result[1])

    _df = pd.concat(_list_timebar_df, axis = 0)

    # すべてが0の行を取り除く
    _df = _df[(_df.T != 0).any()]
    return _df

# ADF検定を実施する関数
def adf_stationary_test(y: pd.Series = None):
    _r = ADF(y, low_memory = len(y) > 10_000)
    return _r.pvalue
