import pandas as pd
import numpy as np
import datetime
import time
import os
from pathlib import Path
import re
import requests
import zipfile
from io import BytesIO, StringIO
from joblib import Parallel, delayed
from retrying import retry
import argparse
from exercise_util import tqdm_joblib, identify_datafiles, target_symbols

# ファイル保存ディレクトリの中を見て、まだダウンロードしていないデータファイル名を返す関数
def identify_not_yet_downloaded_dates(symbol: str = None, datadir: str = None) -> set:
    assert symbol is not None
    assert datadir is not None

    _symbol = symbol.upper()
    _d_today = datetime.date.today()
    _d_cursor = datetime.date(year = 2019, month = 9, day = 8)
    if symbol in target_symbols:
        _initial_date = target_symbols[symbol]
        _d_cursor = datetime.date(year = _initial_date[0], month = _initial_date[1], day = _initial_date[2])
    
    Path(f'{datadir}/trades/{_symbol}').mkdir(parents = True, exist_ok = True)
    
    _set_all_filenames = set()
    while _d_cursor < _d_today:
        _filename = f'{_symbol}-trades-{_d_cursor.year:04}-{_d_cursor.month:02}-{_d_cursor.day:02}.zip'
        _set_all_filenames.add(_filename)
        _d_cursor = _d_cursor + datetime.timedelta(days = 1)
    
    _list_existing_files = identify_datafiles(datadir, 'trades', _symbol)
    _set_existing_filenames = set()
    for _existing_file in _list_existing_files:
        _filename = _existing_file.name
        _set_existing_filenames.add(_filename.replace('.pkl.gz', '.zip'))
    
    return sorted(_set_all_filenames - _set_existing_filenames)

# 指定されたファイル名をもとに、.zipをダウンロードしてデータフレームを作り、pkl.gzとして保存する関数
@retry(stop_max_attempt_number = 5, wait_fixed = 1000)
def download_trade_zip(target_file_name: str = None, datadir: str = None) -> None:
    assert str is not None
    assert datadir is not None
    
    _m = re.match('(.+)-trades.*', target_file_name)
    _symbol = _m.group(1)
    _stem = Path(target_file_name).stem
    
    _url = f'https://data.binance.vision/data/futures/um/daily/trades/{_symbol}/{target_file_name}'
    
    _r = requests.get(_url)
    if _r.status_code != requests.codes.ok:
        print(f'response.get({_url})からHTTPステータスコード {_r.status_code} が返されました。このファイルをスキップします。')
        time.sleep(1)
        return
    
    _csvzip = zipfile.ZipFile(BytesIO(_r.content))
    if _csvzip.testzip() != None:
        print(f'Corrupt zip file from {_url}. Retry.')
        raise Exception
    _csvraw = _csvzip.read(f'{_stem}.csv')
    
    if chr(_csvraw[0]) == 'i':
        # ヘッダーラインがあるので削除しないといけない
        _header = 0
    else:
        _header = None
    
    try:
        _df = pd.read_csv(BytesIO(_csvraw), names = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker'], dtype = {0: int, 1: float, 2: float, 3: float, 4: float, 5: bool}, header = _header)
    except Exception as e:
        print(f'pd.read_csv({_url})が例外 {e} を返しました。リトライします。')
        raise e
        
    _df['time'] = pd.to_datetime(_df['time'] , unit = 'ms')
    _df.to_pickle(f'{datadir}/trades/{_symbol}/temp-{_stem}.pkl.gz')
    _tempfile = Path(f'{datadir}/trades/{_symbol}/temp-{_stem}.pkl.gz')
    _tempfile.rename(f'{datadir}/trades/{_symbol}/{_stem}.pkl.gz')

    return

# joblibを使って4並列でダウンロードジョブを実行する関数
def download_trade_from_binance(symbol: str = None) -> None:
    assert symbol is not None
    
    _datadir = 'data/binance'    
    _symbol = symbol.upper()

    # 処理開始前に全ての未完了ファイルを削除する
    _list_incomplete_files = identify_datafiles(_datadir, 'trades', _symbol, incomplete = True)
    for _incomplete_file in _list_incomplete_files:
        _incomplete_file.unlink()

    _set_target_files = identify_not_yet_downloaded_dates(_symbol, _datadir)  
    _num_files = len(_set_target_files)
    print(f'{symbol}の約定履歴ファイルを{_num_files}個ダウンロードします')
    
    with tqdm_joblib(total = _num_files):
        r = Parallel(n_jobs = -1, timeout = 60*60*24)([delayed(download_trade_zip)(_f, _datadir) for _f in _set_target_files])

    # 処理開始後に全ての未完了ファイルを削除する
    _list_incomplete_files = identify_datafiles(_datadir, 'trades', _symbol, incomplete = True)
    for _incomplete_file in _list_incomplete_files:
        _incomplete_file.unlink()

    return

# 引数処理とダウンロード関数の起動部分
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', help = 'ダウンロードする対象の銘柄 例:BTCUSDT')
    args = parser.parse_args()

    symbol = args.symbol
    if symbol:
        download_trade_from_binance(symbol)
    else:
        for _symbol in target_symbols.keys():
            download_trade_from_binance(_symbol)
