import pandas as pd
import numpy as np
import numpy.ma as ma
from numpy.lib.stride_tricks import sliding_window_view

import matplotlib.pyplot as plt
import japanize_matplotlib

from pathlib import Path

import joblib
import re
import os

import datetime

import argparse

from exercise_util import tqdm_joblib, identify_datafiles

datadir = 'data/binance'

def calc_timebar_from_trades(idx, filename, interval):
    _interval_str = f'{interval}S'

    _m = re.match('(.+)/trades/(.+?)/.*-trades-(\d{4})-(\d{2})-(\d{2})\.pkl\.gz', filename)
    _datadir = _m.group(1)
    _symbol = _m.group(2)
    _year = _m.group(3)
    _month = _m.group(4)
    _day = _m.group(5)
    _datetime_from = datetime.datetime(year = int(_year), month = int(_month), day = int(_day), hour = 0, minute = 0, second = 0)
    _datetime_to = datetime.datetime(year = int(_year), month = int(_month), day = int(_day), hour = 23, minute = 59, second = 59, microsecond = 999999)

    _df = pd.read_pickle(filename).set_index('time', drop = True)

    _df_timebar = _df['price'].resample(_interval_str, closed = 'left').ohlc()
    _df_timebar['buy_trade_count'] = _df.loc[_df['is_buyer_maker'] == False, 'is_buyer_maker'].resample(f'{interval}S').count().astype(int)
    _df_timebar['sell_trade_count'] = _df.loc[_df['is_buyer_maker'] == True, 'is_buyer_maker'].resample(f'{interval}S').count().astype(int)
    _df_timebar['quote_buy_volume'] = _df.loc[_df['is_buyer_maker'] == False, 'quote_qty'].resample(f'{interval}S').sum().astype(float)
    _df_timebar['quote_sell_volume'] = _df.loc[_df['is_buyer_maker'] == True, 'quote_qty'].resample(f'{interval}S').sum().astype(float)
    _df_timebar = _df_timebar.reindex(pd.date_range(_datetime_from, _datetime_to, freq = _interval_str, inclusive = 'both'))

    _df_timebar['close'] = _df_timebar['close'].fillna(method = 'ffill')
    _df_timebar['open'] = _df_timebar['open'].fillna(_df_timebar['close'])
    _df_timebar['high'] = _df_timebar['high'].fillna(_df_timebar['close'])
    _df_timebar['low'] = _df_timebar['low'].fillna(_df_timebar['close'])
    _df_timebar['buy_trade_count'] = _df_timebar['buy_trade_count'].fillna(0).astype(int)
    _df_timebar['sell_trade_count'] = _df_timebar['sell_trade_count'].fillna(0).astype(int)
    _df_timebar['quote_buy_volume'] = _df_timebar['quote_buy_volume'].fillna(0).astype(float)
    _df_timebar['quote_sell_volume'] = _df_timebar['quote_sell_volume'].fillna(0).astype(float)
    
    Path(f'{_datadir}/timebar/{_symbol}/{interval}').mkdir(parents = True, exist_ok = True)

    # 1行目のOpenがNaNの場合は、全ての時間足ファイルの生成が終わってから前日Closeを使ってOpenを埋める必要があるので、ファイル名でマークしておく
    if pd.isna(_df_timebar.iloc[0, _df_timebar.columns.get_loc('open')]) == True:
        _pickle_filename = f'{_datadir}/timebar/{_symbol}/{interval}/incomplete-{_symbol}-timebar-{interval}sec-{_year}-{_month}-{_day}.pkl.gz'
    else:
        _pickle_filename = f'{_datadir}/timebar/{_symbol}/{interval}/{_symbol}-timebar-{interval}sec-{_year}-{_month}-{_day}.pkl.gz'
    _df_timebar.to_pickle(_pickle_filename)

    return idx

# ファイル保存ディレクトリの中を見て、まだタイムバーを生成していない日の約定履歴データファイル名を返す関数
def identify_available_trades_files(datadir: str = None, symbol: str = None, interval: int = None) -> set:
    assert symbol is not None
    assert datadir is not None
    assert interval is not None
    
    _symbol = symbol.upper()
        
    Path(f'{datadir}/trades/{_symbol}').mkdir(parents = True, exist_ok = True)
    _p = Path(f'{datadir}/trades/{_symbol}')    

    _set_existing_trades_filenames = set([str(_.resolve().relative_to(Path.cwd())) for _ in identify_datafiles(datadir, 'trades', _symbol)])
    
    _p = Path(f'{datadir}/timebar/{_symbol}/{interval}')    
    _p.mkdir(parents = True, exist_ok = True)
    
    _set_unnecessray_trades_filenames = set()
    _list_existing_timebar_filenames = identify_datafiles(datadir, 'timebar', _symbol, interval)
    for _existing_timebar_filename in _list_existing_timebar_filenames:
        _stem = Path(_existing_timebar_filename).stem
        _m = re.match('.*-(\d{4})-(\d{2})-(\d{2}).*', _stem)
        _year = _m.group(1)
        _month = _m.group(2)
        _day = _m.group(3)
        _set_unnecessray_trades_filenames.add(f'{datadir}/trades/{_symbol}/{_symbol}-trades-{_year}-{_month}-{_day}.pkl.gz')
    
    return sorted(_set_existing_trades_filenames - _set_unnecessray_trades_filenames)

# Incompleteなタイムバーファイルを完成させる関数
def finish_incomplete_timebar_files(idx, filename, interval):
    assert idx is not None
    assert filename is not None
    assert interval is not None

    _df_incomplete = pd.read_pickle(filename)

    _m = re.match('(.+)/timebar/(.+?)/(\d+?)/incomplete-.+?-timebar-.*-(\d{4})-(\d{2})-(\d{2})\.pkl\.gz', filename)
    _datadir = _m.group(1)
    _symbol = _m.group(2)
    _interval = _m.group(3)
    _year = _m.group(4)
    _month = _m.group(5)
    _day = _m.group(6)

    _target_date = datetime.date(year = int(_year), month = int(_month), day = int(_day))
    _previous_date = _target_date - datetime.timedelta(days = 1)
    
    _previous_completed_file = Path(f'{_datadir}/timebar/{_symbol}/{_interval}/{_symbol}-timebar-{_interval}sec-{_previous_date.year:04}-{_previous_date.month:02}-{_previous_date.day:02}.pkl.gz')
    _previous_incomplete_file = Path(f'{_datadir}/timebar/{_symbol}/{_interval}/incomplete-{_symbol}-timebar-{_interval}sec-{_previous_date.year:04}-{_previous_date.month:02}-{_previous_date.day:02}.pkl.gz')

    _target_file = None
    if _previous_completed_file.exists() == True:
        _target_file = _previous_completed_file
    elif _previous_incomplete_file.exists() == True:
        _target_file = _previous_incomplete_file

    if _target_file is not None:
        try:
            _df_previous_date = pd.read_pickle(str(_target_file))
        except Exception as e:
            print(f'ファイル {_target_file}を読み込み中に例外{e}が発生しました')
            raise e
        
        _last_close = _df_previous_date.iloc[-1, _df_previous_date.columns.get_loc('close')]
    else:
        # このファイルがこの銘柄の最初の日の記録なので、最終クローズは0とする
        _last_close = 0.0

    _list_target_columns = [_df_incomplete.columns.get_loc(_) for _ in ['open', 'high', 'low', 'close']]
    _df_incomplete.iloc[0, _list_target_columns] = _last_close
    _df_incomplete['close'] = _df_incomplete['close'].fillna(method = 'ffill')
    _df_incomplete['open'] = _df_incomplete['open'].fillna(_df_incomplete['close'])
    _df_incomplete['high'] = _df_incomplete['high'].fillna(_df_incomplete['close'])
    _df_incomplete['low'] = _df_incomplete['low'].fillna(_df_incomplete['close'])

    # 並列処理している他のプロセスが書き込み途中のファイルを読み込まないように、一時ファイルに保存する
    _df_incomplete.to_pickle(f'{_datadir}/timebar/{_symbol}/{_interval}/temp-{_symbol}-timebar-{_interval}sec-{_target_date.year:04}-{_target_date.month:02}-{_target_date.day:02}.pkl.gz')
    _tempfile = Path(f'{_datadir}/timebar/{_symbol}/{_interval}/temp-{_symbol}-timebar-{_interval}sec-{_target_date.year:04}-{_target_date.month:02}-{_target_date.day:02}.pkl.gz')
    _tempfile.rename(f'{_datadir}/timebar/{_symbol}/{_interval}/{_symbol}-timebar-{_interval}sec-{_target_date.year:04}-{_target_date.month:02}-{_target_date.day:02}.pkl.gz')

    return idx

# 全コア数-2個のコアで並列処理を行い、価格ファイルを処理して約定プロファイルを作成する関数
def generate_timebar_files(datadir: str = None, symbol: str = None, interval: int = None):
    assert datadir is not None
    assert symbol is not None
    assert interval is not None

    _symbol = symbol.upper()

    # 処理開始前に全てのincompleteファイルを削除する
    _list_incomplete_files = identify_datafiles(datadir, 'timebar', _symbol, interval, incomplete = True)
    for _incomplete_file in _list_incomplete_files:
        _incomplete_file.unlink()
        
    # タイムバーを生成する (この時点ではまだ一日の始まりのタイムバーのOpenがNaNで、ファイル名先頭にincomplete-がついているものが存在する可能性がある)
    print(f'{symbol}の{interval}秒タイムバーファイルを約定履歴から生成します')
    _set_filenames = identify_available_trades_files(datadir, symbol, interval)
    _list_filenames = list(_set_filenames)
    _num_rows = len(_list_filenames)
    with tqdm_joblib(total = _num_rows):
        results = joblib.Parallel(n_jobs = -2, timeout = 60*60*24)([joblib.delayed(calc_timebar_from_trades)(_idx, _filename, interval) for _idx, _filename in enumerate(_list_filenames)])
    
    # Incompleteなファイルを完成させる
    print(f'未完成の{symbol}の{interval}秒タイムバーファイルがある場合は、それらを完成させます')
    _list_incomplete_files = identify_datafiles(datadir, 'timebar', _symbol, interval, incomplete = True)
    _list_filenames = sorted([str(_) for _ in _list_incomplete_files])
    _num_rows = len(_list_filenames)
    with tqdm_joblib(total = _num_rows):
        results = joblib.Parallel(n_jobs = -2, timeout = 60*60*24)([joblib.delayed(finish_incomplete_timebar_files)(_idx, _filename, interval) for _idx, _filename in enumerate(_list_filenames)])

    # 処理開始後に全てのincompleteファイルを削除する
    _list_incomplete_files = identify_datafiles(datadir, 'timebar', _symbol, interval, incomplete = True)
    for _incomplete_file in _list_incomplete_files:
        _incomplete_file.unlink()

# 引数処理とダウンロード関数の起動部分
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('symbol', help = 'ダウンロードする対象の銘柄 例:BTCUSDT')
    parser.add_argument('interval', type = int, help = '生成するタイムバーの時間間隔 [秒] 例:60')
    args = parser.parse_args()

    symbol = args.symbol
    interval = args.interval
    if len(symbol) > 0:
        generate_timebar_files(datadir, symbol, int(interval))