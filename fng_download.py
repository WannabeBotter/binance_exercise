import requests
import re
import pandas as pd
from io import StringIO
from pathlib import Path

datadir = 'data/alternativeme'

def get_fear_index_csv(datadir:str = None):
    assert datadir is not None
    print('Crypto Fear and Greed Indexの過去データを全てダウンロードします')
    
    _url = 'https://api.alternative.me/fng/?limit=0&format=csv'
    
    _r = requests.get(_url)
    if _r.status_code != requests.codes.ok:
        print(f'response.get({_url})からHTTPステータスコード {_r.status_code} が返されました。')
        return
    
    _csvraw_re = re.compile('^.*\"data\": \[(.*?)\]', re.DOTALL)
    _m = _csvraw_re.match(_r.text)
    _csvraw = _m.group(1)
    
    _df = pd.read_csv(StringIO(_csvraw), names = ['date', 'fng_value', 'fng_classification'], dtype = {0: str, 1: int, 2: str}, header = 0)
    _df['date'] = pd.to_datetime(_df['date'], dayfirst = True)
    _df = _df.sort_values('date')
    _df = _df.set_index('date', drop = True)
    
    Path(f'{datadir}').mkdir(parents = True, exist_ok = True)
    _df.to_pickle(f'{datadir}/FNG-index-86400sec-0000-00-00.pkl.gz')

get_fear_index_csv(datadir)