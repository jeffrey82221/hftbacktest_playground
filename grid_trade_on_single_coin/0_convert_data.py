"""
opt:
    m: Processes markPriceUpdate stream with the following custom event IDs.
        index: 100
        mark price: 101
        funding rate: 102
    t: Processes bookTicker stream with the following custom event IDs.
        best bid: 103
        best ask: 104
"""
import os
from hftbacktest.data.utils import binancefutures
from hftbacktest.data.utils.snapshot import create_last_snapshot

turn_on_stage = ['gzip', 'npz', 'eod'] 

src_folder = '../../collect-binancefutures/example_data/binancefutures'
for coin in ['btc', 'eth']:
    for date_str in ['20240726', '20240727', '20240728', '20240729']:
        original_path = f'{src_folder}/{coin}usdt_{date_str}.dat'
        compressed_path = f'{src_folder}/{coin}usdt_{date_str}.dat.gz'
        npz_path = f'{src_folder}/{coin}usdt_{date_str}.npz'
        if 'gzip' in turn_on_stage:
            compress_cmd = f'gzip -c {original_path} > {compressed_path}'
            os.system(compress_cmd)
            print(compress_cmd, 'done')
        if 'npz' in turn_on_stage:
            print(compressed_path, '-> npz start')
            binancefutures.convert(
                compressed_path,
                output_filename=npz_path,
                opt='',
                compress=True
            )
            print(compressed_path, '-> npz done')
        if 'eod' in turn_on_stage:
            _ = create_last_snapshot(
                [npz_path],
                tick_size=0.1,
                lot_size=0.001,
                output_snapshot_filename=f'{src_folder}/{coin}usdt_{date_str}_eod.npz'
            )
            print(npz_path, '-> eod file done')