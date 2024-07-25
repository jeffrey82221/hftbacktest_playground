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
from hftbacktest.data.utils import binancefutures
import gzip

for date_str in ['20240723', '20240724', '20240725']:
    with open(f'raw/nexousdt_{date_str}.dat', 'rb') as f:
        compressed_data = gzip.compress(f.read())
    with open(f'raw/nexousdt_{date_str}.dat.gz', 'wb') as file:
        file.write(compressed_data)
    binancefutures.convert(
        f'raw/nexousdt_{date_str}.dat.gz',
        output_filename=f'data/nexousdt_{date_str}.npz',
        opt='',
        compress=False
    )