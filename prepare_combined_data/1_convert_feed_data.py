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

for date_str in ['20230404', '20230405']:
    binancefutures.convert(
        f'data/raw/usdm/btcusdt_{date_str}.dat.gz',
        output_filename=f'data/npz/btcusdt_{date_str}.npz',
        opt=''
    )
    for opt in ['m', 't', 'mt']:
        binancefutures.convert(
            f'data/raw/usdm/btcusdt_{date_str}.dat.gz',
            output_filename=f'data/npz/btcusdt_{date_str}_{opt}.npz',
            opt=opt
        )