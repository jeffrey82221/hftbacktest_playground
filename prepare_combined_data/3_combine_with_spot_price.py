import gzip
import numpy as np
import json
from hftbacktest import merge_on_local_timestamp, validate_data


tmp = np.full((100_000, 6), np.nan, np.float64)
i = 0

with gzip.open('data/raw/spot/btcusdt_20230405.dat.gz', 'r') as f:
    while True:
        line = f.readline()
        if line is None or line == b'':
            break
        line = line.decode().strip()
        local_timestamp = int(line[:16])
        obj = json.loads(line[17:])
        if obj['stream'] == 'btcusdt@bookTicker':
            data = obj['data']
            mid = (float(data['b']) + float(data['a'])) / 2.0
            # Sets the event ID to 110 and assign an invalid exchange timestamp,
            # as it's not utilized in the exchange simulation.
            # And stores the mid-price in the price column.
            tmp[i] = [110, -1, local_timestamp, 0, mid, 0]
            i += 1
tmp = tmp[:i]

usdm_feed_data = np.load('data/npz/btcusdt_20230405_m.npz')['data']
merged = merge_on_local_timestamp(usdm_feed_data, tmp)
spot_data_size = len(tmp)
feed_data_size = len(usdm_feed_data)
merged_data_size = len(merged)
assert validate_data(merged) == 0
assert merged_data_size == spot_data_size + feed_data_size
np.savez('data/combined/btcusdt_20230405_m.npz', data=merged)
print('saved')