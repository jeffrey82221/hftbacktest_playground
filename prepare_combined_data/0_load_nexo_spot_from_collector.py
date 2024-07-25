import gzip
from spot_loader import SpotEventLoader, EventLoader

for date_str in ['20240723', '20240724', '20240725']:
    with open(f'data/raw/spot/nexousdt_{date_str}.dat', 'rb') as f:
        compressed_data = gzip.compress(f.read())
    with open(f'data/raw/spot/nexousdt_{date_str}.dat.gz', 'wb') as file:
        file.write(compressed_data)

    data = SpotEventLoader('nexousdt', 110, f'data/raw/spot/nexousdt_{date_str}.dat.gz').load()
    print(f'shape of nexo spot in {date_str} is: {data.shape}')