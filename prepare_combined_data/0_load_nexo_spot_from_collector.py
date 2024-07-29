import os
from spot_loader import SpotEventLoader

for date_str in ['20240723', '20240724', '20240725']:
    original_path = f'data/raw/spot/nexousdt_{date_str}.dat'
    compressed_path = f'data/raw/spot/nexousdt_{date_str}.dat.gz'
    os.system(f'gzip -c {original_path} > {compressed_path}')
    data = SpotEventLoader('nexousdt', 110, f'data/raw/spot/nexousdt_{date_str}.dat.gz').load()
    print(f'shape of nexo spot in {date_str} is: {data.shape}')