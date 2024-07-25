import numpy as np
from hftbacktest import merge_on_local_timestamp, validate_data
from spot_loader import SpotEventLoader, EventLoader


    
def merge_custom_data_to_feed_data(
    event_loader: EventLoader, 
    source_feed_file: str,
    target_file: str
):
    additional_data_array = event_loader.load()
    usdm_feed_data = np.load(source_feed_file)['data']
    print(source_feed_file, 'loaded')
    merged = merge_on_local_timestamp(usdm_feed_data, additional_data_array)
    spot_data_size = len(additional_data_array)
    feed_data_size = len(usdm_feed_data)
    merged_data_size = len(merged)
    assert validate_data(merged) == 0
    assert merged_data_size == spot_data_size + feed_data_size
    np.savez(target_file, data=merged)
    print(target_file, 'saved')


if __name__ == '__main__':
    merge_custom_data_to_feed_data(
        SpotEventLoader('btcusdt', 110, 'data/raw/spot/btcusdt_20230405.dat.gz'),
        source_feed_file = 'data/npz/btcusdt_20230405_mt.npz',
        target_file = 'data/combined/btcusdt_20230405_mt.npz',
    )