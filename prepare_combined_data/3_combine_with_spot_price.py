import abc
import gzip
import numpy as np
import json
from typing import Iterator, Tuple
from hftbacktest import merge_on_local_timestamp, validate_data

class EventLoader:
    def __init__(self, event_id: int, event_file_name: str, buffer_size: int = 100_1000):
        self._event_id = event_id
        self._event_file_name = event_file_name
        self._buffer_size = buffer_size

    @property
    def _empty_buffer(self):
        return np.full((self._buffer_size, 6), np.nan, np.float64)

    @abc.abstractmethod
    def generator(self) -> Iterator[Tuple[int, int, int, int, float, int]]:
        """
        Generate data to append to buffer

        yields:
            item[0]: event_id
            item[1]: ?
            item[2]: local_timestamp
            item[3]: ?
            item[4]: value
            item[5]: ?
        """
        raise NotImplementedError
    
    def load(self) -> np.ndarray:
        tmp = self._empty_buffer
        for i, instance in enumerate(self.generator()):
            tmp[i] = list(instance)    
        print(self._event_file_name, 'loaded and parsed')
        return tmp
    
class SpotEventLoader(EventLoader):
    """
    Load Spot Data of a day as a np.ndarray
    with format combinable with the npz feed data
    """
    def generator(self) -> Iterator[Tuple[int, int, int, int, float, int]]:
        """
        Generate data to append to buffer

        yields:
            item[0]: event_id
            item[1]: ?
            item[2]: local_timestamp
            item[3]: ?
            item[4]: value
            item[5]: ?
        """
        with gzip.open(self._event_file_name, 'r') as f:
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
                    yield self._event_id, -1, local_timestamp, 0, mid, 0

    
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
        SpotEventLoader(110, 'data/raw/spot/btcusdt_20230405.dat.gz'),
        source_feed_file = 'data/npz/btcusdt_20230405_mt.npz',
        target_file = 'data/combined/btcusdt_20230405_mt.npz',
    )