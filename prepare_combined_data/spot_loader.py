import abc
import numpy as np
from typing import Iterator, Tuple
import json
import gzip

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
        return tmp[:i + 1, :]
    
class SpotEventLoader(EventLoader):
    """
    Load Spot Data of a day as a np.ndarray
    with format combinable with the npz feed data
    """
    def __init__(self, symbol: str, event_id: int, event_file_name: str, buffer_size: int = 100_1000):
        self._symbol = symbol
        super().__init__(
            event_id,
            event_file_name,
            buffer_size=buffer_size
        )

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
            print('start loading:', self._event_file_name)
            while True:
                line = f.readline()
                if line is None or line == b'':
                    break
                line = line.decode().strip()
                local_timestamp = int(line[:16])
                obj = json.loads(line[17:])
                if 'stream' in obj and obj['stream'] == f'{self._symbol}@bookTicker':
                    data = obj['data']
                    mid = (float(data['b']) + float(data['a'])) / 2.0
                    yield self._event_id, -1, local_timestamp, 0, mid, 0
