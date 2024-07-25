import numpy as np
from numba import njit
from hftbacktest import BUY, SELL

@njit
def record_arrival_depths(hbt, mid_price_tick):
    depth = -np.inf
    for trade in hbt.last_trades:
        side = trade[3]
        trade_price_tick = trade[4] / hbt.tick_size
        if side == BUY:
            depth = np.nanmax([trade_price_tick - mid_price_tick, depth])
        elif side == SELL:
            depth = np.nanmax([mid_price_tick - trade_price_tick, depth])
        else:
            raise Exception
    return depth