import numpy as np
from numba import njit
from hftbacktest import HftBacktest, FeedLatency, Linear, SquareProbQueueModel, Stat
from grid_manager import set_grids
import matplotlib.pylab as plt
from variables import (
    BUFFER_SIZE,
    ELAPSE_IN_NS
)

VERBOSE = True

@njit
def gridtrading_glft_mm(hbt, stat):
    out = np.full((BUFFER_SIZE, 3), np.nan, np.float64)
    t = 0
    # Checks every 100 milliseconds.
    while hbt.elapse(ELAPSE_IN_NS):
        #--------------------------------------------------------
        mid_price_tick = (hbt.best_bid_tick + hbt.best_ask_tick) / 2.0
        # Records market order's arrival depth from the mid-price.
        hbt.clear_last_trades()
        #--------------------------------------------------------
        # Calibrates A, k and calculates the market volatility.
        #--------------------------------------------------------
        # Updates quotes.
        # update_grids(hbt, grid_interval, bid_price, ask_price)
        mid_price = mid_price_tick * hbt.tick_size
        low = 67400
        high = 69000
        if t == 0:
            set_grids(hbt, low, high, mid_price, 5, 100_000)
        out[t, 0] = low
        out[t, 1] = high
        out[t, 2] = mid_price
        if VERBOSE and t % 10000 == 0:
            print(
                't', t,
                '\tmid_price:', mid_price
            )
        t += 1
        if t >= BUFFER_SIZE:
            raise Exception
        # Records the current state for stat calculation.
        stat.record(hbt)
    return out[:t]

src_folder = '../../collect-binancefutures/example_data/binancefutures'
hbt = HftBacktest(
    [
        f'{src_folder}/btcusdt_20240727.npz'
    ],
    snapshot=f'{src_folder}/btcusdt_20240726_eod.npz',
    tick_size=0.1,
    lot_size=0.001,
    maker_fee=0.0002,
    taker_fee=0.0007,
    order_latency=FeedLatency(),
    asset_type=Linear,
    queue_model=SquareProbQueueModel(),
    trade_list_size=10_000,
)

stat = Stat(hbt)

out = gridtrading_glft_mm(hbt, stat.recorder)

stat.summary(capital=100_000)
plt.show()
plt.plot(out[:, 0])
plt.plot(out[:, 1])
plt.plot(out[:, 2])
plt.show()