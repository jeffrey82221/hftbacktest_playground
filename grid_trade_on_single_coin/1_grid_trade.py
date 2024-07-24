from numba.typed import Dict
import numpy as np
from hftbacktest import BUY, SELL, GTX
from numba import njit
from hftbacktest import HftBacktest, FeedLatency, Linear, SquareProbQueueModel, Stat
from calculators import (
    record_arrival_depths,
    calculate_nearest_bid_ask_price
)
from fit_calculator import fit_parameters
from variables import (
    MAX_REFIT_TICK_CNT,
    NS_IN_ONE_SECOND,
    BUFFER_SIZE,
    ELAPSE_IN_NS,
    REFIT_DURATION,
    FIT_WIN_SIZE
)


# Related to grid setting
order_qty = 1
max_position = 20
grid_num = 20


@njit
def gridtrading_glft_mm(hbt, stat):
    arrival_depth = np.full(BUFFER_SIZE, np.nan, np.float64)
    mid_price_ticks = np.full(BUFFER_SIZE, np.nan, np.float64)
    mid_price_chg = np.full(BUFFER_SIZE, np.nan, np.float64)
    out = np.full((BUFFER_SIZE, 5), np.nan, np.float64)
    t = 0
    tmp = np.zeros(MAX_REFIT_TICK_CNT, np.float64)
    ticks = np.arange(len(tmp)) + .5
    half_spread = np.nan
    skew = np.nan
    A = np.nan
    k = np.nan
    volatility = np.nan
    # Checks every 100 milliseconds.
    while hbt.elapse(ELAPSE_IN_NS):
        #--------------------------------------------------------
        mid_price_ticks[t] = (hbt.best_bid_tick + hbt.best_ask_tick) / 2.0
        mid_price_chg[t] = mid_price_ticks[t] - mid_price_ticks[t - 1]
        # Records market order's arrival depth from the mid-price.
        if t >= 1:
            arrival_depth[t] = record_arrival_depths(hbt, mid_price_ticks[t - 1])
        hbt.clear_last_trades()
        #--------------------------------------------------------
        # Calibrates A, k and calculates the market volatility.
        update_duration = REFIT_DURATION * NS_IN_ONE_SECOND // ELAPSE_IN_NS
        win_size = FIT_WIN_SIZE * NS_IN_ONE_SECOND // ELAPSE_IN_NS
        if t % update_duration == 0:
            if t >= win_size - 1:
                half_spread, skew, volatility, A, k = fit_parameters(
                    arrival_depth[t + 1 - win_size:t + 1],
                    mid_price_chg[t + 1 - win_size:t + 1],
                    ticks,
                    tmp
                )
        #--------------------------------------------------------
        # Records variables and stats for analysis.
        out[t, 0] = half_spread
        out[t, 1] = skew
        out[t, 2] = volatility
        out[t, 3] = A
        out[t, 4] = k
        
        bid_price, ask_price, grid_interval = calculate_nearest_bid_ask_price(
            hbt, mid_price_ticks[t], half_spread, skew)

        #--------------------------------------------------------
        # Updates quotes.

        hbt.clear_inactive_orders()

        # Creates a new grid for buy orders.
        new_bid_orders = Dict.empty(np.int64, np.float64)
        if hbt.position < max_position and np.isfinite(bid_price):
            for i in range(grid_num):
                bid_price -= i * grid_interval
                bid_price_tick = round(bid_price / hbt.tick_size)
                # order price in tick is used as order id.
                new_bid_orders[bid_price_tick] = bid_price
        for order in hbt.orders.values():
            # Cancels if an order is not in the new grid.
            if order.side == BUY and order.cancellable and order.order_id not in new_bid_orders:
                hbt.cancel(order.order_id)
        for order_id, order_price in new_bid_orders.items():
            # Posts an order if it doesn't exist.
            if order_id not in hbt.orders:
                hbt.submit_buy_order(order_id, order_price, order_qty, GTX)

        # Creates a new grid for sell orders.
        new_ask_orders = Dict.empty(np.int64, np.float64)
        if hbt.position > -max_position and np.isfinite(ask_price):
            for i in range(grid_num):
                ask_price += i * grid_interval
                ask_price_tick = round(ask_price / hbt.tick_size)
                # order price in tick is used as order id.
                new_ask_orders[ask_price_tick] = ask_price
        for order in hbt.orders.values():
            # Cancels if an order is not in the new grid.
            if order.side == SELL and order.cancellable and order.order_id not in new_ask_orders:
                hbt.cancel(order.order_id)
        for order_id, order_price in new_ask_orders.items():
            # Posts an order if it doesn't exist.
            if order_id not in hbt.orders:
                hbt.submit_sell_order(order_id, order_price, order_qty, GTX)
        t += 1
        if t >= BUFFER_SIZE:
            raise Exception
        # Records the current state for stat calculation.
        stat.record(hbt)
    return out[:t]

hbt = HftBacktest(
    [
        'data/btcusdt_20230405.npz'
    ],
    snapshot='data/btcusdt_20230404_eod.npz',
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
