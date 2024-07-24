from numba.typed import Dict
import numpy as np
from hftbacktest import BUY, SELL, GTX
from numba import njit
from hftbacktest import HftBacktest, FeedLatency, Linear, SquareProbQueueModel, Stat
from calculators import (
    measure_trading_intensity,
    linear_regression,
    compute_coeff
)

NS_IN_ONE_SECOND = 1000_000
BUFFER_SIZE = 10_000_000
MAX_REFIT_TICK_CNT = 500


# Whole setting: 
elapse_in_ns = 100_000

# Related to Refit Calculation:
refit_duration_in_second = 5
window_size_in_second = 60
refit_tick_cnt = 70
assert refit_tick_cnt <= MAX_REFIT_TICK_CNT
gamma = 0.05
delta = 1
adj1 = 1
adj2 = 0.05

# Related to grid setting
order_qty = 1
max_position = 20
grid_num = 20

@njit
def calculate_nearest_bid_ask_price(hbt, mid_price_tick, half_spread, skew):
    bid_depth = half_spread + skew * hbt.position
    ask_depth = half_spread - skew * hbt.position

    bid_price = min(mid_price_tick - bid_depth, hbt.best_bid_tick) * hbt.tick_size
    ask_price = max(mid_price_tick + ask_depth, hbt.best_ask_tick) * hbt.tick_size

    grid_interval = max(np.round(half_spread) * hbt.tick_size, hbt.tick_size)
    bid_price = np.floor(bid_price / grid_interval) * grid_interval
    ask_price = np.ceil(ask_price / grid_interval) * grid_interval
    return bid_price, ask_price, grid_interval

@njit
def gridtrading_glft_mm(hbt, stat):
    arrival_depth = np.full(BUFFER_SIZE, np.nan, np.float64)
    mid_price_chg = np.full(BUFFER_SIZE, np.nan, np.float64)
    out = np.full((BUFFER_SIZE, 5), np.nan, np.float64)
    elapse_cnt_in_second = NS_IN_ONE_SECOND // elapse_in_ns
    update_duration = refit_duration_in_second * elapse_cnt_in_second
    win_size = window_size_in_second * elapse_cnt_in_second
    t = 0
    prev_mid_price_tick = np.nan
    mid_price_tick = np.nan
    half_spread = np.nan
    skew = np.nan
    tmp = np.zeros(MAX_REFIT_TICK_CNT, np.float64)
    ticks = np.arange(len(tmp)) + .5
    A = np.nan
    k = np.nan
    volatility = np.nan
    # Checks every 100 milliseconds.
    while hbt.elapse(elapse_in_ns):
        hbt.clear_last_trades()
        #--------------------------------------------------------
        # Records market order's arrival depth from the mid-price.
        if not np.isnan(mid_price_tick):
            depth = -np.inf
            for trade in hbt.last_trades:
                side = trade[3]
                trade_price_tick = trade[4] / hbt.tick_size

                if side == BUY:
                    depth = np.nanmax([trade_price_tick - mid_price_tick, depth])
                else:
                    depth = np.nanmax([mid_price_tick - trade_price_tick, depth])
            arrival_depth[t] = depth
        prev_mid_price_tick = mid_price_tick
        mid_price_tick = (hbt.best_bid_tick + hbt.best_ask_tick) / 2.0
        # Records the mid-price change for volatility calculation.
        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick
        #--------------------------------------------------------
        # Calibrates A, k and calculates the market volatility.
        if t % update_duration == 0:
            if t >= win_size - 1:
                # Calibrates A, k
                tmp[:] = 0
                lambda_ = measure_trading_intensity(
                    arrival_depth[t + 1 - win_size:t + 1], tmp)
                lambda_ = lambda_[:refit_tick_cnt] / window_size_in_second
                x = ticks[:len(lambda_)]
                y = np.log(lambda_)
                k_, logA = linear_regression(x, y)
                A = np.exp(logA)
                k = -k_
                # Updates the volatility.
                volatility = np.nanstd(
                    mid_price_chg[t + 1 - win_size:t + 1]
                ) * np.sqrt(elapse_cnt_in_second)
                # 1 seconds 
                #--------------------------------------------------------
                # Computes bid price and ask price.
                c1, c2 = compute_coeff(gamma, gamma, delta, A, k)
                half_spread = (c1 + delta / 2 * c2 * volatility) * adj1
                skew = c2 * volatility * adj2
        
        bid_price, ask_price, grid_interval = calculate_nearest_bid_ask_price(hbt, mid_price_tick, half_spread, skew)
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

        #--------------------------------------------------------
        # Records variables and stats for analysis.
        out[t, 0] = half_spread
        out[t, 1] = skew
        out[t, 2] = volatility
        out[t, 3] = A
        out[t, 4] = k

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
