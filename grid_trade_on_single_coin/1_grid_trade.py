import numpy as np
from numba import njit
from hftbacktest import HftBacktest, FeedLatency, Linear, SquareProbQueueModel, Stat
from index_recorder import (
    record_arrival_depths
)
from strategizer import (
    fit_parameters,
    calculate_nearest_bid_ask_price
)
from strategy_calculators import compute_coeff
from grid_manager import update_grids
from variables import (
    MAX_REFIT_TICK_CNT,
    NS_IN_ONE_SECOND,
    BUFFER_SIZE,
    ELAPSE_IN_NS,
    REFIT_DURATION,
    FIT_WIN_SIZE,
    gamma,
    delta,
    adj1,
    adj2
)

VERBOSE = True
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
    c1 = np.nan
    c2 = np.nan
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
                try:
                    volatility, A, k = fit_parameters(
                        arrival_depth[t + 1 - win_size:t + 1],
                        mid_price_chg[t + 1 - win_size:t + 1],
                        ticks,
                        tmp
                    )
                    k = round(k, 8)
                    if k > 0.0:
                        c1, c2 = compute_coeff(gamma, gamma, delta, A, k)
                        half_spread = (c1 + delta / 2 * c2 * volatility) * adj1
                        skew = c2 * volatility * adj2
                except Exception:
                    pass
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
        update_grids(hbt, grid_interval, bid_price, ask_price)
        mid_price = mid_price_ticks[t] * hbt.tick_size
        bid_percentage = round((bid_price - mid_price) / mid_price * 100, 5)
        ask_percentage = round((ask_price - mid_price) / mid_price * 100, 5)
        if bid_percentage < -100 or (VERBOSE and t % 1000 == 0):
            print(
                't', t,
                '\tA:', A,
                '\tK:', k,
                '\tc1:', c1,
                '\tc2:', c2,
                '\tvolatility:', volatility,
                '\thalf_spread:', half_spread,
                '\tskew:', skew,
                '\tmid_price:', mid_price,
                '\tmid_price_change:', mid_price_chg[t] * hbt.tick_size,
                '\tgrid_interval:', grid_interval,
                '\tbid_price %:', bid_percentage, '%'
                '\task_price %:', ask_percentage, '%'
            )
            """
            for order in hbt.orders.values():
                print('order_id', order.order_id, 
                    'side', order.side,
                    'price %:', round((order.price - mid_price) / mid_price * 100, 5), '%',
                    'qty', order.qty,
                    'cancellable', order.cancellable
                    )
            """
            if bid_percentage < -100:
                raise Exception
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
