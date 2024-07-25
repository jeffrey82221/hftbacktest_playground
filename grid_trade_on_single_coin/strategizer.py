import numpy as np
from numba import njit
from strategy_calculators import (
    measure_trading_intensity,
    linear_regression,
    compute_coeff
)
from variables import (
    NS_IN_ONE_SECOND,
    ELAPSE_IN_NS,
    FIT_WIN_SIZE
)
from variables import (
    refit_tick_cnt,
    gamma,
    delta,
    adj1,
    adj2
)


@njit
def fit_parameters(arrival_depth, mid_price_chg, ticks, tmp):
    tmp[:] = 0
    lambda_ = measure_trading_intensity(
        arrival_depth, tmp)
    lambda_ = lambda_[:refit_tick_cnt] / FIT_WIN_SIZE
    x = ticks[:len(lambda_)]
    y = np.log(lambda_)
    k_, logA = linear_regression(x, y)
    A = np.exp(logA)
    k = -k_
    # Updates the volatility.
    volatility = np.nanstd(
        mid_price_chg
    ) * np.sqrt(NS_IN_ONE_SECOND // ELAPSE_IN_NS)
    # 1 seconds 
    #--------------------------------------------------------
    # Computes bid price and ask price.
    c1, c2 = compute_coeff(gamma, gamma, delta, A, k)
    half_spread = (c1 + delta / 2 * c2 * volatility) * adj1
    skew = c2 * volatility * adj2
    return (
        half_spread,
        skew,
        volatility,
        A,
        k
    )

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