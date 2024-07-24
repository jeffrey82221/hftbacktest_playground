import numpy as np
from numba import njit
from calculators import (
    measure_trading_intensity,
    linear_regression,
    compute_coeff
)
from variables import (
    MAX_REFIT_TICK_CNT, 
    NS_IN_ONE_SECOND,
    ELAPSE_IN_NS,
    FIT_WIN_SIZE
)

refit_tick_cnt = 70
assert refit_tick_cnt <= MAX_REFIT_TICK_CNT
gamma = 0.05
delta = 1
adj1 = 1
adj2 = 0.05

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