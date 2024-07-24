from hftbacktest import COL_PRICE
from numba import njit
from hftbacktest import HftBacktest, FeedLatency, Linear

@njit
def print_funding_rate(hbt):
    # Checks every 60-sec (in microseconds)
    while hbt.elapse(60_000_000):
        best_bid_103 = hbt.get_user_data(103)[COL_PRICE]
        best_ask_103 = hbt.get_user_data(104)[COL_PRICE]
        mark_price = hbt.get_user_data(101)[COL_PRICE]
        funding_rate = hbt.get_user_data(102)[COL_PRICE]
        spot_mid_price = hbt.get_user_data(110)[COL_PRICE]
        mid_price = (hbt.best_bid + hbt.best_ask) / 2.0
        assert best_ask_103 - hbt.best_ask < 0.0001
        assert best_bid_103 - hbt.best_bid < 0.0001
        basis = mid_price - spot_mid_price
        print('current_timestamp:',
            hbt.current_timestamp,
            'mark_price:', 
            round(mark_price, 2), 
            'funding_rate:', 
            funding_rate, 
            'spot_mid_price:', spot_mid_price,
            'futures_mid:',
            round(mid_price),
            'basis:', round(basis, 2)
        )

hbt = HftBacktest(
    [
        'data/combined/btcusdt_20230405_mt.npz'
    ],
    tick_size=0.1,
    lot_size=0.001,
    maker_fee=0.0002,
    taker_fee=0.0007,
    order_latency=FeedLatency(),
    asset_type=Linear,
    snapshot='data/npz/btcusdt_20230404_eod.npz'
)

print_funding_rate(hbt)