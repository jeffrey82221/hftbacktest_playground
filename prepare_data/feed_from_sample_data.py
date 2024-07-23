"""
Question:

1. What is the usage of eod snapshot?
2. How to 
"""
from hftbacktest.data.utils import binancefutures
from hftbacktest import HftBacktest, FeedLatency, SquareProbQueueModel, Linear, Stat
from numba import njit
import numpy as np
from numba.typed import Dict
from hftbacktest import GTX, BUY, SELL

@njit
def gridtrading(hbt, stat):
    max_position = 5
    grid_interval = hbt.tick_size * 10
    grid_num = 20
    half_spread = hbt.tick_size * 20
        
    # Running interval in microseconds
    while hbt.elapse(100_000):
        # Clears cancelled, filled or expired orders.
        hbt.clear_inactive_orders()
        
        mid_price = (hbt.best_bid + hbt.best_ask) / 2.0
        bid_order_begin = np.floor((mid_price - half_spread) / grid_interval) * grid_interval
        ask_order_begin = np.ceil((mid_price + half_spread) / grid_interval) * grid_interval
        
        order_qty = 0.1
        last_order_id = -1
        
        # Creates a new grid for buy orders.
        new_bid_orders = Dict.empty(np.int64, np.float64)
        if hbt.position < max_position:
            for i in range(grid_num):
                bid_order_begin -= i * grid_interval
                bid_order_tick = round(bid_order_begin / hbt.tick_size)
                # Do not post buy orders above the best bid.
                if bid_order_tick > hbt.best_bid_tick:
                    continue
                    
                # order price in tick is used as order id.
                new_bid_orders[bid_order_tick] = bid_order_begin
        for order in hbt.orders.values():
            # Cancels if an order is not in the new grid.
            if order.side == BUY and order.cancellable and order.order_id not in new_bid_orders:
                hbt.cancel(order.order_id)
                last_order_id = order.order_id
        for order_id, order_price in new_bid_orders.items():
            # Posts an order if it doesn't exist.
            if order_id not in hbt.orders:
                hbt.submit_buy_order(order_id, order_price, order_qty, GTX)
                last_order_id = order_id
        
        # Creates a new grid for sell orders.
        new_ask_orders = Dict.empty(np.int64, np.float64)
        if hbt.position > -max_position:
            for i in range(grid_num):
                ask_order_begin += i * grid_interval
                ask_order_tick = round(ask_order_begin / hbt.tick_size)
                # Do not post sell orders below the best ask.
                if ask_order_tick < hbt.best_ask_tick:
                    continue

                # order price in tick is used as order id.
                new_ask_orders[ask_order_tick] = ask_order_begin
        for order in hbt.orders.values():
            # Cancels if an order is not in the new grid.
            if order.side == SELL and order.cancellable and order.order_id not in new_ask_orders:
                hbt.cancel(order.order_id)
                last_order_id = order.order_id
        for order_id, order_price in new_ask_orders.items():
            # Posts an order if it doesn't exist.
            if order_id not in hbt.orders:
                hbt.submit_sell_order(order_id, order_price, order_qty, GTX)
                last_order_id = order_id
        
        # All order requests are considered to be requested at the same time.
        # Waits until one of the order responses is received.
        if last_order_id >= 0:
            if not hbt.wait_order_response(last_order_id):
                return False
        
        # Records the current state for stat calculation.
        stat.record(hbt)
    return True

# Convert Data to npz
_ = binancefutures.convert(
    'usdm/btcusdt_20230404.dat.gz',
    output_filename='usdm/btcusdt_20230404.npz',
    combined_stream=True
)

_ = binancefutures.convert(
    'usdm/btcusdt_20230404.dat.gz',
    output_filename='usdm/btcusdt_20230405.npz',
    combined_stream=True
)

hbt = HftBacktest(
    [
        'usdm/btcusdt_20230404.npz',
        'usdm/btcusdt_20230405.npz'
    ],
    tick_size=0.01,
    lot_size=0.001,
    maker_fee=-0.00005,
    taker_fee=0.0007,
    order_latency=FeedLatency(),
    queue_model=SquareProbQueueModel(),
    asset_type=Linear
    # snapshot='data/ethusdt_20221002_eod.npz'
)

stat = Stat(hbt)
gridtrading(hbt, stat.recorder)
stat.summary(capital=15_000)