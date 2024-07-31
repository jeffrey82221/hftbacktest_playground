from numba.typed import Dict
import numpy as np
from numba import njit
from hftbacktest import BUY, SELL, GTX

from variables import (
    order_qty,
    max_position,
    grid_num
)

@njit
def set_grids(hbt, low, high, current_price, grid_num, order_qty):
    """
    TODO:
        - [ ] add stop loss and stop earning price
        - [ ] add trigger price
    """
    assert low < current_price
    assert current_price < high
    grid_interval = ((high - low) // grid_num)
    print('grid_interval:', grid_interval)
    unit_order_qty = order_qty // grid_num
    assert unit_order_qty >= 1
    unit_order_qty = round(unit_order_qty / hbt.lot_size) * hbt.lot_size
    price = low
    bid_prices = []
    ask_prices = []
    while price <= high:
        print('price:', price)
        if price < current_price:
            bid_price_tick = round(price / hbt.tick_size)
            bid_price = bid_price_tick * hbt.tick_size
            bid_prices.append(bid_price)
        else:
            ask_price_tick = round(price / hbt.tick_size)
            ask_price = ask_price_tick * hbt.tick_size
            ask_prices.append(ask_price)
        price += grid_interval
    for i in range(len(ask_prices)):
        hbt.submit_buy_order(
            i,
            current_price,
            unit_order_qty,
            GTX
        )
        print('submmit buy order', current_price, unit_order_qty)
    i += 1
    for ask_price in ask_prices:
        hbt.submit_sell_order(
                i,
                ask_price,
                unit_order_qty,
                GTX
            )
        i+=1
        print('submmit sell order', ask_price, unit_order_qty)
    for bid_price in bid_prices:
        hbt.submit_buy_order(
                i,
                bid_price,
                unit_order_qty,
                GTX
            )
        i+=1
        print('submmit buy order', bid_price, unit_order_qty)
@njit
def update_grids(hbt, grid_interval, bid_price, ask_price):
    hbt.clear_inactive_orders()
    # Creates a new grid for buy orders.
    new_bid_orders = Dict.empty(np.int64, np.float64)
    _grid_num = grid_num // 2
    if hbt.position < max_position and np.isfinite(bid_price):
        for i in range(_grid_num):
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
        for i in range(_grid_num):
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