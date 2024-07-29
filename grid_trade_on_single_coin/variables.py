NS_IN_ONE_SECOND = 1000_000
BUFFER_SIZE = 10_000_000
MAX_REFIT_TICK_CNT = 500
ELAPSE_IN_NS = 1000_000

# STRATEGY RUN SETTINGS
REFIT_DURATION = 5 # seconds
FIT_WIN_SIZE = 60 # seconds

# STRATEGY PARAMETERS
refit_tick_cnt = 100
assert refit_tick_cnt <= MAX_REFIT_TICK_CNT
gamma = 0.05
delta = 1
adj1 = 1
adj2 = 1

# GRID SETTINGS
order_qty = 10
max_position = 200
grid_num = 20