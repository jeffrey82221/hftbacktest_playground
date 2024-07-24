from hftbacktest.data.utils.snapshot import create_last_snapshot

_ = create_last_snapshot(
    ['data/npz/btcusdt_20230404.npz'],
    tick_size=0.1,
    lot_size=0.001,
    output_snapshot_filename='data/npz/btcusdt_20230404_eod.npz'
)