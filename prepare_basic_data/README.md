# Goal 

Successfully download and process ETH 
data and feed them to hftbacktest


# How?

1. Run `collect.sh` in collect-binancefutures continually to get streaming data.

2. Run `convert.sh` day by day to get `.npz` files.

3. Connect `.npz` to the package and show a visualization plot.


# Questions:

1. What is the usage of eod snapshot?

As Binance Futures exchange runs 24/7, you need the initial snapshot to get the complete(almost) market depth.

2. How to determine tick size and lot size for extracting the snapshot?


2. What is the difference between spot, futures, perpetual futures data?


