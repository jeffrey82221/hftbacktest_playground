# Goal 

Successfully download and process ETH 
data and feed them to hftbacktest


# How?

1. Run `collect.sh` in collect-binancefutures continually to get streaming data.

2. Run `convert.sh` day by day to get `.npz` files.

3. Connect `.npz` to the package and show a visualization plot.

