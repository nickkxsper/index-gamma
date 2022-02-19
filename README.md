# index-gamma

Studying the relationship between gamma exposure, liquidity, and price action in indices and underlying stocks
Original Paper: https://squeezemetrics.com/download/The_Implied_Order_Book.pdf

TODO:

Finish gathering minutely and daily data for SPX first
Gather historical SPX gamma data
Explore paid API (me and michael got it) for other tickers
Grab historical price data for those tickers ^
Make features for liquidity
Liquidity is pretty much how many units can be traded per movement in price
 Liquidity = abs(Open-close)/Volume Traded
Study/test relationships between gamma/liquidity/and price movement
Make intra day, daily, and weekly strategies using some insights we gathered in 6
Backtest and compare to benchmarks
