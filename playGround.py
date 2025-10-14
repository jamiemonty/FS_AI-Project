import mplfinance as mpf
import pandas as pd
import numpy as np
import os

def play():
    theYear = 2024
    period = 20
    std_factor = 2
    max_hold_days = 40

    data_folder = "YahooStockData"
    output_folder = "plots"
    os.makedirs(output_folder, exist_ok=True)

    trades = []

    for file in os.listdir(data_folder):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        df['SMA'] = df['Adj Close'].rolling(window=period).mean()
        df['STD'] = df['Adj Close'].rolling(window=period).std()
        df['Upper Band'] = df['SMA'] + (std_factor * df['STD'])
        df['Lower Band'] = df['SMA'] - (std_factor * df['STD'])

        df = df[df['Date'].dt.year == theYear].reset_index(drop=True)
        if df.empty:
            continue

        in_trade = False
        buy_date = None
        buy_price = None
        buy_pct_below = None
        buy_dates = []
        sell_dates = []

        for i in range(period, len(df)):
            if not in_trade:
                if df.iloc[i]['Adj Close'] < df.iloc[i]['Lower Band']:
                    in_trade = True
                    buy_date = df.iloc[i]['Date']
                    buy_price = df.iloc[i]['Adj Close']
                    buy_pct_below = (df.iloc[i]['SMA'] - buy_price) / df.iloc[i]['SMA'] * 100
                    buy_dates.append(buy_date)
            else:
                days_held = (df.iloc[i]['Date'] - buy_date).days
                if df.iloc[i]['Adj Close'] > df.iloc[i]['Upper Band'] or days_held >= max_hold_days or i == len(df) - 1:
                    sell_date = df.iloc[i]['Date']
                    sell_price = df.iloc[i]['Adj Close']
                    pct_gain = (sell_price - buy_price) / buy_price * 100
                    trades.append([
                        file.replace(".csv", ""),
                        buy_date,
                        buy_price,
                        buy_pct_below,
                        sell_date,
                        sell_price,
                        pct_gain,
                        days_held
                    ])
                    sell_dates.append(sell_date)
                    in_trade = False

        # Candlestick plotting with Bollinger Bands
        df_plot = df.set_index('Date')
        addplots = [
            mpf.make_addplot(df_plot['SMA'], color='orange'),
            mpf.make_addplot(df_plot['Upper Band'], color='green', linestyle='--'),
            mpf.make_addplot(df_plot['Lower Band'], color='red', linestyle='--')
        ]

        # Buy/Sell vertical lines
        vlines = list(buy_dates) + list(sell_dates)
        vcolors = ['purple']*len(buy_dates) + ['black']*len(sell_dates)

        mpf.plot(
            df_plot,
            type='candle',
            style='charles',
            addplot=addplots,
            vlines=dict(vlines=vlines, colors=vcolors, linestyle='--', linewidths=1),
            title=f"{file.replace('.csv','')} Bollinger Bands {theYear}",
            ylabel='Price',
            savefig=os.path.join(output_folder, f"{file.replace('.csv','')}_{theYear}.png")
        )

    trades_df = pd.DataFrame(trades, columns=[
        "Ticker", "Buy Date", "Buy Price", "% Below Lower Band",
        "Sell Date", "Sell Price", "% Gain", "Days Held"
    ])
    trades_df.sort_values("Buy Date", inplace=True)
    trades_df.to_csv(f"{theYear}_perf.csv", index=False)

    print("Bollinger Bands candlestick analysis complete!")
    print(trades_df.head())
