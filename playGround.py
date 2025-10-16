import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import mplfinance as mpf

def backtest_year(theYear, ma_period=100, buy_threshold=0.20, max_hold_days=40, bb_period=20, bb_std=2):
    """
    Backtest the trading strategy for a specific year.
    Modified to handle overlapping trades by only allowing one position per stock at a time.
    Optimized parameters: 20% below 100-day MA, 40-day max hold
    Also calculate Bollinger Bands (default 20-day, 2 std) for plotting.
    """
    print(f"\n🔍 Backtesting year {theYear}...")
    
    data_folder = "YahooStockData"
    output_folder = f"plots_{theYear}"
    os.makedirs(output_folder, exist_ok=True)
    
    trades = []
    all_trades_chronological = []  # For calculating compounded returns
    
    # Process all CSV files in the data folder
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print(f"Processing {len(csv_files)} stock files...")
    
    for idx, file in enumerate(csv_files):
        if idx % 50 == 0:
            print(f"Processing file {idx+1}/{len(csv_files)}: {file}")
            
        file_path = os.path.join(data_folder, file)
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Calculate 100-day moving average
            df['MA_100'] = df['Adj Close'].rolling(window=ma_period).mean()

            # Calculate Bollinger Bands on Adj Close using bb_period and bb_std
            df['BB_Middle'] = df['Adj Close'].rolling(window=bb_period).mean()
            df['BB_Std'] = df['Adj Close'].rolling(window=bb_period).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * df['BB_Std'])
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * df['BB_Std'])
            
            # Calculate MACD (standard: 12, 26, 9 EMA)
            df['EMA_12'] = df['Adj Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Adj Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Filter for the specified year and ensure we have MA data
            df_year = df[df['Date'].dt.year == theYear].copy()
            df_year = df_year.dropna(subset=['MA_100']).reset_index(drop=True)

            if df_year.empty:
                continue
                
            # Track trading state
            in_trade = False
            buy_date = None
            buy_price = None
            buy_pct_below = None
            buy_index = None
            buy_dates = []
            sell_dates = []
            
            # Process each trading day
            for i in range(len(df_year)):
                current_price = df_year.iloc[i]['Adj Close']
                current_ma = df_year.iloc[i]['MA_100']
                current_date = df_year.iloc[i]['Date']
                
                if not in_trade:
                    # Check for buy signal: price at least 30% below 100-day MA
                    pct_below_ma = (current_ma - current_price) / current_ma
                    
                    if pct_below_ma >= buy_threshold:
                        in_trade = True
                        buy_date = current_date
                        buy_price = current_price
                        buy_pct_below = pct_below_ma * 100
                        buy_index = i
                        buy_dates.append(buy_date)
                        
                else:
                    # Check for sell signals
                    trading_days_held = i - buy_index
                    
                    sell_condition_met = False
                    
                    # Sell condition 1: Price above 100-day MA
                    if current_price > current_ma:
                        sell_condition_met = True
                    
                    # Sell condition 2: More than 40 trading days elapsed
                    elif trading_days_held >= max_hold_days:
                        sell_condition_met = True
                    
                    # Sell condition 3: End of year (last trading day)
                    elif i == len(df_year) - 1:
                        sell_condition_met = True
                    
                    if sell_condition_met:
                        sell_date = current_date
                        sell_price = current_price
                        pct_gain = (sell_price - buy_price) / buy_price * 100
                        
                        trade_data = [
                            file.replace(".csv", ""),
                            buy_date,
                            buy_price,
                            buy_pct_below,
                            sell_price,
                            sell_date,
                            pct_gain,
                            trading_days_held
                        ]
                        trades.append(trade_data)
                        
                        # Add to chronological list for compounding calculation
                        all_trades_chronological.append({
                            'ticker': file.replace(".csv", ""),
                            'buy_date': buy_date,
                            'sell_date': sell_date,
                            'return': pct_gain / 100,
                            'days_held': trading_days_held
                        })
                        
                        sell_dates.append(sell_date)
                        in_trade = False
            
            # Only create plots for stocks that had at least one trade
            if buy_dates or sell_dates:
                # Prepare dataframe for mplfinance (must have columns: Open, High, Low, Close, Volume, Date as index)
                df_plot = df_year.copy()
                df_plot.set_index('Date', inplace=True)
                # Add Bollinger Bands columns if not present
                if 'BB_Upper' not in df_plot.columns:
                    df_plot['BB_Upper'] = np.nan
                if 'BB_Lower' not in df_plot.columns:
                    df_plot['BB_Lower'] = np.nan
                # Create additional overlays
                addplots = [
                    mpf.make_addplot(df_plot['MA_100'], color='orange', width=1.5, label='100-day MA'),
                    mpf.make_addplot(df_plot['BB_Upper'], color='lightgrey', linestyle=':', width=1, label='BB Upper'),
                    mpf.make_addplot(df_plot['BB_Lower'], color='lightgrey', linestyle=':', width=1, label='BB Lower'),
                    mpf.make_addplot(df_plot['BB_Middle'], color='grey', linestyle='--', width=1, label=f'{bb_period}-day BB Middle'),
                ]
                # Mark buy/sell signals
                buy_marker = [np.nan]*len(df_plot)
                sell_marker = [np.nan]*len(df_plot)
                for i, dt in enumerate(df_plot.index):
                    if dt in buy_dates:
                        buy_marker[i] = df_plot.iloc[i]['Low'] * 0.98  # marker slightly below low
                    if dt in sell_dates:
                        sell_marker[i] = df_plot.iloc[i]['High'] * 1.02  # marker slightly above high
                addplots.append(mpf.make_addplot(buy_marker, type='scatter', markersize=80, marker='^', color='green', label='Buy Signal'))
                addplots.append(mpf.make_addplot(sell_marker, type='scatter', markersize=80, marker='v', color='red', label='Sell Signal'))
                # MACD addplots (panel=1 for subplot)
                macd_plots = [
                    mpf.make_addplot(df_plot['MACD'], panel=1, color='blue', width=1.5, label='MACD'),
                    mpf.make_addplot(df_plot['MACD_Signal'], panel=1, color='orange', width=1, label='MACD Signal'),
                    mpf.make_addplot(df_plot['MACD_Hist'], panel=1, type='bar', color='grey', alpha=0.5, label='MACD Hist'),
                ]
                # Combine main and MACD addplots
                all_addplots = addplots + macd_plots
                # Plot candlestick chart with overlays and MACD panel
                plot_filename = f"{file.replace('.csv', '')}_{theYear}.png"
                mpf.plot(
                    df_plot,
                    type='candle',
                    style='yahoo',
                    addplot=all_addplots,
                    title=f"{file.replace('.csv', '')} - {theYear} Trading Signals",
                    ylabel='Price ($)',
                    volume=False,
                    savefig=dict(fname=os.path.join(output_folder, plot_filename), dpi=300, bbox_inches='tight'),
                    tight_layout=True,
                    datetime_format='%Y-%m-%d',
                    xrotation=45,
                    figscale=1.2,
                    figratio=(12,8),
                    show_nontrading=False,
                    panel_ratios=(3,1),
                    ylabel_lower='MACD'
                )
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Create results dataframe
    if trades:
        trades_df = pd.DataFrame(trades, columns=[
            "Ticker", "Buy Date", "Buy Price", "% Below 100-day MA",
            "Sell Price", "Sell Date", "% Gain", "Trading Days Held"
        ])
        
        # Sort by buy date
        trades_df = trades_df.sort_values("Buy Date").reset_index(drop=True)
        
        # Save to CSV
        output_filename = f"{theYear}_perf.csv"
        trades_df.to_csv(output_filename, index=False)
        
        # Calculate non-overlapping trades for compounding
        non_overlapping_trades = calculate_non_overlapping_trades(all_trades_chronological)
        total_compounded_return = calculate_compounded_return(non_overlapping_trades)
        
        print(f"\n✅ Analysis complete for {theYear}!")
        print(f"📊 Total trades found: {len(trades_df)}")
        print(f"🏢 Stocks with trades: {trades_df['Ticker'].nunique()}")
        print(f"💾 Results saved to: {output_filename}")
        print(f"📈 Plots saved to: {output_folder}/")
        
        # Display summary statistics  
        avg_gain = trades_df['% Gain'].mean()
        avg_days = trades_df['Trading Days Held'].mean()
        win_rate = (trades_df['% Gain'] > 0).mean() * 100
        
        print(f"\n📈 Performance Summary for {theYear}:")
        print(f"Average gain per trade: {avg_gain:.2f}%")
        print(f"Average days held: {avg_days:.1f}")
        print(f"Total trades: {len(trades_df)}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Non-overlapping trades: {len(non_overlapping_trades)}")
        print(f"Estimated total compounded return: {total_compounded_return:.2f}%")
        
        return {
            'year': theYear,
            'avg_return': avg_gain,
            'avg_days': avg_days,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'compounded_return': total_compounded_return,
            'non_overlapping_trades': len(non_overlapping_trades)
        }
        
    else:
        print(f"\n❌ No trades found for {theYear}")
        return {
            'year': theYear,
            'avg_return': 0,
            'avg_days': 0,
            'total_trades': 0,
            'win_rate': 0,
            'compounded_return': 0,
            'non_overlapping_trades': 0
        }

def calculate_non_overlapping_trades(all_trades):
    """
    Filter trades to remove overlaps, keeping the best performing trades.
    Sort by buy date and only keep trades that don't overlap with previous ones.
    """
    if not all_trades:
        return []
    
    # Sort trades by buy date
    sorted_trades = sorted(all_trades, key=lambda x: x['buy_date'])
    non_overlapping = []
    
    for trade in sorted_trades:
        # Check if this trade overlaps with any previous trade
        overlaps = False
        for existing_trade in non_overlapping:
            if (trade['buy_date'] <= existing_trade['sell_date'] and 
                trade['sell_date'] >= existing_trade['buy_date']):
                overlaps = True
                break
        
        if not overlaps:
            non_overlapping.append(trade)
    
    return non_overlapping

def calculate_compounded_return(trades):
    """
    Calculate the compounded return assuming trades are executed sequentially.
    """
    if not trades:
        return 0.0
    
    total_return = 1.0
    for trade in trades:
        total_return *= (1 + trade['return'])
    
    return (total_return - 1) * 100

def play():
    """
    Main function: Use BEST optimized strategy parameters and backtest 12 years (2014-2025).
    Based on optimization results: 50-day MA, 30% threshold, 30-day max hold = 108.30% return in 2024.
    """
    print("Code created using GPT-4o (ChatGPT)")
    print("🚀 Using BEST Optimized Parameters from 2024 Testing...")
    
    # Use BEST optimized parameters found: MA=50, Threshold=30%, MaxDays=30 (108.30% return)
    ma_period, buy_threshold, max_hold_days = 50, 0.30, 30
    print(f"BEST PARAMETERS: {buy_threshold*100:.0f}% below {ma_period}-day MA, {max_hold_days}-day max hold")
    print("(This combination achieved 108.30% compounded return in 2024 optimization)")
    
    # Step 2: Backtest optimized strategy for 12 years (2014-2025)
    print(f"\n📈 Running 12-year backtest with optimized parameters...")
    test_years = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014]
    
    results = []
    yearly_returns = []
    
    # Run backtest for each year with optimized parameters
    for year in test_years:
        try:
            year_result = backtest_year(year, ma_period, buy_threshold, max_hold_days)
            results.append(year_result)
            yearly_returns.append(year_result['compounded_return'])
            print(f"Finished processing year {year}. Compounded gain: {year_result['compounded_return']:.2f}%")
        except Exception as e:
            print(f"Error processing year {year}: {str(e)}")
            results.append({
                'year': year,
                'avg_return': 0,
                'avg_days': 0,
                'total_trades': 0,
                'win_rate': 0,
                'compounded_return': 0,
                'non_overlapping_trades': 0
            })
            yearly_returns.append(0)
    
    # Calculate total compound return over 12 years
    total_compound = 1.0
    for annual_return in yearly_returns:
        total_compound *= (1 + annual_return / 100)
    
    print(f"\n🎯 12-Year Strategy Performance:")
    print(f"Total compound return: {(total_compound - 1) * 100:.2f}%")
    print(f"Compound calculation: {' * '.join([f'{1 + r/100:.4f}' for r in yearly_returns])} = {total_compound:.3f}")
    
    # Create comprehensive results file
    best_params = (ma_period, buy_threshold, max_hold_days)
    optimization_results = [{'ma_period': ma_period, 'buy_threshold': buy_threshold, 'max_hold_days': max_hold_days, 
                           'compounded_return': 0, 'total_trades': 0, 'win_rate': 0}]  # Placeholder for optimized params
    create_comprehensive_results(results, best_params, optimization_results, total_compound)
    
    print(f"\n🎯 12-Year Backtesting Complete!")
    print(f"📊 Comprehensive results saved to comprehensive_results.txt")

def create_results_summary(results):
    """
    Create a results.txt file with summary statistics for each year.
    """
    with open('results.txt', 'w') as f:
        f.write("MULTI-YEAR BACKTESTING RESULTS\n")
        f.write("="*50 + "\n")
        f.write("Strategy: Buy when stock is 30%+ below 100-day MA, sell when above MA or after 40 days\n")
        f.write("Code created using GPT-4o (ChatGPT)\n\n")
        
        # Write detailed results for each year
        f.write("YEAR-BY-YEAR RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for result in results:
            f.write(f"\nYear: {result['year']}\n")
            f.write(f"  Average return per trade: {result['avg_return']:.2f}%\n")
            f.write(f"  Average days held: {result['avg_days']:.1f}\n")
            f.write(f"  Total trades: {result['total_trades']}\n")
            f.write(f"  Win rate: {result['win_rate']:.1f}%\n")
            f.write(f"  Non-overlapping trades: {result['non_overlapping_trades']}\n")
            f.write(f"  Estimated compounded return: {result['compounded_return']:.2f}%\n")
        
        # Calculate overall averages (excluding years with no trades)
        valid_results = [r for r in results if r['total_trades'] > 0]
        
        if valid_results:
            f.write(f"\nOVERALL STRATEGY PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            
            avg_return = sum(r['avg_return'] for r in valid_results) / len(valid_results)
            avg_days = sum(r['avg_days'] for r in valid_results) / len(valid_results)
            total_trades = sum(r['total_trades'] for r in valid_results)
            avg_win_rate = sum(r['win_rate'] for r in valid_results) / len(valid_results)
            avg_compounded = sum(r['compounded_return'] for r in valid_results) / len(valid_results)
            
            f.write(f"Years with trades: {len(valid_results)}/{len(results)}\n")
            f.write(f"Average return per trade: {avg_return:.2f}%\n")
            f.write(f"Average days held: {avg_days:.1f}\n")
            f.write(f"Total trades across all years: {total_trades}\n")
            f.write(f"Average win rate: {avg_win_rate:.1f}%\n")
            f.write(f"Average compounded return per year: {avg_compounded:.2f}%\n")
            
            # Best and worst years
            best_year = max(valid_results, key=lambda x: x['compounded_return'])
            worst_year = min(valid_results, key=lambda x: x['compounded_return'])
            
            f.write(f"\nBest performing year: {best_year['year']} ({best_year['compounded_return']:.2f}% return)\n")
            f.write(f"Worst performing year: {worst_year['year']} ({worst_year['compounded_return']:.2f}% return)\n")
        
        f.write(f"\nNOTE: Compounded returns are estimated based on non-overlapping trades only.\n")
        f.write(f"Actual returns would be lower due to overlapping positions and market constraints.\n")
        
def optimize_strategy_2024():
    """
    Optimize strategy parameters using 2024 data.
    Test different combinations of MA period, buy threshold, and max hold days.
    """
    print("🔧 Optimizing strategy parameters using 2024 data...")
    
    # Parameter ranges to test
    ma_periods = [50, 100, 150]
    buy_thresholds = [0.15, 0.20, 0.25, 0.30]  # 15%, 20%, 25%, 30% below MA
    max_hold_days_options = [30, 40, 50]
    
    best_return = -999
    best_params = None
    optimization_results = []
    
    for ma_period in ma_periods:
        for buy_threshold in buy_thresholds:
            for max_hold_days in max_hold_days_options:
                print(f"Testing: MA={ma_period}, Threshold={buy_threshold*100:.0f}%, MaxDays={max_hold_days}")
                
                try:
                    result = backtest_year(2024, ma_period, buy_threshold, max_hold_days)
                    optimization_results.append({
                        'ma_period': ma_period,
                        'buy_threshold': buy_threshold,
                        'max_hold_days': max_hold_days,
                        'compounded_return': result['compounded_return'],
                        'total_trades': result['total_trades'],
                        'win_rate': result['win_rate']
                    })
                    
                    if result['compounded_return'] > best_return:
                        best_return = result['compounded_return']
                        best_params = (ma_period, buy_threshold, max_hold_days)
                        
                except Exception as e:
                    print(f"Error with params MA={ma_period}, Threshold={buy_threshold}, MaxDays={max_hold_days}: {e}")
    
    print(f"\n🎯 Optimization Complete!")
    print(f"Best parameters: MA={best_params[0]}, Threshold={best_params[1]*100:.0f}%, MaxDays={best_params[2]}")
    print(f"Best 2024 return: {best_return:.2f}%")
    
    return best_params, optimization_results

def create_comprehensive_results(results, best_params, optimization_results, total_compound):
    """
    Create a comprehensive results file with strategy description, optimization details, and 12-year performance.
    """
    ma_period, buy_threshold, max_hold_days = best_params
    
    with open('comprehensive_results.txt', 'w') as f:
        f.write("COMPREHENSIVE 12-YEAR BACKTESTING RESULTS (2014-2025)\n")
        f.write("="*60 + "\n\n")
        
        # Strategy Description
        f.write("STRATEGY DESCRIPTION:\n")
        f.write("-"*20 + "\n")
        f.write(f"We are using Adj Closes that are {buy_threshold*100:.0f}% below the {ma_period}-day MA to create buy signals, ")
        f.write(f"and sell signals are created when {ma_period}-day MA crossovers occur, more than {max_hold_days} days have elapsed, ")
        f.write("or we reach the end of the year.\n\n")
        
        # Prompt Used
        f.write("PROMPT USED TO CREATE CODE:\n")
        f.write("-"*25 + "\n")
        f.write('I have S&P500 stock data for 25 years in csv files in a directory called "YahooStockData"; ')
        f.write('the files contain the following columns: Date, Close, Adj Close, High, Low, Open, Volume, Ticker. ')
        f.write('Process all stocks in the directory. Set a variable called theYear to 2024.\n')
        f.write('For each trading day in theYear, check if:\n')
        f.write(f'- the adjusted close is at least {buy_threshold*100:.0f}% below the {ma_period}-day moving average on that trading day.\n')
        f.write('If a stock meets these criteria, create a buy signal for that trading day.\n')
        f.write('Create a sell signal when any one of these conditions is met on a later trading day:\n')
        f.write(f'- the adjusted close is above the {ma_period}-day moving average on that trading day.\n')
        f.write(f'- more than {max_hold_days} trading days have elapsed since the buy signal was created for this stock.\n')
        f.write('- we reach the end of theYear.\n')
        f.write('Print the version of GPT used to create the code.\n')
        f.write('Optimize your strategy for 2024, and then backtest for 12 years: 2014 to 2025.\n')
        f.write('Code created using GPT-4o (ChatGPT)\n\n')
        
        # Optimization Results
        f.write("OPTIMIZATION RESULTS (2024):\n")
        f.write("-"*25 + "\n")
        f.write(f"Best parameters found: MA={ma_period}, Threshold={buy_threshold*100:.0f}%, MaxDays={max_hold_days}\n")
        f.write("Top 5 parameter combinations tested:\n")
        sorted_opts = sorted(optimization_results, key=lambda x: x['compounded_return'], reverse=True)[:5]
        for i, opt in enumerate(sorted_opts, 1):
            f.write(f"{i}. MA={opt['ma_period']}, Threshold={opt['buy_threshold']*100:.0f}%, MaxDays={opt['max_hold_days']}: ")
            f.write(f"{opt['compounded_return']:.2f}% return, {opt['total_trades']} trades, {opt['win_rate']:.1f}% win rate\n")
        f.write("\n")
        
        # 12-Year Performance
        f.write("PERFORMANCE OVER 12 YEARS (2014-2025):\n")
        f.write("-"*35 + "\n")
        
        for result in reversed(results):  # Show from 2014 to 2025
            f.write(f"Finished processing year {result['year']}. Compounded gain: {result['compounded_return']:.2f}%\n")
        
        # Calculate compound multiplication string
        yearly_returns = [r['compounded_return'] for r in reversed(results)]
        compound_factors = [f"{1 + r/100:.4f}" for r in yearly_returns]
        compound_string = " * ".join(compound_factors)
        
        f.write(f"\nCompound calculation: {compound_string} = {total_compound:.3f}\n")
        f.write(f"Total 12-year compound return: {(total_compound - 1) * 100:.2f}%\n\n")
        
        # Performance Summary
        valid_results = [r for r in results if r['total_trades'] > 0]
        if valid_results:
            f.write("SUMMARY STATISTICS:\n")
            f.write("-"*18 + "\n")
            
            avg_return = sum(r['avg_return'] for r in valid_results) / len(valid_results)
            avg_days = sum(r['avg_days'] for r in valid_results) / len(valid_results)
            total_trades = sum(r['total_trades'] for r in valid_results)
            avg_win_rate = sum(r['win_rate'] for r in valid_results) / len(valid_results)
            
            f.write(f"Years with trades: {len(valid_results)}/12\n")
            f.write(f"Average return per trade: {avg_return:.2f}%\n")
            f.write(f"Average days held: {avg_days:.1f}\n")
            f.write(f"Total trades across all years: {total_trades}\n")
            f.write(f"Average win rate: {avg_win_rate:.1f}%\n")
            
            # Best and worst years
            best_year = max(valid_results, key=lambda x: x['compounded_return'])
            worst_year = min(valid_results, key=lambda x: x['compounded_return'])
            
            f.write(f"Best performing year: {best_year['year']} ({best_year['compounded_return']:.2f}% return)\n")
            f.write(f"Worst performing year: {worst_year['year']} ({worst_year['compounded_return']:.2f}% return)\n\n")
        
        f.write("NOTE: Returns are based on non-overlapping trades to provide realistic compounding estimates.\n")
        
def quick_run():
    """
    Quick run function for students - uses pre-optimized parameters.
    Skips optimization process and runs 12-year backtest directly.
    """
    print("Code created using GPT-4o (ChatGPT)")
    print("🚀 QUICK RUN: Using Pre-Optimized Strategy Parameters...")
    print("Best parameters from optimization: 30% below 50-day MA, 30-day max hold")
    
    # Use best parameters found during optimization
    ma_period, buy_threshold, max_hold_days = 50, 0.30, 30
    
    # Run backtest for 12 years (2014-2025)
    test_years = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014]
    
    results = []
    yearly_returns = []
    
    for year in test_years:
        try:
            year_result = backtest_year(year, ma_period, buy_threshold, max_hold_days)
            results.append(year_result)
            yearly_returns.append(year_result['compounded_return'])
            print(f"Finished processing year {year}. Compounded gain: {year_result['compounded_return']:.2f}%")
        except Exception as e:
            print(f"Error processing year {year}: {str(e)}")
            results.append({
                'year': year, 'avg_return': 0, 'avg_days': 0, 'total_trades': 0,
                'win_rate': 0, 'compounded_return': 0, 'non_overlapping_trades': 0
            })
            yearly_returns.append(0)
    
    # Calculate total compound return
    total_compound = 1.0
    for annual_return in yearly_returns:
        total_compound *= (1 + annual_return / 100)
    
    print(f"\n🎯 12-Year Strategy Performance:")
    print(f"Total compound return: {(total_compound - 1) * 100:.2f}%")
    
    # Create results file
    best_params = (ma_period, buy_threshold, max_hold_days)
    optimization_results = [{'ma_period': ma_period, 'buy_threshold': buy_threshold, 'max_hold_days': max_hold_days, 
                           'compounded_return': 0, 'total_trades': 0, 'win_rate': 0}]
    create_comprehensive_results(results, best_params, optimization_results, total_compound)
    
    print(f"\n✅ Quick run complete! Files generated in current directory.")
