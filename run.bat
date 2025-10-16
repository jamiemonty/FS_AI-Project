@echo off
REM AI Stock Trading Backtesting Project - Windows Batch Script
REM Created using GPT-4o (ChatGPT)

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="setup" goto setup
if "%1"=="quick" goto quick
if "%1"=="run" goto run
if "%1"=="clean" goto clean
if "%1"=="results" goto results
if "%1"=="status" goto status
goto help

:help
echo AI Stock Trading Backtesting Project - Batch Commands
echo =====================================================
echo.
echo Setup Commands:
echo   run.bat setup     - Create virtual environment and install dependencies
echo   run.bat clean     - Clean up generated files
echo.
echo Run Commands:
echo   run.bat quick     - Quick analysis (uses pre-optimized params, 10-15 min)
echo   run.bat run       - Full analysis (includes optimization, 30+ min)
echo.
echo Info Commands:
echo   run.bat results   - Show latest results summary
echo   run.bat status    - Check project status
echo   run.bat help      - Show this help
echo.
echo Examples:
echo   run.bat setup     First time setup
echo   run.bat quick     Run quick analysis
goto end

:setup
echo Setting up virtual environment...
python -m venv .venv
echo Installing dependencies...
.venv\Scripts\pip.exe install pandas numpy matplotlib
echo Setup complete! Use 'run.bat quick' to start analysis.
goto end

:quick
if not exist ".venv" (
    echo Virtual environment not found. Run 'run.bat setup' first.
    goto end
)
echo Starting quick analysis with pre-optimized parameters...
.venv\Scripts\python.exe -c "from playGround import quick_run; quick_run()"
goto end

:run
if not exist ".venv" (
    echo Virtual environment not found. Run 'run.bat setup' first.
    goto end
)
echo Starting full analysis with optimization (this may take 30+ minutes)...
.venv\Scripts\python.exe main.py
goto end

:clean
echo Cleaning up generated files...
del *_perf.csv 2>nul
del comprehensive_results.txt 2>nul
del results.txt 2>nul
for /d %%i in (plots_*) do rmdir /s /q "%%i" 2>nul
for /d %%i in (backup_*) do rmdir /s /q "%%i" 2>nul
echo Cleanup complete!
goto end

:results
if exist "comprehensive_results.txt" (
    echo === LATEST RESULTS SUMMARY ===
    powershell -Command "Get-Content comprehensive_results.txt | Select-Object -First 20"
    echo.
    echo Full results available in comprehensive_results.txt
) else (
    echo No results found. Run 'run.bat quick' or 'run.bat run' first.
)
goto end

:status
echo === PROJECT STATUS ===
if exist ".venv\Scripts\python.exe" (
    echo ✓ Python environment: Ready
) else (
    echo ✗ Python environment: Not found
)
if exist "YahooStockData" (
    echo ✓ Stock data: Available
) else (
    echo ✗ Stock data: Missing
)
if exist "comprehensive_results.txt" (
    echo ✓ Results: Available
) else (
    echo ✗ Results: Not generated
)
goto end

:end
