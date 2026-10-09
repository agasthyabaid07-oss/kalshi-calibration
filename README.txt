KALSHI CALIBRATION STUDY - HOW TO RUN
=====================================

1. ONE-TIME SETUP
   Put all four .py files in ONE folder (for example Documents/kalshi-project).
   Install the libraries once from your terminal:
       Windows (Command Prompt):  py -m pip install requests pandas numpy matplotlib
       Mac (Terminal):            python3 -m pip install requests pandas numpy matplotlib

2. RUN THE STEPS IN ORDER (in IDLE: File -> Open -> the file, then press F5)
   step1_collect.py      downloads market data from Kalshi   (slowest step)
   step2_clean.py        builds the clean table data/analysis.csv
   step3_calibration.py  the main study: charts + results/summary.txt
   step4_backtest.py     could you profit after fees? results/backtest.txt

   Each script creates its own folders (data, figures, results) next to itself.

3. FIRST RUN = SMALL TEST
   step1_collect.py has MAX_MARKETS = 300 at the top. Run everything once with
   that to check it works. Then change it to 5000 or more and run step 1 again
   (it resumes where it stopped), followed by steps 2, 3 and 4.
   With few markets, steps 3 and 4 may say "too few markets" - that is normal.

4. WHAT TO LOOK AT
   figures/1_calibration.png  the headline chart: dots below the diagonal on
                              the left = longshots win less than their price says
   results/summary.txt        Brier score, longshot test, recalibration slope
   results/backtest.txt       profit after spreads and fees, tested on unseen data

5. IF SOMETHING BREAKS
   Copy the full red error message and paste it to Claude.
