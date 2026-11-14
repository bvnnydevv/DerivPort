# DerivPort Signal

Your balanced **model + momentum** ranking algorithm. It blends the model's return estimate with recent price momentum, then discounts measured forecast error and volatility:

```text
score = 0.6 × predicted_return + 0.4 × momentum
        − k × (0.5 × uncertainty + 0.25 × volatility)
```

The default is **k = 1**. All return-like inputs use the same 21-session horizon and numerical percentage-point scale: `3.5` means 3.5%. k is dimensionless. The coefficients are a transparent research hypothesis, chosen for readability rather than optimized to past results. A ranking score is not a return forecast or a buy/sell recommendation.

## Use in the app

Load **DerivPort Signal** in Algorithm builder, inspect the substituted arithmetic, and run a historical backtest on the comparison group and dates you choose. Download the algorithm file and current feature-input CSV from the builder to reproduce the rankings independently.

Keep only inputs available at your decision cutoff. `uncertainty` must be measured from completed past out-of-sample forecast errors. The file does not fetch data, train the model, or generate future observations. Future scenarios through 2037 remain hypothetical and separate from historical backtests.

## Run the downloaded file

The single [derivport_signal.py](derivport_signal.py) file needs Python 3.10+ and **no external packages**. Its input CSV must have these columns:

```csv
ticker,predicted_return,uncertainty,momentum,volatility
```

`volatility` is trailing daily-return sample standard deviation × √21, in percentage points. Error and volatility must be nonnegative; estimates and momentum may be negative. Every input and k must be finite. Each ticker must be unique.

```sh
python3 derivport_signal.py --input features.csv --output ranked.csv --k 1
python3 derivport_signal.py --demo
```

The demo uses explicitly labeled synthetic arithmetic inputs, not market history. Output contains normalized inputs, full-precision `score`, and descending `rank`; ties share their average rank and ticker order only breaks display ties. Without `--output`, ranked CSV is written to standard output.

You can also import `score_row(row, k=1)` or `rank_stocks(rows, k=1)` from the file. `rows` is a list of dictionaries. Input dictionaries are not modified. `NAME`, `FORMULA`, and `DEFAULT_K` expose the preset metadata.

## What each assumption does

- Increasing the model estimate by 1 pp adds **0.6** score units.
- Increasing momentum by 1 pp adds **0.4** score units.
- Increasing uncertainty by 1 pp subtracts **0.5 × k** score units.
- Increasing volatility by 1 pp subtracts **0.25 × k** score units.

These are effects of the explicit ranking arithmetic, not explanations of the regression's feature influence. Rank changes only when a score crosses another stock's score.

## Initial historical check

On the unchanged five-stock demo cache, with **34 complete held-out periods / 170 forecasts** from 2022–2024 and k = 1, mean per-period Spearman rank correlation was:

| Ranking method | Mean Spearman correlation |
| --- | ---: |
| AI model | +0.085 |
| Momentum baseline | −0.044 |
| DerivPort Signal | −0.141 |

This first rule did **not** show strong ranking accuracy and performed worse by this measure than the two reference methods. The check is exploratory: the rule was selected retrospectively, and these values do not describe expanded histories or other stock groups. No profitable strategy or predictive advantage is claimed. Use the builder to understand the tradeoffs and test revised hypotheses on previously unseen periods.
