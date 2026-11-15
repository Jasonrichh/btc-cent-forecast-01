# btc-cent-forecast-01 — Bitcoin 4-hour-ahead forecaster

[![make.fun · btc-cent-forecast-01](https://make.fun/badge/btc-cent-forecast-01.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)
[![rank](https://make.fun/badge/btc-cent-forecast-01/Jasonrichh/btc-cent-forecast-01/rank.svg)](https://make.fun/bounty/btc-cent-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)
[![reward](https://make.fun/badge/btc-cent-forecast-01/Jasonrichh/btc-cent-forecast-01/reward.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)

Entry for the [make.fun bounty **btc-cent-forecast-01**](https://make.fun/bounty/btc-cent-forecast-01): forecast Bitcoin's price 240 minutes (4 hours) after each checkpoint, 12 times over 48 hours. Score is mean absolute error in cents; a prize needs beating the house persistence baseline and placing in the top 100.

## Program

`main.ts` (Node 22, standard library only, offline, deterministic) reads every `<id>.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) — 4,320 one-minute closes of the Orca cbBTC/USDC pool over the 72 hours up to the checkpoint — and writes `<id>.out` with exactly `{"forecasts":[{"case":"<id>","price":<number>}]}` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Model

An equal-weight ensemble of four shrunk trend + mean-reversion components. Each forecasts a 240-minute log-return `r` applied to the last close (`forecast = last * exp(r)`, `|r|` capped at 2%):

```
r_k = shrink_k * 240 * OLS-slope of log price over the last W_k minutes
      - lambda_k * (log(last) - EWMA of log price, halflife H_k)
```

| component | W (min) | shrink | H (min) | lambda |
|-----------|---------|--------|---------|--------|
| 1 | 1440 | 0.15 | 60 | 0.25 |
| 2 | 1440 | 0.25 | 60 | 0.25 |
| 3 | 720 | 0.25 | 120 | 0.25 |
| 4 | 720 | 0.15 | 60 | 0.35 |

Two regularities survive out-of-sample backtesting at the 4-hour horizon, where BTC is close to a random walk so every signal must be heavily shrunk:

1. **Slow trend persistence** — the 12–24 h OLS slope, shrunk ~5×, points the right way more often than not. Unshrunk momentum hurts badly.
2. **Pool microstructure mean reversion** — the last close's deviation from its 60–120 min EWMA partly reverts within 4 h (a big trade displaces the pool price; arbitrage pulls it back).

Short-horizon momentum (EWMA of 1-minute returns) and strong mean reversion both lose decisively and are not used. Flat series fall back to persistence; bad ticks are carried forward; the forecast is clamped to the scorer's price grid.

## Backtest (MAE in cents, hourly checkpoints, 72 h lookback, 240 min horizon)

| data | checkpoints | persistence | this model | improvement |
|------|-------------|-------------|------------|-------------|
| Orca pool 1-min closes, 20 Oct–15 Nov 2026 | 553 | 51,387 | 50,654 | **−1.43%** |
| Same pool, first half | 239 | 43,606 | 43,143 | −1.06% |
| Same pool, second half | 311 | 54,240 | 53,270 | −1.79% |
| Binance BTCUSDT 1-min, 1–15 Nov 2026 | 265 | 51,472 | 50,954 | −1.01% |
| Kit practice cases p00–p05 | 6 | 44,522 | 45,425 | +2.03% |

The model beats persistence on every large-sample split (800+ checkpoints). The six practice cases are too few to overturn that — their mean has very high variance — and the practice is graded on past prices that pay nothing.

## Files

- `main.ts` — the entrypoint (the whole model, ~190 lines).
- `LICENSE` — MIT (from the kit's starter).
