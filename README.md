# Machine Learning Trading Strategy

This project explores the use of machine learning to forecast financial market movements and evaluate trading strategies using historical data.

## Project Overview
The goal of this project is to build a data-driven trading pipeline that predicts market trends and tests trading performance through backtesting.

The system performs the following steps:
1. Load and clean financial time-series data
2. Generate predictive features
3. Train machine learning models
4. Produce trading signals
5. Evaluate strategy performance using backtesting

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

## Project Structure
- `data_loader.py` – loads and prepares market data
- `features.py` – feature engineering
- `targets.py` – defines prediction targets
- `models.py` – machine learning models
- `backtest.py` – evaluates trading strategy performance
- `tests/` – testing scripts

## Goal
To investigate whether machine learning models can detect patterns in financial markets and generate profitable trading signals.

## Future Improvements
- Add deep learning models
- Improve feature engineering
- Implement real-time trading simulation
