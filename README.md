<<<<<<< HEAD
# Time Series Model Optimization Framework

## Project Overview
A Python-based project that analyzes and forecasts stock prices and volatility using historical financial data.

## Technologies Used
- Python, Pandas, NumPy
- yfinance
- Matplotlib, Plotly
- SciPy
- Scikit-learn

## Methodology
- Collected historical stock data for Apple (AAPL), Cardinal Health (CAH), and GSK.
- Applied data preprocessing and normalization techniques.
- Built and evaluated EWMA, GARCH, and Kalman Filter models.
- Compared model performance using MSE, RMSE, MAE, and R² metrics.
- Visualized stock trends, forecasts, and model evaluation results.

## Key Features
- Historical stock data collection
- Data preprocessing and exploratory analysis
- Stock price forecasting and volatility analysis
- Model performance comparison
- Interactive data visualizations

## How to Run
1. Install the required Python libraries.
2. Open `Project Code(1).ipynb` in Jupyter Notebook or Google Colab.
3. Run the notebook cells in order.

## Objective
To explore time-series forecasting techniques and evaluate their performance on historical stock market data.

## time series dashboard
Interactive stock analysis dashboard using Python, Streamlit, EWMA, GARCH(1,1), and Kalman Filters for time-series analysis, volatility forecasting, and interactive visualizations. 

# time-series-model-optimisation-framework
The project aims to explore the efficiency of EWMA, Kalman Filter and GARCH models in forecasting the stock prices for AAPL, CAH and GSK. The study examines these models under different data transformations, including raw, Z-score normalized and log-transformed data. Results show that each of the models offers something different. The EWMA is stable across transformations; the Kalman Filter with log-transformed data has a stronger ability to externalize; and GARCH displays entirely different stock volatility patterns according to which you are talking about. In its final section, this project proposes integrating NLP techniques to make financial predictions more accurate by incorporating textual data of a financial nature. Through this encompassing approach, it seeks to gain a more nuanced appreciation for stock price changes and to enhance computing capabilities within markets.
>>>>>>> 6c95590a35bf68e62bf78a6c3c9899fe9ba08ba4
