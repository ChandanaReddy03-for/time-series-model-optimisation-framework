import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import base64
from PIL import Image


# Function to fetch stock data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data.index = data.index.tz_localize(None)  # Remove timezone information
    return data['Close']

# Function for data preprocessing
def preprocess_data(data):
    log_data = np.log(data)
    z_score_data = (data - data.mean()) / data.std()
    smoothed_data = data.rolling(window=20).mean()
    rolling_mean = data.rolling(window=20).mean()
    rolling_std = data.rolling(window=20).std()
    pct_change = data.pct_change()
    return log_data, z_score_data, smoothed_data, rolling_mean, rolling_std, pct_change

# Function for EWMA model
def ewma_model(data, alpha):
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_ewma = train_data.ewm(alpha=alpha).mean()

    forecast = []
    last_value = train_ewma.iloc[-1]
    for _ in range(len(test_data)):
        forecast.append(last_value)
        last_value = alpha * test_data.iloc[_] + (1 - alpha) * last_value

    # Future predictions
    avg_daily_change = np.mean(np.diff(forecast[-30:]))
    std_dev = train_data.std()
    future_days = 180
    for _ in range(future_days):
        next_value = last_value + avg_daily_change + np.random.normal(0, std_dev/2)
        forecast.append(next_value)
        last_value = next_value

    future_dates = pd.date_range(start=test_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    forecast_index = test_data.index.union(future_dates)
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

    lower_ci = forecast_df['Forecast'] - 1.96 * std_dev
    upper_ci = forecast_df['Forecast'] + 1.96 * std_dev

    return train_data, test_data, forecast_df, lower_ci, upper_ci

# GARCH model functions
def garch_likelihood(params, returns):
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    return np.sum(np.log(sigma2) + returns**2 / sigma2)

def fit_garch(returns):
    initial_params = [np.var(returns), 0.1, 0.8]
    bounds = [(0, None), (0, 1), (0, 1)]
    result = minimize(garch_likelihood, initial_params, args=(returns,), bounds=bounds, method='L-BFGS-B')
    return result.x

def calculate_garch_vol(params, returns):
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    return np.sqrt(sigma2)

def forecast_garch_with_noise(params, returns, n_steps, noise_level=0.1):
    omega, alpha, beta = params
    last_var = omega + alpha * returns[-1]**2 + beta * np.var(returns)
    forecasts = np.zeros(n_steps)
    for i in range(n_steps):
        noise = np.random.normal(0, noise_level * np.sqrt(last_var))
        forecasts[i] = np.sqrt(last_var) + noise
        last_var = omega + alpha * (forecasts[i]**2) + beta * last_var
    return forecasts

def garch_model(data):
    returns = 100 * data.pct_change().dropna()
    
    split_point = int(len(returns) * 0.8)
    train_returns = returns[:split_point]
    test_returns = returns[split_point:]
    
    # Ensure train_returns and test_returns are Series
    train_returns = pd.Series(train_returns)
    test_returns = pd.Series(test_returns)
    
    params = fit_garch(train_returns)
    in_sample_vol = calculate_garch_vol(params, train_returns)
    
    noise_level = 0.1
    conditional_vol = forecast_garch_with_noise(params, train_returns, len(test_returns), noise_level)
    
    conf_level = 0.95
    z_score = np.abs(np.random.normal(0, 1, 10000)).mean()
    lower_ci = conditional_vol - z_score * conditional_vol / np.sqrt(2)
    upper_ci = conditional_vol + z_score * conditional_vol / np.sqrt(2)
    
    future_steps = 30
    future_volatility = forecast_garch_with_noise(params, returns, future_steps, noise_level)
    future_lower_ci = future_volatility - z_score * future_volatility / np.sqrt(2)
    future_upper_ci = future_volatility + z_score * future_volatility / np.sqrt(2)
    
    return train_returns, test_returns, in_sample_vol, conditional_vol, lower_ci, upper_ci, future_volatility, future_lower_ci, future_upper_ci, params

# def prophet_model(data):
#     # Prepare data for Prophet
#     df = data.reset_index()
#     df.columns = ['ds', 'y']
    
#     # Remove timezone information
#     df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    
#     df = df.dropna()

#     # Split the data
#     split_point = int(len(df) * 0.8)
#     train_data = df[:split_point]
#     test_data = df[split_point:]

#     # Create and fit the model
#     model = Prophet(daily_seasonality=True)
#     model.fit(train_data)

#     # Make future dataframe for predictions
#     future = model.make_future_dataframe(periods=len(test_data) + 180)
#     forecast = model.predict(future)

#     # Extract forecast for test period and future
#     forecast_test = forecast[forecast['ds'].isin(test_data['ds'])]
#     forecast_future = forecast[forecast['ds'] > test_data['ds'].max()]

#     # Calculate metrics
#     mse = mean_squared_error(test_data['y'], forecast_test['yhat'])
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(test_data['y'], forecast_test['yhat'])

#     return model, forecast, train_data, test_data, forecast_test, forecast_future, mse, rmse, mae
# Streamlit app
st.title('Time Series Model Optimization Framework')



# Function to set the logo in the page
# Function to set the logo in the page
def set_logo():
    # Load the image
    image = Image.open('logo.png')
    
    # Create a column layout
    col1, col2, col3 = st.columns([1,1,1])
    
    # Display the image in the third column
    with col3:
        st.image(image, width=230)  # Width is 200

# Call the function to set the logo
set_logo()

# Custom CSS for styling
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#4B0082,#8A2BE2);  /* Indigo to BlueViolet */
        color: white;
    }
    .Widget>label {
        color: white;
        font-family: 'Helvetica', sans-serif;
    }
    .stRadio > div{
        flex-direction:row;
        background-color: #F0F8FF;  /* AliceBlue */
        border-radius: 10px;
        padding: 10px;
    }
    .stRadio label {
        padding: 10px;
        border-radius: 5px;
        color: white;
        margin: 5px;
        transition: all 0.3s ease-in-out;
    }
    .stRadio label:nth-of-type(1) { background-color: #20B2AA; }  /* LightSeaGreen */
    .stRadio label:nth-of-type(2) { background-color: #6495ED; }  /* CornflowerBlue */
    .stRadio label:nth-of-type(3) { background-color: #DA70D6; }  /* Orchid */
    .stRadio label:nth-of-type(4) { background-color: #FF7F50; }  /* Coral */
    .stRadio label:nth-of-type(5) { background-color: #32CD32; }  /* LimeGreen */
    .stRadio label:hover {
        opacity: 0.8;
        cursor: pointer;
    }
    .reportview-container {
        background: #F0FFF0;  /* HoneyDew */
    }
    .main .block-container {
        background: white;
        padding: 3rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #8B4513;  /* SaddleBrown */
    }
    /* Add padding to the top of the main content to make room for the logo */
    .main .block-container {
        padding-top: 5rem;
    }
    
    /* Ensure the sidebar title doesn't overlap with the logo */
    .sidebar .sidebar-content {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Stock Analysis Dashboard")

tabs = {
    "Data Input": "📥 Data Input",
    "Raw Data": "📊 Raw Data",
    "Preprocessed Data": "🔧 Preprocessed Data",
    "Model Analysis": "🧮 Model Analysis",
    "Conclusion": "📝 Conclusion"
}

selected_tab = st.sidebar.radio("Navigation", list(tabs.keys()), format_func=lambda x: tabs[x])

# Main content based on selected tab
if selected_tab == "Data Input":
    st.header("📥 Data Input")
    # Your existing Data Input code here

elif selected_tab == "Raw Data":
    st.header("📊 Raw Data")
    # Your existing Raw Data code here

elif selected_tab == "Preprocessed Data":
    st.header("🔧 Preprocessed Data")
    # Your existing Preprocessed Data code here

elif selected_tab == "Model Analysis":
    st.header("🧮 Model Analysis")
    # Your existing Model Analysis code here

elif selected_tab == "Conclusion":
    st.header("📝 Conclusion")
    # Your existing Conclusion code here

# Global variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None

if selected_tab == "Data Input":
    st.header("Data Input")
    ticker = st.text_input('Enter stock ticker (e.g., AAPL, CAH, GSK):', 'AAPL')
    start_date = st.date_input('Start date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End date', pd.to_datetime('2023-12-31'))
    
    if st.button("Fetch Data"):
        st.session_state.data = get_stock_data(ticker, start_date, end_date)
        st.success("Data fetched successfully!")

elif selected_tab == "Raw Data":
    st.header("Raw Stock Data")
    if st.session_state.data is not None:
        # Line chart
        st.subheader("Line Chart")
        st.line_chart(st.session_state.data)
        
        # Table view
        st.subheader("Data Table")
        st.dataframe(st.session_state.data)
        
        # Option to download the raw data as CSV
        csv = st.session_state.data.to_csv(index=True)
        st.download_button(
            label="Download raw data as CSV",
            data=csv,
            file_name="raw_stock_data.csv",
            mime="text/csv",
        )
    else:
        st.warning("Please fetch data in the Data Input tab first.")


elif selected_tab == "Preprocessed Data":
    st.header("Preprocessed Data")
    if st.session_state.data is not None:
        if st.session_state.preprocessed_data is None:
            st.session_state.preprocessed_data = preprocess_data(st.session_state.data)
        
        log_data, z_score_data, smoothed_data, rolling_mean, rolling_std, pct_change = st.session_state.preprocessed_data
        
        preprocess_option = st.selectbox('Select preprocessing method', 
                                         ['Log Normalization', 'Z-Score Normalization', 'Smoothed Data', 
                                          'Rolling Mean', 'Rolling Std', 'Percent Change'])

        def plot_data_and_histogram(data, title):
            st.line_chart(data)
            st.write("Statistical Values:")
            st.write(data.describe())
            
            # Convert to DataFrame if it's a Series
            if isinstance(data, pd.Series):
                data = data.to_frame()
            
            # Plot histogram
            fig = px.histogram(data, nbins=20, title=f'Histogram of {title}',
                               labels={'value': title}, template='plotly_dark')
            fig.update_layout(
                xaxis_title=title,
                yaxis_title='Frequency',
                bargap=0.2,
                bargroupgap=0.1
            )
            # Calculate and add density line
            for column in data.columns:
                density = gaussian_kde(data[column].dropna())
                x_vals = np.linspace(data[column].min(), data[column].max(), 1000)
                density_vals = density(x_vals)
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=density_vals * len(data) * (data[column].max() - data[column].min()) / 20,
                    mode='lines',
                    name=f'Density ({column})',
                    line=dict(color='red')
                ))
            st.plotly_chart(fig)

        if preprocess_option == 'Log Normalization':
            plot_data_and_histogram(log_data, 'Log Normalized Data')
        elif preprocess_option == 'Z-Score Normalization':
            plot_data_and_histogram(z_score_data, 'Z-Score Normalized Data')
        elif preprocess_option == 'Smoothed Data':
            plot_data_and_histogram(smoothed_data, 'Smoothed Data')
        elif preprocess_option == 'Rolling Mean':
            plot_data_and_histogram(rolling_mean, 'Rolling Mean')
        elif preprocess_option == 'Rolling Std':
            plot_data_and_histogram(rolling_std, 'Rolling Standard Deviation')
        else:  # Percent Change
            plot_data_and_histogram(pct_change, 'Percent Change')

    else:
        st.warning("Please fetch data in the Data Input tab first.")

elif selected_tab == "Model Analysis":
    st.header("Model Analysis")
    
    if st.session_state.data is not None:
        model_choice = st.selectbox('Select model', ['EWMA', 'GARCH', 'Kalman Filter'])
        
        if model_choice == 'EWMA':
            alpha = st.slider('Select alpha value', 0.1, 0.9, 0.3, 0.1)
            train_data, test_data, forecast_df, lower_ci, upper_ci = ewma_model(st.session_state.data, alpha)

            # Calculate metrics
            mse = mean_squared_error(test_data, forecast_df['Forecast'][:len(test_data)])
            rmse = sqrt(mse)
            mae = mean_absolute_error(test_data, forecast_df['Forecast'][:len(test_data)])
            r2 = r2_score(test_data, forecast_df['Forecast'][:len(test_data)])

            st.write(f'MSE: {mse:.4f}')
            st.write(f'RMSE: {rmse:.4f}')
            st.write(f'MAE: {mae:.4f}')
            st.write(f'R2: {r2:.2f}')

            # Create plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data, mode='lines', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Train', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=test_data.index, y=test_data, mode='lines', name='Test', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Predictions', line=dict(color='orange')))

            # Adding confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_df.index.tolist()+forecast_df.index.tolist()[::-1],
                y=upper_ci.tolist()+lower_ci.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))

            # Adding vertical line to separate actual data from future predictions
            fig.add_vline(x=test_data.index[-1], line_dash="dash", line_color="gray")

            fig.update_layout(
                height=600, 
                title_text=f"{model_choice} Model Results with Confidence Intervals and Future Predictions",
                xaxis_title="Date",
                yaxis_title="Stock Price",
                legend_title="Legend"
            )
            fig.update_xaxes(type='date')
            st.plotly_chart(fig)

            # Residuals plot
            residuals = test_data - forecast_df['Forecast'][:len(test_data)]
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(x=test_data.index, y=residuals, mode='lines', name='Residuals', line=dict(color='purple')))
            fig_residuals.add_shape(type="line", x0=test_data.index[0], y0=0, x1=test_data.index[-1], y1=0,
                                    line=dict(color="red", width=2, dash="dash"))
            fig_residuals.update_layout(
                height=400, 
                title_text="Residuals Plot",
                xaxis_title="Date",
                yaxis_title="Residuals",
                showlegend=False
            )
            fig_residuals.update_xaxes(type='date')
            st.plotly_chart(fig_residuals)

        elif model_choice == 'GARCH':
            train_returns, test_returns, in_sample_vol, conditional_vol, lower_ci, upper_ci, future_volatility, future_lower_ci, future_upper_ci, params = garch_model(st.session_state.data)

            # Calculate metrics
            mse = mean_squared_error(test_returns**2, conditional_vol**2)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_returns.abs(), conditional_vol)

            st.write(f'MSE: {mse:.4f}')
            st.write(f'RMSE: {rmse:.4f}')
            st.write(f'MAE: {mae:.4f}')
            

            st.write("Estimated GARCH parameters:")
            st.write(f"omega: {params[0]:.6f}")
            st.write(f"alpha: {params[1]:.6f}")
            st.write(f"beta: {params[2]:.6f}")

            # Plot the results
# Plot the results
            fig = go.Figure()
            full_returns = pd.concat([train_returns, test_returns])
            fig.add_trace(go.Scatter(x=full_returns.index, y=full_returns, mode='lines', name='Actual Returns', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=train_returns.index, y=in_sample_vol, mode='lines', name='In-sample Volatility', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=test_returns.index, y=conditional_vol, mode='lines', name='Predicted Volatility', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=test_returns.index, y=upper_ci, mode='lines', name='Upper CI', line=dict(color='rgba(255,0,0,0.2)')))
            fig.add_trace(go.Scatter(x=test_returns.index, y=lower_ci, mode='lines', name='Lower CI', line=dict(color='rgba(255,0,0,0.2)'), fill='tonexty'))
            fig.update_layout(
                height=600,
                title='GARCH(1,1) Volatility Forecast',
                xaxis_title='Date',
                yaxis_title='Returns / Volatility',
                legend_title='Legend'
                )
            st.plotly_chart(fig)

            # Plot future volatility forecast
            future_dates = pd.date_range(start=test_returns.index[-1], periods=len(future_volatility)+1, freq='D')[1:]
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_volatility, mode='lines', name='Future Volatility Forecast', line=dict(color='green')))
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_upper_ci, mode='lines', name='Upper CI', line=dict(color='rgba(0,255,0,0.2)')))
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_lower_ci, mode='lines', name='Lower CI', line=dict(color='rgba(0,255,0,0.2)'), fill='tonexty'))
            fig_future.update_layout(
                height=400,
                title='GARCH(1,1) Future Volatility Forecast',
                xaxis_title='Date',
                yaxis_title='Predicted Volatility',
                legend_title='Legend'
            )
            st.plotly_chart(fig_future)

            st.write(f"\nFuture volatility forecast for the next {len(future_volatility)} days:")
            st.write(future_volatility)
    

        elif model_choice == 'Kalman Filter':
            class KalmanFilter:
                def __init__(self, initial_state, initial_estimate_error, process_variance, measurement_variance):
                    self.x = initial_state
                    self.P = initial_estimate_error
                    self.Q = process_variance
                    self.R = measurement_variance
                def predict(self):
                    self.x = self.x
                    self.P = self.P + self.Q
                    return self.x
                def update(self, measurement):
                    self.K = self.P / (self.P + self.R)
                    self.x = self.x + self.K * (measurement - self.x)
                    self.P = (1 - self.K) * self.P

            def apply_kalman_filter(data, process_variance, measurement_variance):
                kf = KalmanFilter(data[0], 1, process_variance, measurement_variance)
                filtered_data = []
                for measurement in data:
                    kf.predict()
                    kf.update(measurement)
                    filtered_data.append(kf.x)
                return filtered_data

            AAPL_Master_Data = st.session_state.data
            if not isinstance(AAPL_Master_Data.index, pd.DatetimeIndex):
                AAPL_Master_Data.index = pd.to_datetime(AAPL_Master_Data.index)
            split_point = int(len(AAPL_Master_Data) * 0.8)
            train_AAPL_data = AAPL_Master_Data[:split_point]
            test_AAPL_data = AAPL_Master_Data[split_point:]

            # Apply Kalman Filter to training data
            process_variance = 1e-5
            measurement_variance = 0.1**2
            train_AAPL_data_kalman = apply_kalman_filter(train_AAPL_data, process_variance, measurement_variance)

            # Forecast for test period and future
            kf = KalmanFilter(train_AAPL_data_kalman[-1], 1, process_variance, measurement_variance)
            forecast_AAPL = []
            std_dev_AAPL = train_AAPL_data.std()
            np.random.seed(42)
            for measurement in test_AAPL_data:
                prediction = kf.predict()
                # Add noise to the prediction
                noisy_prediction = prediction + np.random.normal(0, std_dev_AAPL * 0.1)  # 10% of std_dev as noise
                forecast_AAPL.append(noisy_prediction)
                kf.update(measurement)

            # Forecast for future 180 days
            future_days = 180
            avg_daily_change_AAPL = np.mean(np.diff(forecast_AAPL[-30:]))
            for _ in range(future_days):
                prediction = kf.predict()
                # Add noise to the prediction
                noisy_prediction = prediction + avg_daily_change_AAPL + np.random.normal(0, std_dev_AAPL * 0.2)  # 20% of std_dev as noise
                forecast_AAPL.append(noisy_prediction)
                kf.update(noisy_prediction)

            future_dates_AAPL = pd.date_range(start=test_AAPL_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
            forecast_index = test_AAPL_data.index.union(future_dates_AAPL)
            forecast_df_AAPL = pd.DataFrame(forecast_AAPL, index=forecast_index, columns=['Forecast'])
            lower_ci_AAPL = forecast_df_AAPL['Forecast'] - 1.96 * std_dev_AAPL
            upper_ci_AAPL = forecast_df_AAPL['Forecast'] + 1.96 * std_dev_AAPL

            # Calculate metrics
            mse = mean_squared_error(test_AAPL_data, forecast_df_AAPL['Forecast'][:len(test_AAPL_data)])
            rmse = sqrt(mse)
            mae = mean_absolute_error(test_AAPL_data, forecast_df_AAPL['Forecast'][:len(test_AAPL_data)])
            r2 = r2_score(test_AAPL_data, forecast_df_AAPL['Forecast'][:len(test_AAPL_data)])

            st.write(f'MSE: {mse:.4f}')
            st.write(f'RMSE: {rmse:.4f}')
            st.write(f'MAE: {mae:.4f}')
            st.write(f'R2: {r2:.2f}')

            # Create plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=AAPL_Master_Data.index, y=AAPL_Master_Data, mode='lines', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=train_AAPL_data.index, y=train_AAPL_data, mode='lines', name='Train', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=test_AAPL_data.index, y=test_AAPL_data, mode='lines', name='Test', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=forecast_df_AAPL.index, y=forecast_df_AAPL['Forecast'], mode='lines', name='Predictions', line=dict(color='orange')))

            # Adding confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_df_AAPL.index.tolist()+forecast_df_AAPL.index.tolist()[::-1],
                y=upper_ci_AAPL.tolist()+lower_ci_AAPL.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))

            # Adding vertical line to separate actual data from future predictions
            fig.add_vline(x=test_AAPL_data.index[-1], line_dash="dash", line_color="gray")

            fig.update_layout(
                height=600, 
                title_text=f"{model_choice} Model Results with Confidence Intervals and Future Predictions",
                xaxis_title="Date",
                yaxis_title="Stock Price",
                legend_title="Legend"
            )
            fig.update_xaxes(type='date')
            st.plotly_chart(fig)

            # Residuals plot
            residuals = test_AAPL_data - forecast_df_AAPL['Forecast'][:len(test_AAPL_data)]
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(x=test_AAPL_data.index, y=residuals, mode='lines', name='Residuals', line=dict(color='purple')))
            fig_residuals.add_shape(type="line", x0=test_AAPL_data.index[0], y0=0, x1=test_AAPL_data.index[-1], y1=0,
                                    line=dict(color="red", width=2, dash="dash"))
            fig_residuals.update_layout(
                height=400, 
                title_text="Residuals Plot",
                xaxis_title="Date",
                yaxis_title="Residuals",
                showlegend=False
            )
            fig_residuals.update_xaxes(type='date')
            st.plotly_chart(fig_residuals)

    else:
        st.warning("Please fetch data in the Data Input tab first.")

elif selected_tab == "Conclusion":

    st.header("Conclusion")
    st.write("""
                 This dashboard provides a comprehensive framework for time series model optimization, 
                 focusing on the Exponential Weighted Moving Average (EWMA) and GARCH(1,1) models. Users can input different stocks, 
                 select date ranges, and adjust model parameters to analyze and predict stock prices and volatility. 
                 The preprocessing methods and visualization options allow for in-depth analysis of the data. 
                 The model evaluation metrics and plots provide insights into the performance of each model.
                 
                 Key features:
                 
                 1. Interactive data input and model parameter selection
                 2. Visualization of raw and preprocessed data
                 3. EWMA model analysis with adjustable alpha
                 4. GARCH(1,1) model for volatility forecasting
                 5. Performance metrics (MSE, RMSE, MAE)
                 6. Plotly charts for actual vs predicted prices/volatility, including confidence intervals and future predictions
                 7. Residuals plot for model diagnostics (EWMA)
                 8. Future volatility forecast (GARCH)
                 
                 For further improvements, consider:
                 
                 1. Adding more advanced models like ARIMA or LSTM
                 2. Implementing cross-validation for more robust model evaluation
                 3. Incorporating sentiment analysis or other external factors that might influence stock prices
                 4. Providing options for portfolio optimization based on the predictions
                 5. Allowing for comparison between different models
                 6. Implementing automated model selection based on performance metrics
                 """)