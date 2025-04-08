# 🚕 Sweet Lift Taxi: Predicting Hourly Demand

## Project Overview
Sweet Lift Taxi wants to improve driver availability at airports during busy times — and that starts with smarter predictions. This project explores hourly demand forecasting using historical data from taxi orders, aiming to predict the number of orders for the next hour.

The goal? Build a model with an RMSE below 48. My best model came in at RMSE ~37.84. ✅

## 🛠 Tools & Techniques
To get the best forecast, I combined traditional time series techniques with machine learning approaches:

- Baseline model — A simple “predict the last hour’s value” approach
- Linear Regression — With lag and rolling features
- SARIMA (Seasonal ARIMA) — Captured strong daily seasonality
- XGBoost — Nearly matched SARIMA’s performance with faster training

Along the way, I explored time series patterns, engineered lag features, and tested multiple modeling strategies to see what stuck.

## 📊 Key Visuals

Here are a few visualizations that helped me tell the story of this time series modeling project:

1. **Hourly taxi orders with outliers highlighted**  
   Helps identify extreme values and motivates the decision to cap or clip outliers during preprocessing.  
   ![Hourly Orders with Outliers](https://github.com/user-attachments/assets/2a2282be-52bb-4f17-9b66-a3a6a79d3345)

2. **Rolling averages (daily and weekly)**  
   Shows strong daily and weekly seasonality in the data using 24- and 168-hour rolling windows.  
   ![Rolling Averages](https://github.com/user-attachments/assets/6f63fbc3-8672-4189-9652-d6dac0058041)

3. **Seasonal decomposition**  
   Breaks the series into observed, trend, seasonality, and residual components — key for deciding to use SARIMA.  
   ![Seasonal Decomposition](https://github.com/user-attachments/assets/b9d11762-aa39-4893-ba12-3b482e2fcc8c)

4. **XGBoost predictions vs actual**  
   Machine learning model tracks real demand closely — a strong performer alongside SARIMA.  
   ![XGBoost Forecast](https://github.com/user-attachments/assets/bdb3d4dc-ff1d-46e6-89fc-44e91dc97a0b)

5. **XGBoost feature importance**  
   Highlights that recent lags and rolling features contributed most to the model's accuracy.  
   ![XGBoost Feature Importance](https://github.com/user-attachments/assets/abe20a21-5e5d-4aea-9458-62289efd219f)

6. **SARIMA predictions vs actual**  
   Best-performing model, capturing daily cycles well — shows SARIMA was a great fit.  
   ![SARIMA Forecast](https://github.com/user-attachments/assets/406b3faa-bd5b-4d2f-a41c-b6decf03977e)

7. **SARIMA residuals over time**  
   Residuals appear randomly distributed with no clear seasonal pattern — confirming good model fit.  
   ![SARIMA Residuals](https://github.com/user-attachments/assets/4641a7a3-a8c1-40eb-9505-a2ad253b8a68)



   

## 🧠 What I Learned
- **Time series = patterns + memory:** Lags and rolling averages really helped both classical and ML models.
- **SARIMA is powerful** — but computationally expensive on large seasonal windows.
- **XGBoost scales better:** It came close to SARIMA’s performance and would be my pick for larger datasets or real-time deployment.
- **Residuals matter:** Checking errors helped confirm that the model captured core structure, not just noise.

## ✅ Results

| Model                  | Description                          | RMSE    | MAE     |
|------------------------|--------------------------------------|---------|---------|
| SARIMAX (m=24, D=1)    | SARIMA with daily seasonality        | **37.84** | 29.01  |
| Linear Regression      | Lag + rolling features               | 39.57   | 30.83  |
| XGBoost                | Tuned gradient boosting              | 39.59   | 30.53  |
| Baseline               | Previous hour’s value                | 55.44   | 43.40  |
| ARIMA(3,1,5)           | Differenced ARMA                     | 61.49   | 47.43  |
| ARMA(3,5)              | Non-differenced ARMA                 | 63.21   | 49.26  |
| AutoRegressive (AR)    | Lags=28, seasonal=True               | 68.07   | 54.30  |

**Why both RMSE and MAE?**  
RMSE penalizes larger errors more heavily, while MAE gives the average error directly in number of orders. Including both helps compare accuracy and stability across models.

## 🧭 Next Steps
If I had more time or compute power, here’s what I’d try:
- Deeper hyperparameter tuning for SARIMA/XGBoost
- Explore multi-seasonality (e.g., daily + weekly patterns)
- Add event-based features for extreme outlier spikes
- Use SHAP values to explain XGBoost decisions more clearly
- Try lightweight LSTM or Prophet models to expand the toolkit

## 📚 Libraries Used
This project uses a mix of data manipulation, time series analysis, and machine learning libraries:
- pandas – For working with time-indexed data and feature engineering
- numpy – Numerical operations and array manipulation
- matplotlib / seaborn – Data visualization and exploratory analysis
- datetime – Handling time-based indexing and features


For the models:   
- scikit-learn – Linear regression and model evaluation 
- statsmodels – Classical time series modeling (ARIMA/SARIMA)
- xgboost – Gradient boosting for time series regression



