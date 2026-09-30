# Ride Fare Dynamic Pricing 

An end-to-end Machine Learning project designed to build a dynamic price prediction model for a ride-sharing service. The system goes beyond basic trip duration pricing by incorporating real-time supply-demand dynamics, customer loyalty profiles, time of day, location categories, and vehicle types.

## Business Objective
A leading ride-sharing company currently prices rides primarily based on expected duration. The objective of this project is to implement a dynamic pricing model to:
- Increase Revenue: Leverage real-time supply and demand dynamics to adjust ride pricing dynamically.
- Improve Fleet Allocation: Optimize driver distribution during peak demand periods.
- Maintain Customer Retention: Prevent churn by avoiding over-pricing for loyal customers.

## Dataset & FeaturesThe model utilizes historical ride data containing the following features:
```
----------------------------------------------------------------------------------
| Feature       | Type        | Description                                      |
----------------------------------------------------------------------------------
| Riders        | Numerical   | Number of active riders requesting rides         |
| Drivers       | Numerical   | Number of available drivers                      |
| Past_Rides    | Numerical   | Total completed rides by the customer            |
| Ratings       | Numerical   | Customer/Driver rating                           |
| Duration      | Numerical   | Duration of the trip in minutes                  |
| Cost          | Numerical   | Target Variable: Total ride fare                 |
| Category      | Categorical | Location category (Urban, Suburban, Rural)       |
| Loyalty       | Categorical | Customer loyalty tier (Gold, Silver, Regular)    |
| Time          | Categorical | Time of day (Morning, Afternoon, Evening, Night) |
| Vehicle_Type  | Categorical | Class of vehicle (Economy, Premium)              |
----------------------------------------------------------------------------------
```
## Feature Engineering
To capture non-linear relationship and market dynamics, several derivative features were engineered:
- Supply Gap: Riders - Drivers
- Demand Ratio: Riders / Drivers
- Demand Status: Categorized into Low Demand, Moderate, High Demand, and Extreme Surge using quantile binning (pd.qcut).
- Unit Price: Cost / Duration (Price per minute)
- Priority Mappings & Ordinal Encoding: Encoded order-sensitive variables such as Loyalty tier and Vehicle_Type.

## Data Preprocessing 
- Numerical Features: Median imputation using SimpleImputer followed by standard feature scaling via StandardScaler.
- Categorical Features: Frequent-value imputation using SimpleImputer followed by OneHotEncoder (dropping the first category to avoid multicollinearity).

## Model Evaluation & Selection
- Trained on an 80/20 train-test split (stratified by Demand_Status). 
- Trained by Gradient Boosting Regressor, Random Forest Regressor and Linear Regression.
- Evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R^2 Score.
- Serialized and saved as model.pkl

## Clone the repository
```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
git clone https://github.com/ujjwalkumar14b/Dynamic_Pricing.git
cd Dynamic_Price_Prediction
```

## Author
Ujjwal Kumar
GitHub: [https://github.com/ujjwalkumar14b](https://github.com/ujjwalkumar14b)
