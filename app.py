import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        riders = float(request.form["riders"])
        drivers = float(request.form["drivers"])
        past_rides = float(request.form["past_rides"])
        ratings = float(request.form["ratings"])
        duration = float(request.form["duration"])
        category = request.form["category"]
        loyalty = request.form["loyalty"]
        time = request.form["time"]
        vehicle_type = request.form["vehicle_type"]
        

        supply_gap = riders - drivers
        demand = riders / drivers if drivers > 0 else riders

        if demand <= 1.5:
            demand_status = "Low Demand"
        elif demand <= 2.5:
            demand_status = "Moderate"
        elif demand <= 3.5:
            demand_status = "High Demand"
        else:
            demand_status = "Extreme Surge"

        unit_price = 0.0

        time_map = {"Morning": 1, "Night": 1, "Afternoon": 2, "Evening": 2}
        loyalty_map = {"Gold": 1, "Silver": 2, "Regular": 3}
        vehicle_map = {"Premium": 1, "Economy": 2}

        time_priority = time_map.get(time, 1)
        loyalty_priority = loyalty_map.get(loyalty, 3)
        vehicle_priority = vehicle_map.get(vehicle_type, 2)

        input_data = pd.DataFrame([{
            "Riders": riders,
            "Drivers": drivers,
            "Category": category,
            "Loyalty": loyalty,
            "Time": time,
            "Vehicle_Type": vehicle_type,
            "Past_Rides": past_rides,
            "Ratings": ratings,
            "Duration": duration,
            "Supply_Gap": supply_gap,
            "Demand": demand,
            "Demand_Status": demand_status,
            "Unit_Price": unit_price,
            "Time_Priority": time_priority,
            "Vehicle_Priority": vehicle_priority,
            "Loyalty_Priority": loyalty_priority
        }])

        prediction = model.predict(input_data)[0]
        predicted_cost = round(float(prediction), 2)

        return render_template(
            "index.html", 
            prediction_text=f"Estimated Ride Fare: ${predicted_cost}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
