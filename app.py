from flask import Flask, render_template, request
import pandas as pd
from collections import defaultdict
import numpy as np
import os

app = Flask(__name__)

# ------------------------
# States and Cities
# ------------------------
INDIAN_STATES_UTS = [
    "Andaman & Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Delhi", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka", "Kerala",
    "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

CITY_TO_STATE = {
    "Delhi": "Delhi",
    "Mumbai": "Maharashtra",
    "Pune": "Maharashtra",
    "Nagpur": "Maharashtra",
    "Bangalore": "Karnataka",
    "Mysore": "Karnataka",
    "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu",
    "Hyderabad": "Telangana",
    "Kolkata": "West Bengal",
    "Kochi": "Kerala",
    "Thiruvananthapuram": "Kerala",
    "Jaipur": "Rajasthan",
    "Lucknow": "Uttar Pradesh",
    "Kanpur": "Uttar Pradesh",
    "Patna": "Bihar",
    "Guwahati": "Assam",
    "Imphal": "Manipur",
    "Agartala": "Tripura",
    "Gangtok": "Sikkim",
    "Aizawl": "Mizoram",
    "Shillong": "Meghalaya",
    "Itanagar": "Arunachal Pradesh",
    "Ranchi": "Jharkhand",
    "Bhopal": "Madhya Pradesh",
    "Raipur": "Chhattisgarh",
    "Gandhinagar": "Gujarat",
    "Shimla": "Himachal Pradesh",
    "Srinagar": "Jammu & Kashmir",
    "Leh": "Ladakh",
    "Port Blair": "Andaman & Nicobar Islands",
    "Kavaratti": "Lakshadweep",
    "Puducherry": "Puducherry",
    "Panaji": "Goa",
    "Chandigarh": "Punjab"
}

# ------------------------
# Load CSV
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "data", "air_quality_data.csv")
df = pd.read_csv(CSV_FILE)

# ------------------------
# Aggregate city → state
# ------------------------
state_pollutants = defaultdict(lambda: defaultdict(list))
city_history = defaultdict(list)

for _, row in df.iterrows():
    city = row["City"]
    state = CITY_TO_STATE.get(city, city)
    for p in ["PM2.5","PM10","NO2","SO2","O3","CO"]:
        state_pollutants[state][p].append(row[p])
    city_history[state].append(row["AQI"])

# ------------------------
# Build city_data dictionary
# ------------------------
city_data = {}
for state in INDIAN_STATES_UTS:
    if state in state_pollutants:
        pollutants = {p: round(sum(vals)/len(vals),2) for p, vals in state_pollutants[state].items()}
        AQI = city_history[state][-1]  # latest AQI
        history = city_history[state]
    else:
        pollutants = {"PM2.5":0,"PM10":0,"NO2":0,"SO2":0,"O3":0,"CO":0}
        AQI = 0
        history = [0]
    city_data[state] = {"AQI": AQI, "pollutants": pollutants, "history": history}

cities = INDIAN_STATES_UTS

# ------------------------
# AQI color and health advice
# ------------------------
def get_aqi_color(aqi):
    if aqi <= 50: return "bg-green-500"
    if aqi <= 100: return "bg-yellow-400"
    if aqi <= 150: return "bg-orange-500"
    if aqi <= 200: return "bg-red-600"
    if aqi <= 300: return "bg-purple-700"
    return "bg-gray-800"

def get_health_advice(aqi):
    if aqi <= 50: return "Air is Good 🌿. Safe to go outside."
    if aqi <= 100: return "Air is Moderate 🌤. Sensitive groups should take care."
    if aqi <= 150: return "Unhealthy for Sensitive Groups ⚠️. Reduce outdoor activities."
    if aqi <= 200: return "Unhealthy 🔴. Avoid long exposure outdoors."
    if aqi <= 300: return "Very Unhealthy 🟣. Limit outdoor activities."
    return "Hazardous 🛑. Stay indoors."

# ------------------------
# Future AQI prediction (7 days hourly) with realistic fluctuation
# ------------------------
def predict_next_week(state_name):
    history = city_data[state_name]["history"]
    last_aqi = history[-1]
    future_values = []
    future_labels = []

    for i in range(1, 7*24+1):  # 7 days hourly
        change = np.random.normal(0, 5)  # small fluctuation
        next_aqi = max(0, round(last_aqi + change, 2))
        future_values.append(next_aqi)
        future_labels.append(f"T+{i}h")
        last_aqi = next_aqi
    return future_labels, future_values

# ------------------------
# Flask Route
# ------------------------
@app.route("/", methods=["GET","POST"])
def home():
    selected_city_name = cities[0]
    if request.method == "POST":
        selected_city_name = request.form.get("city")

    selected_city = city_data[selected_city_name]

    # Past AQI trend
    history_values = selected_city["history"]
    history_labels = [f"T-{len(history_values)-i}" for i in range(len(history_values))]

    # Predicted AQI trend
    future_labels, future_values = predict_next_week(selected_city_name)

    return render_template(
        "index.html",
        cities=cities,
        selected_city={**selected_city, "name": selected_city_name,
                       "history_values": history_values,
                       "history_labels": history_labels},
        future_labels=future_labels,
        future_values=future_values,
        aqi_color=get_aqi_color(selected_city['AQI']),
        health_advice=get_health_advice(selected_city['AQI'])
    )

if __name__ == "__main__":
    app.run(debug=True)