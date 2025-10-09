import pandas as pd
import requests
from collections import defaultdict

# Mapping of major Indian cities to their states/UTs
CITY_TO_STATE = {
    "Delhi": "Delhi",
    "Mumbai": "Maharashtra",
    "Bangalore": "Karnataka",
    "Chennai": "Tamil Nadu",
    "Kolkata": "West Bengal",
    "Hyderabad": "Telangana",
    "Pune": "Maharashtra",
    "Ahmedabad": "Gujarat",
    "Lucknow": "Uttar Pradesh",
    "Jaipur": "Rajasthan",
    # Add more major cities as needed
}

# Function to fetch OpenAQ real-time data
def fetch_openaq_data():
    url = "https://api.openaq.org/v2/latest?country=IN&limit=1000"
    resp = requests.get(url)
    data = resp.json()
    
    rows = []
    for item in data.get("results", []):
        city = item.get("city")
        measurements = item.get("measurements", [])
        pollutants = {"PM2.5":0,"PM10":0,"NO2":0,"SO2":0,"O3":0,"CO":0}
        for m in measurements:
            param = m["parameter"].upper()
            if param in pollutants:
                pollutants[param] = m["value"]
        if city:
            row = {
                "City": city,
                "PM2.5": pollutants["PM2.5"],
                "PM10": pollutants["PM10"],
                "NO2": pollutants["NO2"],
                "SO2": pollutants["SO2"],
                "O3": pollutants["O3"],
                "CO": pollutants["CO"]
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv("data/air_quality_data.csv", index=False)
    return df

# Aggregate state-level AQI
def aggregate_state_aqi(df, indian_states_uts):
    from collections import defaultdict
    state_pollutants = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        city = row["City"]
        state = CITY_TO_STATE.get(city)
        if state:
            for param in ["PM2.5","PM10","NO2","SO2","O3","CO"]:
                state_pollutants[state][param].append(row[param])
    city_data = {}
    for state in indian_states_uts:
        pollutants = {}
        if state in state_pollutants:
            for p in ["PM2.5","PM10","NO2","SO2","O3","CO"]:
                pollutants[p] = round(sum(state_pollutants[state][p])/len(state_pollutants[state][p]),2)
            AQI = max(pollutants.values())
        else:
            pollutants = {"PM2.5":0,"PM10":0,"NO2":0,"SO2":0,"O3":0,"CO":0}
            AQI = 0
        city_data[state] = {"AQI": AQI, "pollutants": pollutants, "history":[AQI]}
    return city_data
