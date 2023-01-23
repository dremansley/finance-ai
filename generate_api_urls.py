import calendar
import time
import requests
import pandas as pd
from datetime import datetime
import ai_model

api_key = "G7zO8ZpnCiHb5Wyom5qNe_waoRTZ6zYE"
limit = 50000
base_url = "https://api.polygon.io/v2/aggs/ticker/X:ETHUSD/range/1/minute/"

f = open("api_urls.txt", "w")

for year in range(2021, 2023):
    for month in range(1, 13):
        first_day = f"{year}-{month}-01"
        last_day = f"{year}-{month}-{calendar.monthrange(year, month)[1]}"

        first_day = datetime.strptime(first_day, "%Y-%m-%d").strftime("%Y-%m-%d")
        last_day = datetime.strptime(last_day, "%Y-%m-%d").strftime("%Y-%m-%d")

        print(first_day)
        # Make the request to fetch the data
        api_url = f"{base_url}{first_day}/{last_day}?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
        
        print(f"*** Making Request to fetch the data for {first_day} - {last_day}")
        response = requests.get(api_url)
        response_data = response.json()
        results_count = response_data.get("resultsCount")
        print(f"{results_count} results found for this month")
        if results_count > 0:
            f.write(api_url)

            df = pd.DataFrame.from_dict(response_data["results"])
            file_name = f"data/{first_day}-{last_day}.csv"
            df.to_csv(f"{file_name}", encoding='utf-8', index=False)

            # Pass the file to the AI model for training
            ai_model.process_data(file_name)

f.close()