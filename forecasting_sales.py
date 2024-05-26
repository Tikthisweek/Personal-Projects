#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:50:44 2024

@author: KartikPatel
"""

import pandas as pd
import statsmodels.api as sm
import black
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from datetime import datetime
import numpy as np
import csv
import statistics

# Load data
quince = pd.read_excel("/Users/KartikPatel/Desktop/Quince/Quince_Python.xlsx")
predict = pd.read_excel("/Users/KartikPatel/Desktop/Quince/Predict.xlsx")

# Convert date columns
quince["Week_Start"] = pd.to_datetime(quince["Week_Start"], infer_datetime_format=True)
predict["Week_Start"] = pd.to_datetime(predict["Week_Start"])

# Set index
IndexedQuince = quince.set_index(["Week_Start"])
predict.set_index("Week_Start", inplace=True)

# Filter data for specific products and colors
sweater_black = quince[
    (quince["Product_Description"] == "Mongolian_Cashmere_Crewneck_Sweater")
    & (quince["Color"] == "Black")
    & (quince["Week_Start"] >= "2020-08-16")
]
sweater_heather = quince[
    (quince["Product_Description"] == "Mongolian_Cashmere_Crewneck_Sweater")
    & (quince["Color"] == "Heather_Grey")
    & (quince["Week_Start"] >= "2020-08-16")
]
sweater_oatmeal = quince[
    (quince["Product_Description"] == "Mongolian_Cashmere_Crewneck_Sweater")
    & (quince["Color"] == "Oatmeal")
]

dress_black = quince[
    (quince["Product_Description"] == "Mongolian_Cashmere_Turtleneck_Sweater_Dress")
    & (quince["Color"] == "Black")
    & (quince["Week_Start"] >= "2020-09-06")
]
dress_heather = quince[
    (quince["Product_Description"] == "Mongolian_Cashmere_Turtleneck_Sweater_Dress")
    & (quince["Color"] == "Heather_Grey")
]
dress_oatmeal = quince[
    (quince["Product_Description"] == "Mongolian_Cashmere_Turtleneck_Sweater_Dress")
    & (quince["Color"] == "Oatmeal")
]


# Calculate rolling statistics
def calculate_rolling_stats(data):
    rolmean = data["Unit_Sales"].rolling(window=7).mean()
    rolstd = data["Unit_Sales"].rolling(window=7).std()
    return rolmean, rolstd


# Sweater
rolmean_sweater_black, rolstd_sweater_black = calculate_rolling_stats(sweater_black)
rolmean_sweater_heather, rolstd_sweater_heather = calculate_rolling_stats(
    sweater_heather
)
rolmean_sweater_oatmeal, rolstd_sweater_oatmeal = calculate_rolling_stats(
    sweater_oatmeal
)

# Dress
rolmean_dress_black, rolstd_dress_black = calculate_rolling_stats(dress_black)
rolmean_dress_heather, rolstd_dress_heather = calculate_rolling_stats(dress_heather)
rolmean_dress_oatmeal, rolstd_dress_oatmeal = calculate_rolling_stats(dress_oatmeal)

# Growth factors
weeks_in_month = 4.34524  # Average weeks in a month
customer_growth_rate = 1.06 ** (
    predict.index.to_series().diff().dt.days.cumsum() / (7 * weeks_in_month)
)
organic_growth = 0.5 * 3 * (predict.index.to_series().diff().dt.days.cumsum() / 365.25)
economic_growth_rate = 1.035 ** (
    predict.index.to_series().diff().dt.days.cumsum() / 365.25
)
ecommerce_growth_rate = 1.08 ** (
    predict.index.to_series().diff().dt.days.cumsum() / 365.25
)

# Combine regressors
regressors = pd.DataFrame(
    {
        "customer_growth_rate": customer_growth_rate,
        "organic_growth": organic_growth,
        "economic_growth_rate": economic_growth_rate,
        "ecommerce_growth_rate": ecommerce_growth_rate,
    },
    index=predict.index,
)

regressors.fillna(1, inplace=True)


# Forecast sales using Auto ARIMA
def forecast_sales(data, product_name):
    model = auto_arima(
        data["Unit_Sales"],
        trend="t",
        seasonal=True,
        exogenous=regressors,
        stepwise=True,
        m=52,
    )
    predictions = pd.Series(model.predict(n_periods=len(predict)))
    predictions.index = predict.index
    predictions.name = "Unit Sales"
    prediction_all[product_name] = predictions


prediction_all = pd.DataFrame()

forecast_sales(sweater_black, "sweater_black")
forecast_sales(sweater_heather, "sweater_heather")
forecast_sales(sweater_oatmeal, "sweater_oatmeal")
forecast_sales(dress_black, "dress_black")
forecast_sales(dress_heather, "dress_heather")
forecast_sales(dress_oatmeal, "dress_oatmeal")

prediction_all["Week_Start"] = predict.index
prediction_orig = prediction_all
prediction_all = prediction_all[prediction_all["Week_Start"] >= "2022-07-31"]


# Calculate weighted average unit sales for sweaters
def calculate_weighted_sales(black, heather, oatmeal):
    result = pd.DataFrame(columns=["Week_Start", "Unit_Sales"])
    for week in black["Week_Start"]:
        if week < pd.Timestamp("2020-09-13"):
            continue
        week_sales_black = black[black["Week_Start"] == week]["Unit_Sales"].values[0]
        week_sales_heather = heather[heather["Week_Start"] == week][
            "Unit_Sales"
        ].values[0]
        week_sales_oatmeal = oatmeal[oatmeal["Week_Start"] == week][
            "Unit_Sales"
        ].values[0]
        total_sales = week_sales_black + week_sales_heather + week_sales_oatmeal
        weighted_black = week_sales_black * (week_sales_black / total_sales)
        weighted_heather = week_sales_heather * (week_sales_heather / total_sales)
        weighted_oatmeal = week_sales_oatmeal * (week_sales_oatmeal / total_sales)
        unit_sales = statistics.mean(
            [weighted_black, weighted_heather, weighted_oatmeal]
        )
        new_row = pd.DataFrame({"Week_Start": [week], "Unit_Sales": [unit_sales]})
        result = pd.concat([result, new_row], ignore_index=True)
    return result


sweater_red = calculate_weighted_sales(sweater_black, sweater_heather, sweater_oatmeal)
forecast_sales(sweater_red, "sweater_red")

dress_blue = calculate_weighted_sales(dress_black, dress_heather, dress_oatmeal)
forecast_sales(dress_blue, "dress_blue")

# Save predictions to CSV
with open("Predictions11.csv", "a") as file:
    writer = csv.DictWriter(file, fieldnames=["Week_Start", "Type", "Unit_Sales"])
    writer.writeheader()
    for index, row in prediction_all.iterrows():
        for col, value in row.items():
            if col == "Week_Start":
                continue
            if col in ["sweater_black", "sweater_heather"]:
                writer.writerow({"Week_Start": index, "Type": col, "Unit_Sales": value})


# Monte Carlo inventory simulation
def monte_carlo_inventory_simulation(
    prediction_all, start, lead_time, reorder_point, reorder_quantity, name, product
):
    final_results = pd.DataFrame(
        columns=[
            "Product",
            "Simulation",
            "Stockout",
            "Reorder_Point",
            "Reorder_Quantity",
            "Reorders",
            "Minreorders",
            "Total_Sales",
            "Remaining_Inventory",
        ]
    )
    for i in range(50):
        current_inventory = int(start)
        predictions = prediction_all.loc[:, ["Week_Start", name]]
        weeks_passed = 0
        stockout = 0
        reorders = 0
        minreorders = 0
        total_sales = 0.0
        for index, row in predictions.iterrows():
            if weeks_passed == lead_time + 1:
                weeks_passed = 0
                if reorder_quantity < 100:
                    current_inventory += 100
                    minreorders += 1
                else:
                    current_inventory += reorder_quantity
                    reorders += 1
            forecasted_sales = np.random.poisson(lam=row[name])
            if forecasted_sales <= current_inventory:
                current_inventory -= forecasted_sales
                total_sales += forecasted_sales
            else:
                total_sales += current_inventory
                current_inventory = 0
                stockout += 1
            if current_inventory <= reorder_point:
                weeks_passed += 1
        new_row = pd.DataFrame(
            {
                "Product": [name],
                "Simulation": [i],
                "Stockout": f"{(stockout / 27) * 100:.2f}%",
                "Reorder_Point": reorder_point,
                "Reorder_Quantity": reorder_quantity,
                "Reorders": [reorders],
                "Minreorders": [minreorders],
                "Total_Sales": [total_sales],
                "Remaining_Inventory": [current_inventory],
            }
        )
        final_results = pd.concat([final_results, new_row], ignore_index=True)
    return final_results


# Run Monte Carlo simulation
simulation_results = pd.DataFrame()
products = [
    sweater_black,
    sweater_heather,
    sweater_oatmeal,
    sweater_red,
    dress_black,
    dress_heather,
    dress_oatmeal,
    dress_blue,
]
names = [
    "sweater_black",
    "sweater_heather",
    "sweater_oatmeal",
    "sweater_red",
    "dress_black",
    "dress_heather",
    "dress_oatmeal",
    "dress_blue",
]
lead_list = [4, 4, 4, 4, 7, 7, 7, 7]
start_inv = [
    statistics.mean(prediction_all[name]) * lead for name, lead in zip(names, lead_list)
]

for product, name, start, lead in zip(products, names, start_inv, lead_list):
    std_dev_demand = statistics.stdev(product["Unit_Sales"])
    safety_stock = 1.65 * std_dev_demand * np.sqrt(lead)
    blended_avg_demand = (
        statistics.mean(product["Unit_Sales"]) + statistics.mean(prediction_all[name])
    ) / 2
    rp = blended_avg_demand * lead
    rp_with_safety_stock = rp + safety_stock
    rp_min, rp_max = (
        rp_with_safety_stock - std_dev_demand,
        rp_with_safety_stock + std_dev_demand,
    )
    rq_min, rq_max = ((min(prediction_all[name]) + min(product["Unit_Sales"])) / 2), (
        (max(prediction_all[name]) + max(product["Unit_Sales"])) / 2
    ) * lead * 2
    rp_range = np.arange(rp_min, rp_max + std_dev_demand, std_dev_demand)
    rq_range = np.arange(rq_min, rq_max + std_dev_demand, std_dev_demand)
    for rp in rp_range:
        for rq in rq_range:
            results = monte_carlo_inventory_simulation(
                prediction_all, start, lead, rp, rq, name, product
            )
            simulation_results = pd.concat(
                [simulation_results, results], ignore_index=True
            )

# Save simulation results to CSV
with open("Simulations8.csv", "a") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Product",
            "Simulation",
            "Stockout",
            "Reorder_Point",
            "Reorder_Quantity",
            "Reorders",
            "Minreorders",
            "Total_Sales",
            "Remaining_Inventory",
        ],
    )
    writer.writeheader()
    for index, row in simulation_results.iterrows():
        writer.writerow(row.to_dict())
