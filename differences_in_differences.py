#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:18:21 2024

@author: KartikPatel
"""
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import sklearn
import black
from sklearn.linear_model import LogisticRegression

# Read in file
DiD = pd.read_csv(
    "/Users/KartikPatel/Desktop/MSBA/MSBA_Capstone/modeling_dataset_5Y_LogTranformOnly_20240421.csv"
)

# Code to remove 2017 (treatment year) and 2020 (*covid year*)
DiD_Original = DiD.copy()
DiD = DiD.loc[DiD["year"] != 2017, :]

# Update post column to show 0 for pre-treatment and 1 for post-treatment
DiD.loc[DiD["year"] < 2017, "post"] = 0
DiD.loc[DiD["year"] >= 2017, "post"] = 1

# #Pre-covid and post-covid
# DiD.loc[DiD["year"]<2020,"post_covid"]=0
# DiD.loc[DiD["year"]>=2021,"post_covid"]=1

# CDs that need to be dropped - 3 districts that missing data, 8(9) that reecieved Relay programs later down the road
cds_to_drop = [
    "Staten_Island_3",
    "Staten_Island_2",
    "Brooklyn_7",
    "Manhattan_6",
    "Bronx_3&6",
    "Manhattan_3",
    "Queens_12",
    "Brooklyn_14",
]

# Assign appropriate districts to treatment and control groups + remove cds that have missing data or recieved Relay programs down the road
treatment_group = []
control_group = []
for cd in DiD.district.unique():
    if cd in cds_to_drop:
        DiD = DiD.loc[DiD["district"] != cd]
    elif cd in ["Bronx_7", "Staten_Island_1", "Manhattan_12"]:
        treatment_group.append(cd)
    else:
        control_group.append(cd)

# Update post column to show 0 for not treatment group and 1 for treatment group
DiD["treatment"] = 0
for cd in treatment_group:
    DiD.loc[DiD["district"] == cd, "treatment"] = 1

# Create treatment post interaction term
DiD["treatment_post"] = DiD["treatment"] * DiD["post"]


# Create column for each borough, containing binary values:
for borough in DiD.borough.unique():
    DiD[borough] = 0
    DiD.loc[DiD["borough"] == borough, borough] = 1

# Create column for each district, containing binary value
for district in DiD.district.unique():
    DiD[district] = 0
    DiD.loc[DiD["district"] == district, district] = 1

# Create dictionary for each column andindex:
DiD_dict = {}
for index, value in enumerate(DiD.columns):
    DiD_dict[index] = value


# function to calculate linear regression
def linear_reg(time_frame, iv_list):
    DiD_filtered = DiD.loc[DiD["year"].isin(time_frame), :]
    X = DiD_filtered[iv_list]
    y = DiD_filtered["log_overdose_deaths_per_100k"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    result = model.fit(cov_type="HC3")
    return result.summary()


def linear_reg_borough(time_frame, iv_list, b_list):
    DiD_filtered = DiD.loc[DiD["year"].isin(time_frame), :]
    DiD_filtered = DiD.loc[DiD["borough"].isin(b_list), :]
    X = DiD_filtered[iv_list]
    y = DiD_filtered["log_overdose_deaths_per_100k"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    result = model.fit(cov_type="HC3")
    return result.summary()


# Average opioid deaths perk 100k per year for treatment group and control group
average_by_group = pd.DataFrame()
average_by_group["year"] = DiD.year.unique()
average_by_group[["control", "treatment"]] = ""


for year in average_by_group.year:
    for i in range(0, 2):
        condition = (DiD["treatment"] == i) & (DiD["year"] == year)
        average_by_group.loc[
            average_by_group["year"] == year, average_by_group.columns[i + 1]
        ] = round(
            statistics.mean(DiD.loc[condition, "log_overdose_deaths_per_100k"]), 3
        )
# Review
print(average_by_group)


# Plot before treatment
plt.plot(
    average_by_group["year"][:2],
    average_by_group["treatment"][:2],
    label="treatment",
    color="purple",
)
plt.plot(
    average_by_group["year"][:2],
    average_by_group["control"][:2],
    label="control",
    color="gray",
)
plt.xticks(average_by_group["year"][:2])
plt.xlabel("year")
plt.ylabel("Average Opioid Deaths")
plt.legend(["treatment", "control"], loc=0, frameon=False)
plt.title("Average Deaths per 100k")
plt.show()

# Plot before and after treatment
plt.plot(
    average_by_group["year"][0:2],
    average_by_group["treatment"][0:2],
    label=None,
    color="purple",
)
plt.plot(
    average_by_group["year"][2:4],
    average_by_group["treatment"][2:4],
    label="treatment",
    color="purple",
)
plt.plot(
    average_by_group["year"][0:2],
    average_by_group["control"][0:2],
    label=None,
    color="gray",
)
plt.plot(
    average_by_group["year"][2:4],
    average_by_group["control"][2:4],
    label="control",
    color="gray",
)
plt.xticks(list(average_by_group["year"][:4]) + [2017])
plt.axvline(x=2017, color="black", linestyle="--")
plt.xlabel("year")
plt.ylabel("Average Opioid Deaths")
plt.legend(loc=0, frameon=False)
plt.title("Average Deaths per 100k")
plt.show()


# ORIGINAL ANALYSIS
# Create 1st regression model with only treatment, post,Interaction Term, and post_covid variables
time_frame = [2015, 2016, 2018, 2019]
IVs = ["treatment", "post", "treatment_post"]
result = linear_reg(time_frame, IVs)
print(result)


# Average opioid deaths per 100k by borough per year
opd_by_borough = pd.DataFrame()
opd_by_borough["year"] = DiD.year.unique()

for borough in DiD.borough.unique():
    opd_by_borough[borough] = ""

for b, borough in enumerate(
    ["Bronx", "Brooklyn", "Staten_Island", "Manhattan", "Queens"]
):
    for y, year in enumerate(opd_by_borough.year):
        rates = list(
            (
                DiD.loc[
                    (DiD["year"] == year) & (DiD["borough"] == borough),
                    "log_overdose_deaths_per_100k",
                ].values
            )
        )
        opd_by_borough.iloc[y, b + 1] = round(statistics.mean(rates), 2)


# #Plotting
# colors = {'Bronx': 'Magenta', 'Brooklyn': 'Gray', 'Staten_Island': 'Gray', 'Manhattan': "Purple", 'Queens': 'Gray'}
# for borough in opd_by_borough.columns[1:]:
#     plt.plot(opd_by_borough['year'],opd_by_borough[borough],label=borough,color=colors[borough])

# plt.axvline(x=2017, color='green', linestyle='--')
# plt.axvline(x=2020, color='red', linestyle='--')
# plt.xlabel("year")
# plt.ylabel("Average Opioid Deaths")
# plt.title("Average Opioid Deaths by year")
# plt.legend(['Bronx','Brooklyn','Staten_Island','Manhattan','Queens','Relay Intro','ovid: 2020'],loc = 0,frameon=False)


# Opioid deaths per 100k per year per district
opd_by_district = pd.DataFrame()
opd_by_district["year"] = DiD.year.unique()


for district in DiD.district.unique():
    opd_by_district[district] = ""

for d, district in enumerate(opd_by_district.columns[1:]):
    for y, year in enumerate(opd_by_district.year):
        opd_by_district.iloc[y, d + 1] = round(
            DiD.loc[
                (DiD["year"] == year) & (DiD["district"] == district),
                "log_overdose_deaths_per_100k",
            ].values[0],
            2,
        )

opd_by_district

change_15_16 = pd.DataFrame()
change_15_16["district"] = DiD.district.unique()
change_15_16["delta"] = ""

for district in change_15_16.district:
    final = (
        opd_by_district.loc[opd_by_district["year"] == 2016, district].values[0]
        - opd_by_district.loc[opd_by_district["year"] == 2015, district].values[0]
    )
    change_15_16.loc[change_15_16["district"] == district, "delta"] = final


# DISTRICT WITH SIMILAR YOY 2015 TO 2016 AS BRONX_7
labels = {
    "Brooklyn_15": "yellow",
    "Queens_8": "gray",
    "Bronx_7": "red",
    "Queens_3": "green",
    "Manhattan_12": "blue",
}
plt.figure(figsize=(8, 5))
IVs = [
    "treatment",
    "post",
    "treatment_post",
    "Bronx",
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Staten_Island",
]
for district in labels.keys():
    plt.plot(
        opd_by_district["year"][:2],
        opd_by_district[district][:2],
        label=district,
        color=labels[district],
    )
    plt.plot(
        opd_by_district["year"][2:4],
        opd_by_district[district][2:4],
        color=labels[district],
    )
    IVs.append(district)
plt.xlabel("year")
plt.xticks([2015, 2016, 2017, 2018, 2019])
plt.axvline(x=2017, color="black", linestyle="--")
plt.ylabel("log_Overdose Deaths per 100k")
plt.title("Districts with similar trends from 2015 to 2016 as Bronx 7")
plt.legend(loc="best")

time_frame = [2015, 2016, 2018, 2019]
result = linear_reg(time_frame, IVs)
print(result)


# DISTRICT WITH SIMILAR YOY 2015 TO 2016 AS BRONX_7
labels = {"Bronx_7": "red", "Queens_8": "gray"}
plt.figure(figsize=(8, 5))
IVs = ["treatment", "post", "treatment_post"]
for district in labels.keys():
    plt.plot(
        opd_by_district["year"][:2],
        opd_by_district[district][:2],
        label=district,
        color=labels[district],
    )
    plt.plot(
        opd_by_district["year"][2:4],
        opd_by_district[district][2:4],
        color=labels[district],
    )
    IVs.append(district)
plt.xlabel("year")
plt.xticks([2015, 2016, 2017, 2018, 2019])
plt.axvline(x=2017, color="black", linestyle="--")
plt.ylabel("log_Overdose Deaths per 100k")
plt.title("Districts with similar trends from 2015 to 2016 as Bronx 7")
plt.legend(loc="best")

time_frame = [2015, 2016, 2018, 2019]
result = linear_reg(time_frame, IVs)
print(result)


# DISTRICT WITH SIMILAR YOY 2015 TO 2016 AS MANHATTAN_12

labels = {
    "Bronx_7": "red",
    "Queens_3": "gray",
    "Manhattan_12": "blue",
    "Queens_7": "gray",
    "Brooklyn_2": "gray",
}
plt.figure(figsize=(8, 5))
IVs = [
    "treatment",
    "post",
    "treatment_post",
    "Bronx",
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Staten_Island",
]
for district in labels.keys():
    plt.plot(
        opd_by_district["year"][:2],
        opd_by_district[district][:2],
        label=district,
        color=labels[district],
    )
    plt.plot(
        opd_by_district["year"][2:4],
        opd_by_district[district][2:4],
        color=labels[district],
    )
    IVs.append(district)
plt.xlabel("year")
plt.xticks([2015, 2016, 2017, 2018, 2019])
plt.axvline(x=2017, color="black", linestyle="--")
plt.ylabel("Overdose Deaths per 100k")
plt.title("Districts with similar trends from 2015 to 2016 as Manhattan 12")
plt.legend(loc="best")

time_frame = [2015, 2016, 2018, 2019]
result = linear_reg(time_frame, IVs)
print(result)

# DISTRICT WITH SIMILAR YOY 2015 TO 2016 AS STATEN_ISLAND_1

labels = {
    "Manhattan_10": "gray",
    "Queens_5": "gray",
    "Staten_Island_1": "green",
    "Brooklyn_10": "gray",
    "Brooklyn_3": "gray",
}
plt.figure(figsize=(8, 5))
IVs = [
    "treatment",
    "post",
    "treatment_post",
    "Bronx",
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Staten_Island",
]
for district in labels.keys():
    plt.plot(
        opd_by_district["year"][:2],
        opd_by_district[district][:2],
        label=district,
        color=labels[district],
    )
    plt.plot(
        opd_by_district["year"][2:4],
        opd_by_district[district][2:4],
        color=labels[district],
    )
    IVs.append(district)
plt.xlabel("year")
plt.xticks([2015, 2016, 2017, 2018, 2019])
plt.axvline(x=2017, color="black", linestyle="--")
plt.ylabel("Overdose Deaths per 100k")
plt.title("Districts with similar trends from 2015 to 2016 as Staten Island 1")
plt.legend(loc="best")

time_frame = [2015, 2016, 2018, 2019]
result = linear_reg(time_frame, IVs)
print(result)


# Plotting
color_dict = {}
for district in DiD.district.unique():
    if district == "Bronx_7":
        color_dict[district] = "Red"
    elif district == "Manhattan_12":
        color_dict[district] = "Blue"
    elif district == "Staten_Island_1":
        color_dict[district] = "Green"
    else:
        color_dict[district] = "Gray"

plt.figure(figsize=(15, 10))
for district in opd_by_district.columns[1:]:
    plt.plot(
        opd_by_district["year"],
        opd_by_district[district],
        label=district if district in treatment_group else None,
        color=color_dict[district],
    )
plt.legend()


# Create column for each treatment group, label 1 if occurence is said treatment group, 0 if not

for cd in treatment_group:
    DiD[cd] = 0
    DiD.loc[DiD["district"] == cd, cd] = 1

# ADDITIONAL INTERACTION TERMS
DiD["treatment_post:Manh12"] = DiD["treatment"] * DiD["post"] * DiD["Manhattan_12"]
DiD["treatment_post:Bronx7"] = DiD["treatment"] * DiD["post"] * DiD["Bronx_7"]
DiD["treatment_post:Stat1"] = DiD["treatment"] * DiD["post"] * DiD["Staten_Island_1"]

time_frame = [2015, 2016, 2018, 2019]
IVs = [
    "treatment",
    "post",
    "treatment_post",
    "treatment_post:Manh12",
    "treatment_post:Bronx7",
    "treatment_post:Stat1",
]
result = linear_reg(time_frame, IVs)
print(result)


# FIXED EFFECTS WITH GROUPS FOR TREATMENT DISTRICTS
def create_groups(group_name, group):
    DiD[group_name] = 0
    for cd in group:
        DiD.loc[DiD["district"] == cd, group_name] = 1


Manhattan_group = ["Manhattan_12", "Manhattan_11", "Manhattan_10", "Manhattan_9"]
Mannhattan_group_name = ["Manh12_11_10_9"]
create_groups(Mannhattan_group_name, Manhattan_group)

Bronx_group = ["Bronx_7", "Bronx_5", "Bronx_8", "Bronx_12"]
Bronx_group_name = ["Bronx7_5_8_12"]
create_groups(Bronx_group_name, Bronx_group)

# Stat_group=['Staten_Island_1','Staten_Island_2','Staten_Island_3']
# Stat_group_name=["Stat1_2_3"]
# create_groups(Stat_group_name,Stat_group)

time_frame = [2015, 2016, 2018, 2019]
IVs = [
    "treatment",
    "treatment_post",
    "post",
    "Manh12_11_10_9",
    "Bronx7_5_8_12",
    "Staten_Island_1",
]
result = linear_reg(time_frame, IVs)
print(result)

# FIXED EFFECTS FOR ALL district
time_frame = [2015, 2016, 2018, 2019]
IVs = ["treatment", "post", "treatment_post"] + list(DiD.district.unique())
result = linear_reg(time_frame, IVs)
print(result)


# FIXED EFFECTS FOR BOROUGH
time_frame = [2015, 2016, 2018, 2019]
IVs = ["treatment", "post", "treatment_post"] + list(DiD.borough.unique())
result = linear_reg(time_frame, IVs)
print(result)


DiD_prop = pd.read_csv(
    "/Users/KartikPatel/Desktop/MSBA/MSBA_Capstone/modeling_dataset_5Y_LogTranformOnly_20240421.csv"
)
DiD_prop = DiD_prop.loc[DiD_prop["year"] != 2017, :]

# Update post column to show 0 for pre-treatment and 1 for post-treatment
DiD_prop.loc[DiD_prop["year"] < 2017, "post"] = 0
DiD_prop.loc[DiD_prop["year"] >= 2017, "post"] = 1

# #Pre-covid and post-covid
# DiD.loc[DiD["year"]<2020,"post_covid"]=0
# DiD.loc[DiD["year"]>=2021,"post_covid"]=1

# CDs that need to be dropped - 3 districts that missing data, 8(9) that reecieved Relay programs later down the road
cds_to_drop = [
    "Staten_Island_3",
    "Staten_Island_2",
    "Brooklyn_7",
    "Manhattan_6",
    "Bronx_3&6",
    "Manhattan_3",
    "Queens_12",
    "Brooklyn_14",
]

# Assign appropriate districts to treatment and control groups + remove cds that have missing data or recieved Relay programs down the road
treatment_group = []
control_group = []
for cd in DiD_prop.district.unique():
    if cd in cds_to_drop:
        DiD_prop = DiD_prop.loc[DiD_prop["district"] != cd]
    elif cd in ["Bronx_7", "Staten_Island_1", "Manhattan_12"]:
        treatment_group.append(cd)
    else:
        control_group.append(cd)

# Update post column to show 0 for not treatment group and 1 for treatment group
DiD_prop["treatment"] = 0
for cd in treatment_group:
    DiD_prop.loc[DiD_prop["district"] == cd, "treatment"] = 1

# Create treatment post interaction term
DiD_prop["treatment_post"] = DiD_prop["treatment"] * DiD_prop["post"]


# function to calculate linear regression
def linear_reg_prop(time_frame, iv_list):
    DiD_filtered = DiD_prop.loc[DiD_prop["year"].isin(time_frame), :]
    X = DiD_filtered[iv_list]
    y = DiD_filtered["log_overdose_deaths_per_100k"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    result = model.fit(cov_type="HC3")
    return result.summary()


covariates = [
    "treatment",
    "post",
    "post_covid",
    "treatment_post",
    "log_Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_of_any_race_Puerto_Rican",
    "log_Percent_DISABILITY_STATUS_OF_THE_CIVILIAN_NONINSTITUTIONALIZED_POPULATION_18_to_64_years_With_a_disability",
    "log_Percent_GROSS_RENT_Occupied_units_paying_rent_Less_than_500",
    "log_Percent_YEAR_STRUCTURE_BUILT_Total_housing_units_Built_1990_to_1999",
    "log_cons_Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_All_families_With_related_children_of_the_householder_under_18_years_With_related_children_of_the_householder_under_5_years_only",
    "log_Estimate_Percent_Male_Total_population_AGE_50_to_54_years",
    "log_Percent_HEALTH_INSURANCE_COVERAGE_Civilian_noninstitutionalized_population_With_health_insurance_coverage",
    "log_Percent_YEAR_STRUCTURE_BUILT_Total_housing_units_Built_2000_to_2009",
    "log_Percent_VALUE_Owner_occupied_units_100_000_to_149_999",
    "log_Percent_VETERAN_STATUS_Civilian_population_18_years_and_over_Civilian_veterans",
    "log_cons_Percent_EMPLOYMENT_STATUS_Population_16_years_and_over_In_labor_force_Armed_Forces",
    "log_Percent_SELECTED_MONTHLY_OWNER_COSTS_AS_A_PERCENTAGE_OF_HOUSEHOLD_INCOME_SMOCAPI_Housing_units_with_a_mortgage_excluding_units_where_SMOCAPI_cannot_be_computed_20_0_to_24_9_percent",
    "log_Percent_DISABILITY_STATUS_OF_THE_CIVILIAN_NONINSTITUTIONALIZED_POPULATION_Total_Civilian_Noninstitutionalized_Population_With_a_disability",
    "log_Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_All_families_With_related_children_of_the_householder_under_18_years",
    "log_Percent_VALUE_Owner_occupied_units_50_000_to_99_999",
    "log_Percent_OCCUPANTS_PER_ROOM_Occupied_housing_units_1_00_or_less",
    "log_cons_Percent_HOUSE_HEATING_FUEL_Occupied_housing_units_No_fuel_used",
    "log_Percent_MARITAL_STATUS_Males_15_years_and_over_Divorced",
    "log_Estimate_Percent_Female_Total_population_AGE_70_to_74_years",
    "log_Estimate_Percent_Female_Total_population_AGE_45_to_49_years",
    "log_Percent_LANGUAGE_SPOKEN_AT_HOME_Population_5_years_and_over_Other_Indo_European_languages",
    "log_Percent_RACE_Total_population_One_race",
    "log_cons_Percent_HOUSE_HEATING_FUEL_Occupied_housing_units_Wood",
    "log_Percent_SELECTED_MONTHLY_OWNER_COSTS_AS_A_PERCENTAGE_OF_HOUSEHOLD_INCOME_SMOCAPI_Housing_units_with_a_mortgage_excluding_units_where_SMOCAPI_cannot_be_computed_25_0_to_29_9_percent",
    "log_Percent_GRANDPARENTS_Number_of_grandparents_living_with_own_grandchildren_under_18_years_Grandparents_responsible_for_grandchildren",
    "log_Percent_HOUSE_HEATING_FUEL_Occupied_housing_units_Utility_gas",
    "log_Percent_SELECTED_MONTHLY_OWNER_COSTS_SMOC_Housing_units_without_a_mortgage_400_to_599",
    "log_Percent_SELECTED_MONTHLY_OWNER_COSTS_AS_A_PERCENTAGE_OF_HOUSEHOLD_INCOME_SMOCAPI_Housing_unit_without_a_mortgage_excluding_units_where_SMOCAPI_cannot_be_computed_30_0_to_34_9_percent",
    "log_cons_Percent_GRANDPARENTS_Number_of_grandparents_living_with_own_grandchildren_under_18_years_Years_responsible_for_grandchildren_Less_than_1_year",
    "log_Percent_GROSS_RENT_AS_A_PERCENTAGE_OF_HOUSEHOLD_INCOME_GRAPI_Occupied_units_paying_rent_excluding_units_where_GRAPI_cannot_be_computed_25_0_to_29_9_percent",
    "log_cons_Percent_SELECTED_MONTHLY_OWNER_COSTS_AS_A_PERCENTAGE_OF_HOUSEHOLD_INCOME_SMOCAPI_Housing_unit_without_a_mortgage_excluding_units_where_SMOCAPI_cannot_be_computed_25_0_to_29_9_percent",
    "log_cons_Percent_UNITS_IN_STRUCTURE_Total_housing_units_Boat_RV_van_etc_",
    "log_Percent_SELECTED_MONTHLY_OWNER_COSTS_SMOC_Housing_units_without_a_mortgage_250_to_399",
    "log_Percent_YEAR_STRUCTURE_BUILT_Total_housing_units_Built_1960_to_1969",
    "log_Percent_SCHOOL_ENROLLMENT_Population_3_years_and_over_enrolled_in_school_Nursery_school_preschool",
    "log_Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Asian_alone",
    "log_Percent_YEAR_STRUCTURE_BUILT_Total_housing_units_Built_1940_to_1949",
    "log_Percent_BEDROOMS_Total_housing_units_5_or_more_bedrooms",
    "log_Percent_CLASS_OF_WORKER_Civilian_employed_population_16_years_and_over_Self_employed_in_own_not_incorporated_business_workers",
    "log_Percent_HOUSING_TENURE_Occupied_housing_units_Owner_occupied",
]


# Covariates
time_frame = [2015, 2016, 2018, 2019, 2021]
linear_reg_prop(time_frame, covariates)


# Propensity score

from sklearn.preprocessing import StandardScaler

top_coeff = [
    "log_Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_of_any_race_Puerto_Rican",
    "log_Percent_DISABILITY_STATUS_OF_THE_CIVILIAN_NONINSTITUTIONALIZED_POPULATION_18_to_64_years_With_a_disability",
    "log_Percent_GROSS_RENT_Occupied_units_paying_rent_Less_than_500",
    "log_Percent_YEAR_STRUCTURE_BUILT_Total_housing_units_Built_1990_to_1999",
    "log_cons_Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_All_families_With_related_children_of_the_householder_under_18_years_With_related_children_of_the_householder_under_5_years_only",
    "log_Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Asian_alone",
    "log_Percent_YEAR_STRUCTURE_BUILT_Total_housing_units_Built_1940_to_1949",
    "log_Percent_BEDROOMS_Total_housing_units_5_or_more_bedrooms",
    "log_Percent_CLASS_OF_WORKER_Civilian_employed_population_16_years_and_over_Self_employed_in_own_not_incorporated_business_workers",
    "log_Percent_HOUSING_TENURE_Occupied_housing_units_Owner_occupied",
]

logistic_reg = LogisticRegression(solver="liblinear")

y_treatment = DiD_prop.loc[DiD_prop["year"] == 2016, "treatment"]
X_variables = DiD_prop.loc[DiD_prop["year"] == 2016, top_coeff]


logistic_reg.fit(X_variables, y_treatment)
propensity_scores_16 = logistic_reg.predict_proba(X_variables)[:, 1]
data = pd.DataFrame(
    {
        "propensity_score_16": propensity_scores_16,
        "treatment": y_treatment,
        "district": DiD_prop.loc[DiD_prop["year"] == 2016, "district"],
    }
)

threshold_min = min(data.loc[data["treatment"] == 1, "propensity_score_16"])
threshold_max = max(data.loc[data["treatment"] == 1, "propensity_score_16"])

# Filter control group districts based on the propensity score threshold
matched_control_group = data[
    (data["propensity_score_16"] >= threshold_min)
    & (data["propensity_score_16"] <= threshold_max)
    & (data["treatment"] == 0)
]
matched_control_group = list(matched_control_group["district"].values)


criteria = DiD_prop["year"].isin([2015, 2016, 2018, 2019])
DiD_filtered_prop = DiD_prop.loc[criteria, :]
criteria = DiD_prop["district"].isin(treatment_group + matched_control_group)
DiD_filtered_prop = DiD_filtered_prop.loc[criteria, :]
iv_list = ["treatment", "post", "treatment_post"]
X = DiD_filtered_prop[iv_list]
y = DiD_filtered_prop["log_overdose_deaths_per_100k"]
X = sm.add_constant(X)
model = sm.OLS(y, X)
result = model.fit(cov_type="HC3")
result.summary()



