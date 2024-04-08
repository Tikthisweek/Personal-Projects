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



# Read in file
DiD=pd.read_csv("/Users/KartikPatel/Desktop/MSBA/MSBA_Capstone/modeling_LogTranformOnly_20240403.csv")

#Code to remove 2017 (treatment year) and 2020 (*covid year*)
DiD_Original=DiD.copy()
DiD=DiD.loc[DiD['year']!=2017,:]

#Update post column to show 0 for pre-treatment and 1 for post-treatment
DiD.loc[DiD["year"]<2017,"post"]=0
DiD.loc[DiD["year"]>=2017,"post"]=1

# #Pre-covid and post-covid
# DiD.loc[DiD["year"]<2020,"post_covid"]=0
# DiD.loc[DiD["year"]>=2021,"post_covid"]=1



#CDs that need to be dropped - 3 districts that missing data, 8(9) that reecieved Relay programs later down the road
cds_to_drop=['Staten_Island_3','Staten_Island_2','Brooklyn_7',
             'Manhattan_6','Bronx_3&6','Manhattan_3',
             'Queens_12','Brooklyn_14','Manhattan_1',
             'Queens_4','Queens_6']

#Assign appropriate districts to treatment and control groups + remove cds that have missing data or recieved Relay programs down the road
treatment_group=[]
control_group=[]
for cd in DiD.district.unique():
    if cd in cds_to_drop:
        DiD=DiD.loc[DiD['district']!=cd]
    elif cd in ['Bronx_7','Staten_Island_1','Manhattan_12']:
        treatment_group.append(cd)
    else:
        control_group.append(cd) 
        
#Update post column to show 0 for not treatment group and 1 for treatment group
DiD["treatment"]=0
for cd in treatment_group:
    DiD.loc[DiD['district']==cd,"treatment"]=1

#Create treatment post interaction term
DiD["treatment:post"]=DiD['treatment']*DiD["post"]

    
#Create column for each borough, containing binary values:
for borough in DiD.borough.unique():
    DiD[borough]=0
    DiD.loc[DiD['borough']==borough,borough]=1

#Create column for each district, containing binary value
for district in DiD.district.unique():
        DiD[district]=0
        DiD.loc[DiD['district']==district,district]=1

#Create dictionary for each column andindex:
DiD_dict={}
for index,value in enumerate(DiD.columns):
    DiD_dict[index]=value


#function to calculate linear regression
def linear_reg(time_frame,iv_list):
    DiD_filtered=DiD.loc[DiD['year'].isin(time_frame),:]
    X=DiD_filtered[iv_list]
    y=DiD_filtered["log_overdose_deaths_per_100k"]
    X=sm.add_constant(X)
    model=sm.OLS(y,X)
    result=model.fit()
    return result.summary()

#Average opioid deaths perk 100k per year for treatment group and control group
average_by_group=pd.DataFrame()
average_by_group["year"]=DiD.year.unique()
average_by_group[["control","treatment"]]=""


for year in average_by_group.year:
    for i in range(0,2):
        condition=(DiD['treatment']==i) & (DiD['year']==year)
        average_by_group.loc[average_by_group['year']==year,average_by_group.columns[i+1]]= round(statistics.mean(DiD.loc[condition,'overdose_deaths_per_100k']),2)
#Review
print(average_by_group)


#Plot before treatment
plt.plot(average_by_group['year'][:2],average_by_group['treatment'][:2],label = 'treatment',color = "purple")
plt.plot(average_by_group['year'][:2],average_by_group['control'][:2],label = "control", color = "gray")
plt.xticks(average_by_group['year'][:2])
plt.xlabel("year")
plt.ylabel("Average Opioid Deaths")
plt.legend(['treatment', 'control'],loc = 0,frameon=False)
plt.title("Average Deaths per 100k")
plt.show()

#Plot before and after treatment
plt.plot(average_by_group['year'][0:2],average_by_group['treatment'][0:2],label = None, color = "purple")
plt.plot(average_by_group['year'][2:4],average_by_group['treatment'][2:4],label = "treatment",color = "purple")
plt.plot(average_by_group['year'][0:2],average_by_group['control'][0:2],label =None, color="gray")
plt.plot(average_by_group['year'][2:4],average_by_group['control'][2:4],label = "control", color = "gray")
plt.xticks([2015,2016,2017,2018,2019])
plt.axvline(x=2017, color='black', linestyle='--')
plt.xlabel("year")
plt.ylabel("Average Opioid Deaths")
plt.legend(loc = 0,frameon=False)
plt.title("Average Deaths per 100k")
plt.show()


#ORIGINAL ANALYSIS
#Create 1st regression model with only treatment, post,Interaction Term, and post_covid variables
time_frame=[2018,2019,2021]
IVs=['treatment','post_covid']
result=linear_reg(time_frame,IVs)
print(result)


#Average opioid deaths per 100k by borough per year
opd_by_borough=pd.DataFrame()
opd_by_borough['year']=DiD.year.unique()

for borough in DiD.borough.unique():
    opd_by_borough[borough]=""

for b,borough in enumerate(['Bronx','Brooklyn','Staten_Island','Manhattan','Queens']):
    for y,year in enumerate(opd_by_borough.year):
        rates=list((DiD.loc[(DiD["year"]==year)&(DiD['borough']==borough),"overdose_deaths_per_100k"].values))
        opd_by_borough.iloc[y,b+1]=round(statistics.mean(rates),2)
        

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




#Opioid deaths per 100k per year per district
opd_by_district=pd.DataFrame()
opd_by_district['year']=DiD.year.unique()


for district in DiD.district.unique():
    opd_by_district[district]=""

for d,district in enumerate(opd_by_district.columns[1:]):
    for y,year in enumerate(opd_by_district.year):
        opd_by_district.iloc[y,d+1]=round(DiD.loc[(DiD["year"]==year)&(DiD['district']==district),"log_overdose_deaths_per_100k"].values[0],2)

opd_by_district

change_15_16=pd.DataFrame()
change_15_16['district']=DiD.district.unique()
change_15_16["delta"]=""

for district in change_15_16.district:
    final=opd_by_district.loc[opd_by_district['year']==2016,district].values[0]-opd_by_district.loc[opd_by_district['year']==2015,district].values[0]
    change_15_16.loc[change_15_16["district"]==district,"delta"]=final



    
#DISTRICT WITH SIMILAR YOY 2015 TO 2016 AS BRONX_7
labels={"Brooklyn_15":'gray',"Queens_8":'gray','Bronx_7':'red','Queens_3':'gray','Manhattan_12':'gray'}
plt.figure(figsize=(8, 5))
IVs=['treatment','post','treatment:post','Bronx','Manhattan',"Brooklyn","Queens",'Staten_Island']
for district in labels.keys():
    plt.plot(opd_by_district['year'][:2],opd_by_district[district][:2],label=district,color=labels[district])
    plt.plot(opd_by_district['year'][2:4],opd_by_district[district][2:4],color=labels[district])
    IVs.append(district)
plt.xlabel("year")
plt.xticks([2015,2016,2017,2018,2019])
plt.axvline(x=2017, color='black', linestyle='--')
plt.ylabel("log_Overdose Deaths per 100k")
plt.title("Districts with similar trends from 2015 to 2016 as Bronx 7")
plt.legend(loc='best')

time_frame=[2015,2016,2018,2019]
result=linear_reg(time_frame,IVs)
print(result)

#DISTRICT WITH SIMILAR YOY 2015 TO 2016 AS MANHATTAN_12

labels={'Bronx_7':'gray','Queens_3':'gray','Manhattan_12':'blue','Queens_7':'gray','Brooklyn_2':'gray',}
plt.figure(figsize=(8, 5))
IVs=['treatment','post','treatment:post','Bronx','Manhattan',"Brooklyn","Queens",'Staten_Island']
for district in labels.keys():
    plt.plot(opd_by_district['year'][:2],opd_by_district[district][:2],label=district,color=labels[district])
    plt.plot(opd_by_district['year'][2:4],opd_by_district[district][2:4],color=labels[district])
    IVs.append(district)
plt.xlabel("year")
plt.xticks([2015,2016,2017,2018,2019])
plt.axvline(x=2017, color='black', linestyle='--')
plt.ylabel("Overdose Deaths per 100k")
plt.title("Districts with similar trends from 2015 to 2016 as Manhattan 12")
plt.legend(loc='best')

time_frame=[2015,2016,2018,2019]
result=linear_reg(time_frame,IVs)
print(result)


labels={'Queens_5':'gray','Manhattan_10':'gray','Staten_Island_1':'green','Brooklyn_3':'gray','Brooklyn_10':'gray'}
plt.figure(figsize=(8, 5))
IVs=['treatment','post','treatment:post','Bronx','Manhattan',"Brooklyn","Queens",'Staten_Island']
for district in labels.keys():
    plt.plot(opd_by_district['year'][:2],opd_by_district[district][:2],label=district,color=labels[district])
    plt.plot(opd_by_district['year'][2:4],opd_by_district[district][2:4],color=labels[district])
    IVs.append(district)
plt.xlabel("year")
plt.xticks([2015,2016,2017,2018,2019])
plt.axvline(x=2017, color='black', linestyle='--')
plt.ylabel("Overdose Deaths per 100k")
plt.title("Districts with similar trends from 2015 to 2016 as Staten Island 1")
plt.legend(loc='best')

time_frame=[2015,2016,2018,2019]
result=linear_reg(time_frame,IVs)
print(result)


#Plotting
color_dict={}
for district in DiD.district.unique():
    if district == "Bronx_7":
        color_dict[district]="Red"
    elif district == "Manhattan_12":
        color_dict[district]="Blue"
    elif district== "Staten_Island_1":
        color_dict[district]="Green"
    else:
        color_dict[district]="Gray"

plt.figure(figsize=(15, 10))
for district in opd_by_district.columns[1:]:
    plt.plot(opd_by_district['year'],opd_by_district[district],label=district if district in treatment_group else None,color=color_dict[district])
plt.legend()





#Create column for each treatment group, label 1 if occurence is said treatment group, 0 if not

for cd in treatment_group:
    DiD[cd]=0
    DiD.loc[DiD['district']==cd,cd]=1

#ADDITIONAL INTERACTION TERMS
DiD["treatment:post:Manh12"]=DiD['treatment']*DiD["post"]*DiD['Manhattan_12']
DiD["treatment:post:Bronx7"]=DiD['treatment']*DiD["post"]*DiD['Bronx_7']
DiD["treatment:post:Stat1"]=DiD['treatment']*DiD["post"]*DiD['Staten_Island_1']

time_frame=[2015,2016,2018,2019]
IVs=['treatment','post','treatment:post',"treatment:post:Manh12","treatment:post:Bronx7","treatment:post:Stat1"]
result=linear_reg(time_frame,IVs)
print(result)


# FIXED EFFECTS WITH GROUPS FOR TREATMENT DISTRICTS
def create_groups(group_name,group):
    DiD[group_name]=0
    for cd in group:
        DiD.loc[DiD['district']==cd,group_name]=1
        
Manhattan_group=['Manhattan_12','Manhattan_11','Manhattan_10','Manhattan_9']
Mannhattan_group_name=["Manh12_11_10_9"]
create_groups(Mannhattan_group_name,Manhattan_group)

Bronx_group=['Bronx_7','Bronx_5','Bronx_8','Bronx_12']
Bronx_group_name=["Bronx7_5_8_12"]
create_groups(Bronx_group_name,Bronx_group)

# Stat_group=['Staten_Island_1','Staten_Island_2','Staten_Island_3']
# Stat_group_name=["Stat1_2_3"]
# create_groups(Stat_group_name,Stat_group)

time_frame=[2015,2016,2018,2019]
IVs=['treatment','treatment:post','post',"Manh12_11_10_9","Bronx7_5_8_12"]
result=linear_reg(time_frame,IVs)
print(result)

# FIXED EFFECTS FOR ALL district
time_frame=[2015,2016,2018,2019]
IVs=['treatment','post','treatment:post']+list(DiD.district.unique())
result=linear_reg(time_frame,IVs)
print(result)


#FIXED EFFECTS FOR BOROUGH
time_frame=[2015,2016,2018,2019]
IVs=['treatment','post','treatment:post']+list(DiD.borough.unique())
result=linear_reg(time_frame,IVs)
print(result)


#Covariates
selected_columns=DiD.columns[[3,71,92,486,323,20,290,298,39,469,516,21,514,66,29,506]]
time_frame=[2015,2016,2018,2019]
linear_reg(time_frame, selected_columns)





