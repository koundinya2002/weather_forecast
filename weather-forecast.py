#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


#defining the main func
def main():
    #importing the weather dataset
    weather=pd.read_csv(r"C:\Users\DELL\Desktop\ENTHS\Techno Mist\Weather Forecast\3371527.csv", index_col="DATE")
    print(weather)

    #considering only the important values
    core_weather=weather[["PRCP","TMAX","TMIN"]].copy()

    #changing the column names into lowercase for ease of access
    core_weather.columns=["prcp","tmax","tmin"]

    #printing imp_weather
    print(core_weather)

    #finding the null values in the important dataset
    print(core_weather.apply(pd.isnull).sum()/weather.shape[0])
    #finding null values in prcp
    print(core_weather[pd.isnull(core_weather["prcp"])])

    #finding the no of null values in prcp
    print(core_weather["prcp"].value_counts())

    #filling the null values of prcp with 0 as most of the days it's 0
    core_weather["prcp"]=core_weather["prcp"].fillna(0)

    #finding null values in tmax
    print(core_weather[pd.isnull(core_weather["tmax"])]/core_weather.shape[0])

    #filling the null values based on forward and backward values
    core_weather=core_weather.fillna(method="ffill")
    core_weather=core_weather.fillna(method="bfill")

    #printing to check if all the values are filled
    print(core_weather)

    #indexing with years
    core_weather.index=pd.to_datetime(core_weather.index)
    print(core_weather.index.year)

    #checking to trim any data is not properly received
    print(core_weather.apply(lambda x:(x==9999).sum()))

    #plotting a graph to showcase the difference between tmax and tmin
    plt.plot((core_weather[["tmax","tmin"]]))
    plt.show()

    #sorting the datset based on indices
    core_weather.index.year.value_counts().sort_index()

    #plotting prcp graph
    plt.plot((core_weather[["prcp"]]))
    plt.show()

    #grouping the data by prcp
    core_weather.groupby(core_weather.index.year).sum()["prcp"]


    #creating a target value in forecasting the data
    core_weather["target_max"]=core_weather.shift(-1)["tmax"]

    core_weather=core_weather.iloc[:-1,:].copy()

    reg=Ridge(alpha=.1)

    predictors=["prcp","tmin","tmax"]

    #providing the model with training and test data
    train=core_weather.loc[:"2020-12-31"]
    test=core_weather.loc["2021-01-01":]

    reg.fit(train[predictors],train["target_max"])
    
    predictors=reg.predict(test[predictors])


    combination=pd.concat([test["target_max"],pd.Series(predictors,index=test.index)],axis=1)
    combination.columns=["actual","prediction"]

    #printing to see the difference between actual and predicted values
    plt.plot(combination)
    plt.show()

    #which value has the most affect on the rest values
    print(reg.coef_)

    print("Predictions:",predictors,sep='\n')


    print(core_weather.corr()["target_max"])

    print(core_weather)

    print("Maximum Error:",mean_absolute_error(test["target_max"],predictors))
    
    #getting the day average based on the past year's averages
    core_weather["mon_max"]=core_weather["tmax"].rolling(30).mean()

    core_weather["day_max"]=core_weather["mon_max"]/core_weather["tmax"]

    
    
    print(core_weather)
    
    
    


main()
    
