from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler,StandardScaler,MinMaxScaler,RobustScaler,PowerTransformer
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor,RandomForestClassifier,StackingRegressor,VotingRegressor
from sklearn.metrics import mean_absolute_error,auc,roc_curve
from sklearn.linear_model import LinearRegression,ElasticNet,Lasso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


def train_model_log(df):
    

    params = {
        'random_state': 42,
            "n_estimators":100, 
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Step 1: Scale the data
        ('regressor', ExtraTreesRegressor(**params))  # Step 2: Train ExtraTreesRegressor
    ])

    results=pd.DataFrame()
    tot_mae=[]
    for y in [2010,2011,2012,2013,2014,2015,2016,2017]:

        df_train=df.loc[(~df.year.isin([2018,2019,y]))&(df["biomass"]),:]
        df_test=df.loc[df.year==y,:]

        X_train=df_train.drop(columns=["log_biomass","biomass","year"])
        y_train=df_train[["log_biomass"]].values.ravel()

        X_test=df_test.drop(columns=["log_biomass","biomass","year"])
        y_test=df_test[["log_biomass"]].values.ravel()

        pipeline.fit(X_train,y_train)

        mae_test=np.abs(10**pipeline.predict(X_test)-10**y_test).mean()
        mae_train=np.abs(10**pipeline.predict(X_train)-10**y_train).mean()
        tot_mae+=[(y,mae_train,mae_test)]
        print(y,mae_train,mae_test)


        df_test.loc[:,"pred"]=10**pipeline.predict(X_test)
        df_test.loc[:,"mae"]=10**y_test-10**pipeline.predict(X_test)
        results=pd.concat([results,df_test])
        
        
        
        
        
        
        
    sum_df=pd.DataFrame(tot_mae,columns=["year","mae_train","mae_test"]).T
    sum_df["mean"]=sum_df.mean(axis=1)

    print(sum_df)

    return pipeline,results


def generate_predictions(model,df,filename):
    
    df_train=df.loc[~df.year.isin([2018,2019]),:]

    X_train=df_train.drop(columns=["log_biomass","year","biomass"])
    y_train=df_train[["log_biomass"]].values.ravel()


    model.fit(X_train,y_train)

    X_2018=df[df.year==2018].drop(columns=["log_biomass","year","biomass"])
    X_2019=df[df.year==2019].drop(columns=["log_biomass","year","biomass"])

    y_2018=10**model.predict(X_2018)
    y_2019=10**model.predict(X_2019)

    fig,ax=plt.subplots(1,2,figsize=(20,10))
    X_2018["2018"]=y_2018

    X_2018.plot(kind="scatter",x="Longitude",y="Latitude",c="2018",cmap="rainbow",ax=ax[0],title="2018 forecast: "+str(int(np.sum(y_2018))))

    X_2019["2019"]=y_2019
    X_2019.plot(kind="scatter",x="Longitude",y="Latitude",c="2019",cmap="rainbow",ax=ax[1],title="2019 forecast: "+str(int(np.sum(y_2019))))

    pd.concat([X_2018["2018"].reset_index(),X_2019["2019"].reset_index()],axis=1).drop(columns="index").to_csv(f"../forecast/{filename}")


    return y_2018,y_2019


def add_theo_biomass(df,val_year=None):

    if "theo_biomass" in df.columns:
        df=df.drop(columns=["theo_biomass"])

    params = {
        'random_state': 42,
            "criterion": "log_loss",
            "class_weight":"balanced",
            "n_estimators":100, 
    }

    pipeline = LinearRegression()

    prediction_columns = ["production-Cotton(lint)"]


    if val_year is None:
        tot_mae=[]
        for y in [2010,2011,2012,2013,2014,2015,2016,2017]:
            #if y!=2012:continue
            
            df_train=df.loc[~df.year.isin([2018,2019,y]),:]
            df_test=df.loc[df.year==y,:]

            X_train=df_train[prediction_columns]
            y_train=df_train["biomass"].values.ravel()

            X_test=df_test[prediction_columns]
            y_test=df_test["biomass"].values.ravel()

            pipeline.fit(X_train,y_train)

            mae_test=np.abs(pipeline.predict(X_test)-y_test).mean()
            mae_train=np.abs(pipeline.predict(X_train)-y_train).mean()
            tot_mae+=[(y,mae_train,mae_test)]
            print(y,mae_train,mae_test)
            


        
        sum_df=pd.DataFrame(tot_mae,columns=["year","auc_test"]).T
        sum_df["mean"]=sum_df.mean(axis=1)

        print(sum_df)

        df_train=df.loc[~df.year.isin([2018,2019]),:]

        X_train=df_train[prediction_columns]
        y_train=df_train["biomass"].values.ravel()


        pipeline.fit(X_train,y_train)

        df["theo_biomass"]=np.where(df.year.isin([2018,2019]),pipeline.predict(df[prediction_columns]),df["biomass"])


    else:
        df_train=df.loc[~df.year.isin([2018,2019,val_year]),:]

        X_train=df_train[prediction_columns]
        y_train=df_train["biomass"].values.ravel()


        pipeline.fit(X_train,y_train)
 
        df["theo_biomass"]=np.where(df.year.isin([2018,2019,val_year]),pipeline.predict(df[prediction_columns]),df["biomass"])

    return df
