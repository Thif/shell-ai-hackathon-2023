# pylint: disable=C0103,R0914,R0913,R0911,E0401
"""
A module for functions related to cost calculations
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import opt_functions as of

sys.path.insert(0,'../src')


CAP_DEPOT=20000
CAP_RAF=100000


def get_cost(depot_index,
             raf_index,
             biomass_mat
             ,dist_mat,
             pellet_mat,
             forecast,adjust,
             ignore_constraints=False):
    """
    Gets the cost for a given setup
    """
    transport_biomass=np.sum(np.multiply(dist_mat[:,depot_index],biomass_mat))
    transport_pellet=np.sum(np.multiply(dist_mat[depot_index,:][:,raf_index],pellet_mat))

    underutil_pellet=np.sum(CAP_DEPOT-np.sum(biomass_mat,axis=0))
    underutil_raf=np.sum(CAP_RAF-np.sum(pellet_mat,axis=0))

    if not ignore_constraints:
        if len(pd.unique(depot_index))!=len(depot_index):
            print("duplicate depot index")
            return None
        if (biomass_mat<0).any():
            print("negative biomass supply")
            return None
        if (pellet_mat<0).any():
            print("negative pellet supply")
            return None
        if any(forecast.flatten()-np.sum(biomass_mat,axis=1)<0):
            print("more biomass than forecast")
            return None
        if (forecast<0).any():
            print("negative forecast")
            return None
        if any(np.sum(biomass_mat,axis=0)>CAP_DEPOT):
            print( "depot capacity" )
            return None
        if any(np.sum(pellet_mat,axis=0)>CAP_RAF):
            print( "refinery capacity" )
            return None
        if len(depot_index)>25:
            print("too many depots")
            return None
        if len(raf_index)>5:
            print("too many depots")
            return None
        if pellet_mat.sum()-0.8*forecast.sum()<0:
            print(" less than 80% is processed")
            return None
        if any(np.abs(np.sum(biomass_mat,axis=0)-np.sum(pellet_mat,axis=1))>1e-3):
            print("difference in biomass and pellet amount")
            return None
    underutil_cost=np.sum(underutil_pellet)+np.sum(underutil_raf)
    transport_cost=np.sum(transport_biomass)*0.001+np.sum(transport_pellet)*0.001

    total_cost=underutil_cost+transport_cost+np.sum(np.abs(adjust))
    return underutil_cost,transport_cost,total_cost

def plot_network(df_biomass,df_forecast,depot_index,raf_index):
    """
    plots network
    """
    df=df_biomass.copy()
    df["biomass"]=df_forecast
    for i in depot_index:
        df.loc[i,"depot"]=1

    for i in raf_index:
        df.loc[i,"raf"]=1

    _,ax=plt.subplots()

    depots=df.loc[df["depot"]==1]
    rafs=df.loc[df["raf"]==1]

    df.plot(kind="scatter",x="Longitude",y="Latitude",c="biomass",ax=ax,cmap="rainbow")

    for row in depots.itertuples():
        circle = plt.Circle((row.Longitude, row.Latitude), 0.1, color='black',fill=False,lw=2)
        ax.add_patch(circle)

    for row in rafs.itertuples():
        circle = plt.Circle((row.Longitude, row.Latitude), 0.1, color='black',fill=False,lw=5)
        ax.add_patch(circle)

def create_submission_file(forecast_2018,
                           forecast_2019,
                           depot_index,
                           raf_index,
                           biomass_mat_2018,
                           biomass_mat_2019,
                           pellet_mat_2018,
                           pellet_mat_2019):
    """
    Exports submission file for a given depot/ref configuration
    """
    df=pd.DataFrame([],columns=["year","data_type","source_index","destination_index","value"])

    #forecast 2018
    df_forecast_2018=pd.DataFrame(forecast_2018,columns=["value"])
    df_forecast_2018["data_type"]="biomass_forecast"
    df_forecast_2018["year"]="2018"
    df_forecast_2018["source_index"]=df_forecast_2018.index

    #forecast 2019
    df_forecast_2019=pd.DataFrame(forecast_2019,columns=["value"])
    df_forecast_2019["data_type"]="biomass_forecast"
    df_forecast_2019["year"]="2019"
    df_forecast_2019["source_index"]=df_forecast_2019.index

    #refinery location
    df_refinery_loc=pd.DataFrame(raf_index,columns=["source_index"])
    df_refinery_loc["data_type"]="refinery_location"
    df_refinery_loc["year"]="20182019"

    #depot location
    df_depot_loc=pd.DataFrame(depot_index,columns=["source_index"])
    df_depot_loc["data_type"]="depot_location"
    df_depot_loc["year"]="20182019"

    columns={"index": "source_index", "variable": "destination_index"}
    #biomass demand_supply 2018
    df_biomass_2018=pd.melt(pd.DataFrame(biomass_mat_2018,columns=depot_index)
                            .reset_index(),id_vars="index").rename(columns=columns)
    df_biomass_2018["data_type"]="biomass_demand_supply"
    df_biomass_2018["year"]="2018"

    #biomass demand_supply 2018
    df_biomass_2019=pd.melt(pd.DataFrame(biomass_mat_2019,columns=depot_index)
                            .reset_index(),id_vars="index").rename(columns=columns)
    df_biomass_2019["data_type"]="biomass_demand_supply"
    df_biomass_2019["year"]="2019"

    #pellet demand_supply 2018
    df_pellet_2018=pd.melt(pd.DataFrame(pellet_mat_2018,index=depot_index,columns=raf_index)
                           .reset_index(),id_vars="index").rename(columns=columns)
    df_pellet_2018["data_type"]="pellet_demand_supply"
    df_pellet_2018["year"]="2018"

    #pellet demand_supply 2019
    df_pellet_2019=pd.melt(pd.DataFrame(pellet_mat_2019,index=depot_index,columns=raf_index)
                           .reset_index(),id_vars="index").rename(columns=columns)
    df_pellet_2019["data_type"]="pellet_demand_supply"
    df_pellet_2019["year"]="2019"

    df=pd.concat([df,df_depot_loc,df_refinery_loc,df_forecast_2018,df_biomass_2018,
                  df_pellet_2018,df_forecast_2019,df_biomass_2019,df_pellet_2019])
    df["source_index"]=df["source_index"].astype('Int64')
    df["destination_index"]=df["destination_index"].astype('Int64')
    df=df[df["value"]!=0]
    return df

def get_cost_from_lp(depot_index,raf_index,forecast,dist_mat):
    """
    Runs linear optimisation and gets the matrices
    """
    model=of.create_biomass_mat_lp(depot_index,raf_index,forecast,dist_mat)

    biomass_mat,pellet_mat,adjust=of.get_matrices_from_lp(model,len(depot_index),len(raf_index))
    costs=get_cost(depot_index,raf_index,biomass_mat,dist_mat,pellet_mat,forecast,adjust)
    return costs

def get_cost_from_lp_with_adjust(depot_index,raf_index,forecast,dist_mat):
    """
    Runs linear optimisation and gets the matrices, takes into account possible forecast adjustment
    """
    model=of.create_biomass_mat_lp_with_adjust(depot_index,raf_index,forecast,dist_mat)

    biomass_mat,pellet_mat,adjust=of.get_matrices_from_lp(model,len(depot_index),len(raf_index))
    forecast_new=forecast.flatten()+adjust.flatten()
    costs=get_cost(depot_index,raf_index,biomass_mat,dist_mat,pellet_mat,forecast_new,adjust)
    return costs
