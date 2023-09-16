
from matplotlib import pyplot
import geopandas as gpd
from shapely import geometry
import pandas as pd
import numpy as np

df_biomass=pd.read_csv("../dataset/Biomass_History.csv")

def create_map(file_path):
 
    #read shapefile as df
    shapefile = gpd.read_file(file_path)

    shapefile=shapefile[shapefile["statename"]=="Gujarat"]

    shapefile=shapefile.replace({'distname': {"Batod":"Botad",'Aravali':'Aravalli','Sabar Kantha':'Sabar kantha',
                                            'Gir Somnath':'Gir somnath','Devbhumi Dwarka':'Devbhumi dwarka',
                                            'Banas Kantha':'Banas kantha','Panch Mahals':'Panch mahals',
                                            'Chhota Udaipur':'Chhotaudepur',"The Dangs":"Dang"}})
    

    #assign each index to a district
    
    df=df_biomass.copy()

    for row1 in df.itertuples():
        offset=0.03
        longitude=row1.Longitude
        latitude=row1.Latitude
        point_b_left = geometry.Point(longitude-offset, latitude-offset)
        point_t_right = geometry.Point(longitude+offset, latitude+offset)
        point_t_left = geometry.Point(longitude-offset, latitude+offset)
        point_b_right = geometry.Point(longitude+offset, latitude-offset)

        for row2 in shapefile.itertuples():
            polygon=row2.geometry
            if polygon.contains(point_b_left) and polygon.contains(point_t_right) and polygon.contains(point_t_left) and polygon.contains(point_b_right):
                df.loc[row1.Index,"distname"]=row2.distname
                break

    print(df.distname.isna().sum()," not assigned to a district")

    return df

"""
def post_process_ditricts(df):
    for t in range(5):
        for i in range(5):
            for row in df.itertuples():
                if pd.isnull(row.distname):
                    
                    
                    longitude=row.Longitude
                    latitude=row.Latitude
                    input=[longitude,latitude]
                    

                    
                    df_closest=df.iloc[(df[['Longitude',"Latitude"]]-input).abs().mean(axis=1).sort_values().index[1+t]]

                    if not pd.isnull(df_closest.distname):
                        df.loc[row.Index,"distname"]=df_closest.distname

    print(df.distname.isna().sum()," not assigned to a district")
    
    return df

"""
def post_process_ditricts(df):

    df_corr=df.groupby(["distname","year"])["biomass"].sum().reset_index()
    df_corr=pd.pivot(df_corr,index=["year"],columns=["distname"])
    df_corr.columns = [ str(c[1]) for c in df_corr.columns.to_flat_index()]


    for g,group in df.groupby(["Index"]):
        df_corr["index_val"]=group["biomass"].values
        corr=df_corr.corr()["index_val"].sort_values()
        
        df.loc[df["Index"]==g[0],"distname"]=corr.index[-2]

    print(df.distname.isna().sum()," not assigned to a district")
    
    df.plot(kind="scatter",x="Longitude",y="Latitude",c=df.distname.astype("category").cat.codes,cmap="rainbow")

    return df


def add_biomass_prod(df_map,df_biomass):
    df_sum=pd.DataFrame()
    drop_cols=["2010","2011","2012","2013","2014","2015","2016","2017"]
    df=df_map.copy()
    df=df.drop(columns=drop_cols)
    
    for y in range(2010,2020):
        df["year"]=y
        try:
            df["biomass"]=df_biomass[str(y)]
        except:
            df["biomass"]=0
        df_sum=pd.concat([df_sum,df])
        
    print("final shape:",df.shape)
    return df_sum

def get_biomass_df(file_path):
    df_biomass=pd.read_csv(file_path)
    df_biomass["flag"]=0
    df_biomass.loc[df_biomass["2013"]==df_biomass["2014"],"flag"]=1

    for row in df_biomass.itertuples():
        if row.flag==1:
            df_biomass.loc[row.Index,"2010"]=np.nan
            df_biomass.loc[row.Index,"2011"]=row._4
            df_biomass.loc[row.Index,"2012"]=row._5
            df_biomass.loc[row.Index,"2013"]=np.nan

    df_biomass=df_biomass.drop(columns=["flag"])        
    df_biomass=df_biomass.melt(id_vars=["Index","Longitude","Latitude"],var_name="year",value_name="biomass")
    df_biomass["biomass"]=df_biomass.groupby(["Index"])["biomass"].transform(lambda x:x.interpolate().fillna(method="bfill"))
    df_biomass=pd.pivot(df_biomass,index=["Index","Longitude","Latitude"],columns=["year"])

    df_biomass.columns=[c[1] for c in df_biomass.columns]
    

    return df_biomass.reset_index()