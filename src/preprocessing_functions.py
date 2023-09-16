import numpy as np
import pandas as pd

def create_final_dataset(df_map,df_crop,df_elev,df_cropland):

    df_merged=df_map.merge(df_crop,left_on=["distname","year"],right_on=["distname","year"],how="left")
    df_merged=df_merged.merge(df_elev[["index","elevation"]],left_on=["Index"],right_on=["index"],how="left").drop(columns=["index"])
    df_merged=df_merged.merge(df_cropland[["index","cropland"]],left_on=["Index"],right_on=["index"],how="left").drop(columns=["index"])


    df_pivot=pd.pivot_table(df_merged,columns=["crop_type"],index=["Index","Longitude","Latitude","distname","biomass","year","elevation","cropland"])
    df_pivot.columns = [ c[0]+"-"+str(c[1]) for c in df_pivot.columns.to_flat_index()]
    df_pivot=df_pivot.reset_index()

    df_pivot=df_pivot.fillna(0) #Nan means no prod

    print("final shape:",df_pivot.shape)

    return df_pivot

def add_features(df):
    
    cat_dict=dict( enumerate(df["distname"].astype("category").cat.categories ) )
    df["distname"]=df["distname"].astype("category").cat.codes
    

    print(cat_dict)

    for c in df.distname.unique():
        df["in_district_"+str(c)]=np.where(df["distname"]==c,1,0)


    df["total_crop_prod"]=df[[c for c in df.columns if "production" in c]].sum(axis=1)
    df["count_district"]=df.groupby(["distname"])["total_crop_prod"].transform(lambda x:x.count())
 
    df["index_cotton_prod_share"]=df["production-Cotton(lint)"]/df["count_district"]

    for c in ["production-Cotton(lint)"]:
    
        df["district_prod_"+c+"_sum"]=df.groupby(["distname","year"])[c].transform(lambda x:x.sum()) 

    df["log_biomass"]=np.log10(df["biomass"]+1e-5)

    return df


