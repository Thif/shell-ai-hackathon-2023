import pandas as pd
import os

keep_crop=["Sugarcane",
       'Bajra', 'Castor seed',
       'Cotton(lint)', 'Arhar/Tur',
        'Gram', 'Groundnut',
        'Jowar', 'Maize',
       'Moong(Green Gram)', 
         'Rapeseed &Mustard',
       'Rice', 'Sesamum',
       'Soyabean',  'Tobacco',
       'Wheat']

sum_districts=[["Aravalli",'Sabar kantha'],
    ["Botad",'Ahmadabad','Bhavnagar'],
    ['Chhotaudepur','Vadodara'],
    ['Kheda','Mahisagar','Panch mahals'],
    ['Morbi','Rajkot','Surendranagar','Jamnagar',"Devbhumi dwarka"],
    ['Gir somnath','Junagadh'] ]

def process_df(df):
    df_s=df.copy()

    crop_name=df_s.columns[0][0][6:]


    df_s.columns=[c[1]+"_"+c[2] for c in df_s.columns ]
    df_s=df_s[df_s["State_State"]=="Gujarat"]

    df_s=df_s.melt(id_vars=['S.No._S.No.', 'State_State', 'District_District'])
    df_s["year"]=df_s["variable"].str[:4].astype("int")
    df_s["crop"]=crop_name
    df_s=df_s.drop(columns=['S.No._S.No.', 'State_State', 'variable'])
    df_s.columns=["distname","production","year","crop_type"]
    return df_s

def generate_crop_df(root_folder):

    df_all=pd.DataFrame()

    for f in os.listdir(root_folder):
        df_raw=pd.read_html(root_folder+f)
        df=process_df(df_raw[0])

        df_all=pd.concat([df_all,df])

    df_pivot=pd.pivot(df_all,columns=["crop_type","distname"],index=["year"])
    df_pivot.columns = [ str(c[1])+"-"+str(c[2]) for c in df_pivot.columns.to_flat_index()]
    
    
    return df_pivot

def filter_crop(df):
    
    return df[[c for c in df.columns if c.split("-")[0] in keep_crop]]


"""
def impute_dataset(df):
    return df.fillna(0)

"""
def impute_dataset(df):

    district_to_impute = [item for sublist in sum_districts for item in sublist]
    df_impute=df[[c for c in df.columns if c.split("-")[1] in district_to_impute]].copy()

    for crop in keep_crop:
        for s in sum_districts:
            relevant_cols=[crop+"-"+c for c in s ]

            for col in relevant_cols:
                if col not in df_impute.columns:
                    df_impute.loc[:,col]=0
            df_impute.loc[:,"sum_"+s[0]+"-"+s[1]+"-"+crop]=df_impute[relevant_cols].sum(axis=1)

    X_train=df_impute[(df_impute.index>2013)]
    X_pred=df_impute[df_impute.index<2014]

    for crop in keep_crop:
        for s in sum_districts:
            for district in s:
                share=(X_train[crop+"-"+district]/X_train["sum_"+s[0]+"-"+s[1]+"-"+crop]).mean()
                X_pred.loc[:,crop+"-"+district]=share*X_pred["sum_"+s[0]+"-"+s[1]+"-"+crop]

    df_reconstructed=pd.concat([X_train,X_pred]).sort_index()

    for c in df_reconstructed.columns:
        if "sum" in c:continue
        df.loc[:,c]=df_reconstructed.loc[:,c]

    return df

    

def unpivot(df_crop):
    df=df_crop.reset_index().melt(id_vars=["year"],value_name="production")
    df["distname"]=df["variable"].str.split("-").str[1]
    df["crop_type"]=df["variable"].str.split("-").str[0]
    df=df.drop(columns=["variable"])

    print("final shape:",df.shape)

    return df