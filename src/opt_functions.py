from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable,LpMinimize,LpConstraint,lpDot,LpAffineExpression
import pulp as pl
from sklearn.cluster import KMeans
import numpy as np
import random
import pandas as pd
import cost_functions as cf


CAP_DEPOT=20000
CAP_RAF=100000
EPS=1e-6
df_biomass=pd.read_csv("../dataset//Biomass_History.csv")

def run_substraction(dist_mat,forecast_df,index_thresh=200,raf_thresh=500,min_depot=15,max_depot=60,N_raf=3):

  top_index=forecast_df[forecast_df["biomass"]>index_thresh].index
  top_raf=[985, 1358, 1861] #forecast_df[forecast_df["biomass"]>raf_thresh].index
  cost_dict=[]
 
  current_depot_index=random.sample(list(top_index), max_depot)
  current_raf_index=random.sample(list(top_raf), N_raf)


  for N_depot in reversed(range(min_depot,max_depot+1)):

    #raf_index=of.get_raf_index(df_biomass,current_depot_index,N_raf)

    model=create_biomass_mat_lp_with_adjust(current_depot_index,current_raf_index,forecast_df.to_numpy(),dist_mat)

    biomass_mat,pellet_mat,adjust=get_matrices_from_lp(model,N_depot,N_raf)
    forecast_new=forecast_df.to_numpy().flatten()+adjust.flatten()
    costs=cf.get_cost(current_depot_index,current_raf_index,biomass_mat,dist_mat,pellet_mat,forecast_new,adjust,ignore_constraints=True)

    if costs[-1] is not None:
      cost_dict+=[(current_depot_index.copy(),current_raf_index,costs[-1])]

    min_biomass=np.argmin(biomass_mat.sum(axis=0))
    del current_depot_index[min_biomass]
  

  df_opt=pd.DataFrame(cost_dict,columns=["depot","raf","cost"])
    

  return df_opt

def run_optimization(dist_mat,forecast_df,index_thresh=200,raf_thresh=500,min_depot=15,max_depot=16,N_raf=3,max_iter=1,dlong=0.05,dlat=0.05,n=1,ignore=32000):
  df_opt=run_substraction(dist_mat,forecast_df,index_thresh=index_thresh,raf_thresh=raf_thresh,min_depot=min_depot,max_depot=max_depot,N_raf=N_raf)
  best_run=df_opt.iloc[df_opt["cost"].idxmin()]

  print("best run has cost of ",best_run["cost"])
  if best_run["cost"]>ignore:
     return best_run["raf"],best_run["depot"]
  else:
    best_raf_index=best_run["raf"]
    best_depot_index=best_run["depot"]

    final_raf_index,final_depot_index=run_greedy_search(best_depot_index,best_raf_index,forecast_df.to_numpy(),dist_mat,max_iter=max_iter,dlong=dlong,dlat=dlat,n=n)

    return final_raf_index,final_depot_index




def find_new_index(index,dlat,dlong,n):
    v=df_biomass.loc[index,["Latitude","Longitude"]].values+np.array([dlat,dlong])

    index_list=np.abs(df_biomass[["Latitude","Longitude"]]-v).sort_values(["Latitude","Longitude"]).head(n+1).index.values.tolist()
    index_list=[c for c in index_list if c!=index]
    return index_list


def run_greedy_search(depot_index,raf_index,forecast,dist_mat,max_iter,dlong,dlat,n):
      
      init_cost=cf.get_cost_from_lp_with_adjust(depot_index,raf_index,forecast,dist_mat)[-1]
      best_cost=init_cost

      init_dlong=dlong
      init_dlat=dlat


      for i in range(max_iter):
            found=False

            for i in range(len(raf_index)):
                  old_index=raf_index[i]
                  for new in find_new_index(raf_index[i],dlong,dlat,n):
                        
                        raf_index[i]=new
                        new_cost=cf.get_cost_from_lp_with_adjust(depot_index,raf_index,forecast,dist_mat)[-1]

                        if new_cost<best_cost:
                              print("new best cost found",new_cost)
                              print("new raf index is", raf_index[i])
                              best_cost=new_cost
                              dlong=init_dlong
                              dlat=init_dlat
                              print(raf_index)
                              break

                        
                        raf_index[i]=old_index
                        
            for i in range(len(depot_index)):
                  old_index=depot_index[i]
                  print("index",i)
                  for new in find_new_index(depot_index[i],dlong,dlat,n):
                        
                        depot_index[i]=new
                        try:  
                              new_cost=cf.get_cost_from_lp_with_adjust(depot_index,raf_index,forecast,dist_mat)[-1]
                        except:
                              new_cost=1e9

                        if new_cost<best_cost:
                              found=True
                              print("new best cost found",new_cost)
                              print("new depot index is", depot_index[i])
                              best_cost=new_cost
                              dlong=init_dlong
                              dlat=init_dlat
                              print(depot_index)
                              break

                        
                        depot_index[i]=old_index
                  
            if found==False:
                  print("no improvement for all points, extending reach")
                  dlong+=0.05
                  dlat+=0.05

      return raf_index,depot_index



def create_biomass_mat_lp_with_adjust(depot_index,raf_index,forecast,dist_mat):

  forecast=np.around(forecast,4)
  dist_mat_biomass=dist_mat[:,depot_index]
  dist_mat_pellet=dist_mat[depot_index,:][:,raf_index]
  model = LpProblem(name="biomass-allocation", sense=LpMinimize)
  I = range(2418)
  J = range(len(depot_index))
  K= range(len(raf_index))


  # Define the decision variables
  x = LpVariable.dicts("biomass matrix",((i,j)for i in I for j in J),cat="Continuous",lowBound=0)
  y = LpVariable.dicts("pellet matrix",((j,k)for j in J for k in K),cat="Continuous",lowBound=0)
  d = [LpVariable(f"forecast_adjust_var_{i}") for i in I]
  d_abs = [LpVariable(f"forecast_abs_adjust_var_{i}",lowBound=0) for i in I]

  # Add constraints
  model += (lpSum(y[j,k] for j in J for k in K) >= lpSum(forecast[i]+d[i] for i in I)*0.8,"min biomass prod")

  for k in K:
    model += (lpSum(y[j,k] for j in J)<=CAP_RAF-1e-3,"max pellet prod "+str(k))

  for j in J:
    model += (lpSum(x[i,j] for i in I)<=CAP_DEPOT-1e-3,"max depot prod "+str(j))

  for i in I:
    model += ( lpSum(x[i,j] for j in J) <= forecast[i]+ d[i]-1e-4,"max biomass extracted "+str(i))

  for j in J:
    model += ((lpSum(y[j,k] for k in K) - lpSum(x[i,j] for i in I)) <= 1e-4,"max biomass extracted pellet 1 "+str(j))
    model += ((lpSum(x[i,j] for i in I)-lpSum(y[j,k] for k in K)) <= 1e-4,"max biomass extracted pellet 2 "+str(j))

  for i in I:
    model += (d_abs[i]>=d[i],"abs constraint plus_"+str(i))
    model += (d_abs[i]>=-d[i],"abs constraint minus_"+str(i))


  # Set objective

  model += LpAffineExpression([(x[i,j],dist_mat_biomass[i,j]) for i in I for j in J])*0.001+LpAffineExpression([(y[j,k],dist_mat_pellet[j,k]) for j in J for k in K])*0.001+lpSum(CAP_DEPOT-lpSum(x[i,j] for i in I) for j in J)+lpSum(CAP_RAF-lpSum(y[j,k] for j in J) for k in K)+LpAffineExpression([(d_abs[i],1) for i in I])


  # Solve the optimization problem
  status = model.solve(pl.PULP_CBC_CMD(gapRel=0.01))

  print(f"status: {model.status}, {LpStatus[model.status]}")
  print(f"objective: {model.objective.value()}")

  #for var in model.variables():
  #  print(f"{var.name}: {var.value()}")

  #for name, constraint in model.constraints.items():
  #  if "min biomass prod" in name:
  #    print(f"{name}: {constraint.value()}")

  return model

def create_biomass_mat_lp(depot_index,raf_index,forecast,dist_mat):

  forecast=np.around(forecast,4)
  dist_mat_biomass=dist_mat[:,depot_index]
  dist_mat_pellet=dist_mat[depot_index,:][:,raf_index]
  model = LpProblem(name="biomass-allocation", sense=LpMinimize)
  I = range(2418)
  J = range(len(depot_index))
  K= range(len(raf_index))


  # Define the decision variables
  x = LpVariable.dicts("biomass matrix",((i,j)for i in I for j in J),cat="Continuous",lowBound=0)
  y = LpVariable.dicts("pellet matrix",((j,k)for j in J for k in K),cat="Continuous",lowBound=0)


  # Add constraints
  model += (lpSum(y[j,k] for j in J for k in K) >= lpSum(forecast)*0.8,"min biomass prod")

  for k in K:
    model += (lpSum(y[j,k] for j in J)<=CAP_RAF-1e-3,"max pellet prod "+str(k))

  for j in J:
    model += (lpSum(x[i,j] for i in I)<=CAP_DEPOT-1e-3,"max depot prod "+str(j))

  for i in I:
    model += ( lpSum(x[i,j] for j in J) <= forecast[i]-1e-4,"max biomass extracted "+str(i))

  for j in J:
    model += ((lpSum(y[j,k] for k in K) - lpSum(x[i,j] for i in I)) <= 1e-4,"max biomass extracted pellet 1 "+str(j))
    model += ((lpSum(x[i,j] for i in I)-lpSum(y[j,k] for k in K)) <= 1e-4,"max biomass extracted pellet 2 "+str(j))

  # Set objective

  model += LpAffineExpression([(x[i,j],dist_mat_biomass[i,j]) for i in I for j in J])*0.001+LpAffineExpression([(y[j,k],dist_mat_pellet[j,k]) for j in J for k in K])*0.001+lpSum(CAP_DEPOT-lpSum(x[i,j] for i in I) for j in J)+lpSum(CAP_RAF-lpSum(y[j,k] for j in J) for k in K)


  # Solve the optimization problem
  status = model.solve(pl.PULP_CBC_CMD(gapRel=0.01))

  print(f"status: {model.status}, {LpStatus[model.status]}")
  print(f"objective: {model.objective.value()}")

  #for var in model.variables():
  #  print(f"{var.name}: {var.value()}")

  #for name, constraint in model.constraints.items():
  #  if "min biomass prod" in name:
  #    print(f"{name}: {constraint.value()}")

  return model

def get_matrices_from_lp(model,n_depot,n_raf):
  biomass_mat=np.zeros((2418,n_depot))
  pellet_mat=np.zeros((n_depot,n_raf))
  adjust=np.zeros(2418)
 

  eps=1e-5

  for i,var in enumerate(model.variables()):
      if "biomass" in var.name:
        index=var.name[16:-1].split(",_")
        biomass_mat[int(index[0]),int(index[1])]=var.value()
      if "pellet" in var.name:
        index=var.name[15:-1].split(",_")
        pellet_mat[int(index[0]),int(index[1])]=var.value()
      if "forecast_adjust" in var.name:
        index=var.name.split("_")[-1]
        adjust[int(index)]=var.value()


   #adjust to avoid constraint violations
  

  return biomass_mat,pellet_mat,adjust


def get_raf_index(df_biomass,current_depot_index,n_raf):
  m=KMeans(n_clusters=n_raf)
  depot_coords=df_biomass.loc[current_depot_index,["Latitude","Longitude"]]
  m.fit(depot_coords)

  raf_index=[]
  for c in m.cluster_centers_:
    closest_index = depot_coords.iloc[(depot_coords[["Latitude","Longitude"]]-c).mean().abs().argsort()[:1]].index.values[0]
    raf_index+=[closest_index]

  return raf_index

