# Copyright 2022 Cristian Grosu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from model_function import construct_model_function
from utils.plotter import plot_data, plot_data_and_fit_no_pooling_and_mix_pooling, plot_predictions
from utils.read import read_data, read_stan_results
import numpy as np
import stan
from utils.stan_models import get_stan_code
from utils.plotter import plot_data_and_fit
from utils.write import write_results
import os
import arviz as av

def hierarchical_flow( question = "Q1"):
    stan_code = get_stan_code( question= question)
    
    # read data
    data_df = read_data("data/merged_data.csv", cols = ["d18_O_w", "d18_O", "temperature", "species"])
    
    # for each species we fit a model
    for species in data_df["species"].unique():
        data_df_species = data_df[data_df["species"] == species]
        
        data = get_data_for_species(data_df_species, species, question = question)
    
        x = np.array(data_df_species["d18_O"]) - np.array(data_df_species["d18_O_w"])
        y = data_df_species["temperature"]
        
        # plotting the data
        plot_data(x, y, folder = os.path.join( question, species), title = "Temperature vs. d18_O for " + species)
    
        # fitting the model
        posterior = stan.build(stan_code, data=data, random_seed=1)
    
        fit = posterior.sample(num_chains=4, num_samples=100)
        df = fit.to_frame()
        print(av.summary(fit))        
        print(df.describe().T)    
       
        write_results(df, file_name = "results.txt", cols = ["a", "b", "sigma"], folder=os.path.join( question, "species_" + species))

        if  question == "Q3_B":
            df_no_pooling = stan.build(get_stan_code(question="Q3_A"), data=data, random_seed=1).sample(num_chains=4, num_samples=100).to_frame()
            plot_data_and_fit_no_pooling_and_mix_pooling(x, y, df_no_pooling, df, construct_model_function(cols = ["a", "b"]), folder = os.path.join( question, species), title = "Temperature vs. d18_O for " + species)
        elif question == "Q3_A":
            plot_data_and_fit(x, y, df, construct_model_function(), folder = os.path.join( question, species), cols = ["a", "b", "sigma"], title = "Temperature vs. d18_O for " + species)
    
def hierarchical_flow_Q4(question = "Q4_A"):
    stan_code = get_stan_code( question= question)
    
    # read data
    data_df = read_data("data/merged_data.csv", cols = ["d18_O_w", "d18_O", "temperature", "species", "d18_O_w_sd", "d18_O_sd"])
    
    # for each species we fit a model
    for species in data_df["species"].unique():
        data_df_species = data_df[data_df["species"] == species]
        
        data = get_data_for_species(data_df_species, species, question = question)
    
        x = np.array(data_df_species["d18_O"]) - np.array(data_df_species["d18_O_w"])
        y = data_df_species["temperature"]
        
        # plotting the data
        plot_data(x, y, folder = os.path.join( question, species), title = "Temperature vs. d18_O for " + species)
    
        # fitting the model
        posterior = stan.build(stan_code, data=data, random_seed=1)
        
        # predicting the temperature from the posterior
        print("Predicting temperature from posterior")
        fit = posterior.sample(num_samples=1000, num_chains=4)
        df = fit.to_frame()
        cols = [f"y_new.{(i+1)}" for i in range(len(x))]
        df = df[cols].describe().T
        print(av.summary(fit))
        
        df_simple = None
        if question == "Q4_B":
            df_simple = read_stan_results(os.path.join("results", "Q4_A", "species_" + species, "results.txt"))

        plot_predictions(x, y, df, another_pred = df_simple, folder = os.path.join(question, species), title = "Temperature vs. d18_O for " + species, file_name = "predictions.png")
        
        write_results(df, file_name = "results.txt", cols = ["mean", "std"], folder=os.path.join( question, "species_" + species), described=True)
        
def get_data_for_species(data_df_species, species, question = "Q1"):
    if question == "Q3_A":
        data = {
            "N": len(data_df_species),
            "d18_O_w": data_df_species["d18_O_w"].to_list(),
            "d18_O_c": data_df_species["d18_O"].to_list(),
            "y": data_df_species["temperature"].to_list()
        }
    elif question in ["Q3_B", "Q4_A", "Q4_B"]:
        df_global = read_stan_results(os.path.join("results", "Q2", "results.txt"))
        df_species = read_stan_results(os.path.join("results", "Q3_A", "species_" + species, "results.txt"))
        
        data = {
            "N": len(data_df_species),
            "d18_O_w": data_df_species["d18_O_w"].to_list(),
            "d18_O_c": data_df_species["d18_O"].to_list(),
            "y": data_df_species["temperature"].to_list(),
            "a_m": df_global["a"][1],
            "b_m": df_global["b"][1],
            "sigma_a": df_species["a"][2],
            "sigma_b": df_species["b"][2]
        }
        
        if question in ["Q4_A", "Q4_B"]:
            data["d18_O_w_new"] = data["d18_O_w"]
            data["d18_O_c_new"] = data["d18_O_c"]
            if question == "Q4_B":
                data["d18_O_w_std"] = data_df_species["d18_O_w_sd"].to_list()[0]
                data["d18_O_c_std"] = data_df_species["d18_O_sd"].to_list()[0]
        data["K"] = data["N"]        
    return data
    