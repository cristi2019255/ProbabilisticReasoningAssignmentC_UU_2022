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

def hierarchical_flow_Q3_A():
    question = "Q3_A"
    stan_code = get_stan_code( question= question)
    
    # read data
    data_df = read_data("data/merged_data.csv", cols = ["d18_O_w", "d18_O", "temperature", "species"])
    
    # for each species we fit a model
    for species in data_df["species"].unique():
        data_df_species = data_df[data_df["species"] == species]
        
        data = get_data_for_species(data_df_species)
    
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

        plot_data_and_fit(x, y, df, construct_model_function(), folder = os.path.join( question, species), cols = ["a", "b", "sigma"], title = "Temperature vs. d18_O for " + species)
   
def hierarchical_flow_Q3_B():
    question = "Q3_B"        
    stan_code = get_stan_code( question= question)
    
    # read data
    data_df = read_data("data/merged_data.csv", cols = ["d18_O_w", "d18_O", "temperature", "species"])
    
    # for each species we fit a model
    species = data_df["species"].unique()
   
   
   
    data = {
        "J": len(species),
        "N": len(data_df),
        "group": [i+1 for i in range(len(species)) for j in range(len(data_df[data_df["species"] == species[i]]))],
        "d18_O_w": np.array(data_df["d18_O_w"]),
        "d18_O_c": np.array(data_df["d18_O"]),
        "T": np.array(data_df["temperature"]),
    }
    
    # fitting the model
    posterior = stan.build(stan_code, data=data, random_seed=1)
    fit = posterior.sample()
    model_df = fit.to_frame()
    print(av.summary(fit))        
    print(model_df.describe().T)    
     
    for j in range(len(species)):
        data_df_specie = data_df[data_df["species"] == species[j]]
        
        x = np.array(data_df_specie["d18_O"]) - np.array(data_df_specie["d18_O_w"])
        y = data_df_specie["temperature"]
        # plotting the data
        plot_data(x, y, folder = os.path.join( question, species), title = "Temperature vs. d18_O for " + species)
    
        # get the data about the specific specie from df 
        cols = ["A", "B", "sigma_a", "sigma_b", "sigma"]
        df_mix_pooling = model_df[cols]

        write_results(df_mix_pooling, file_name = "results.txt", cols = cols, folder=os.path.join( question, "species_" + species))
    
        data = get_data_for_species(data_df_specie)       
        df_no_pooling = stan.build(get_stan_code(question="Q3_A"), data=data, random_seed=1).sample(num_chains=4, num_samples=100).to_frame()
        plot_data_and_fit_no_pooling_and_mix_pooling(x, y, df_no_pooling, df_mix_pooling, construct_model_function(cols = cols), folder = os.path.join( question, species), title = "Temperature vs. d18_O for " + species)
        
def hierarchical_flow_Q4_A():
    question = "Q4_A"
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

def hierarchical_flow_Q4_B():
    question = "Q4_B"
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

        
def get_data_for_species(data_df_species):
    data = {
            "N": len(data_df_species),
            "d18_O_w": data_df_species["d18_O_w"].to_list(),
            "d18_O_c": data_df_species["d18_O"].to_list(),
            "y": data_df_species["temperature"].to_list()
    }
    return data