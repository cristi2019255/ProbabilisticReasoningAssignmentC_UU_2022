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


from utils.plotter import plot_data, plot_data_and_fit_no_pooling_and_mix_pooling
from utils.read import read_data, read_stan_results
import numpy as np
import stan
from utils.stan_models import get_stan_code
from utils.plotter import plot_data_and_fit
from utils.write import write_results
import os

QUESTION = "Q3_B"

def main():
    if QUESTION in ["Q1", "Q2"]:
        simple_flow()
    elif QUESTION in ["Q3_A", "Q3_B", "Q4"]:
        hirearchical_flow()
    else:
        print("Invalid question number")

def simple_flow():
    stan_code = get_stan_code(question=QUESTION)
    
    data = read_data("data/merged_data.csv", cols = ["d18_O_w", "d18_O", "temperature"])
    
    data = {
        "N": len(data),
        "d18_O_w": data["d18_O_w"].to_list(),
        "d18_O_c": data["d18_O"].to_list(),
        "y": data["temperature"].to_list()
    }
    
    
    # visualizing the data 
    x = np.array(data["d18_O_c"]) - np.array(data["d18_O_w"])
    y = data["y"]
    
    # plotting the data
    plot_data(x, y, folder = QUESTION)
    
    # fitting the model
    posterior = stan.build(stan_code, data=data, random_seed=1)
    
    fit = posterior.sample(num_chains=4, num_samples=1000)
    df = fit.to_frame()
    print(df)
    print(df.describe().T)    
       
    # getting the parameters from the posterior   
    write_results(df, file_name = "results.txt", cols = ["a", "b", "sigma"], folder=QUESTION)
    
    plot_data_and_fit(x, y, df, construct_model_function(), folder = QUESTION)
    
def hirearchical_flow():
    stan_code = get_stan_code(question=QUESTION)
    
    data_df = read_data("data/merged_data.csv", cols = ["d18_O_w", "d18_O", "temperature", "species"])
    
    for species in data_df["species"].unique():
        data_df_species = data_df[data_df["species"] == species]
        
        data = get_data_for_species(data_df_species, species)
    
        x = np.array(data_df_species["d18_O"]) - np.array(data_df_species["d18_O_w"])
        y = data_df_species["temperature"]
        
        # plotting the data
        plot_data(x, y, folder = os.path.join(QUESTION, species), title = "Temperature vs. d18_O for " + species)
    
        # fitting the model
        posterior = stan.build(stan_code, data=data, random_seed=1)
    
        fit = posterior.sample(num_chains=4, num_samples=100)
        df = fit.to_frame()
        print(df)
        print(df.describe().T)    
       
        df_no_pooling = stan.build(get_stan_code(question="Q3_A"), data=data, random_seed=1).sample(num_chains=4, num_samples=100).to_frame()
        
        write_results(df, file_name = "results.txt", cols = ["a", "b", "sigma"], folder=os.path.join(QUESTION, "species_" + species))

        if QUESTION == "Q3_B":
            plot_data_and_fit_no_pooling_and_mix_pooling(x, y, df_no_pooling, df, construct_model_function(cols = ["a", "b"]), folder = os.path.join(QUESTION, species), title = "Temperature vs. d18_O for " + species)
        else:
            plot_data_and_fit(x, y, df, construct_model_function(), folder = os.path.join(QUESTION, species), cols = ["a", "b", "sigma"], title = "Temperature vs. d18_O for " + species)
    

def get_data_for_species(data_df_species, species):
    if QUESTION == "Q3_A":
        data = {
            "N": len(data_df_species),
            "d18_O_w": data_df_species["d18_O_w"].to_list(),
            "d18_O_c": data_df_species["d18_O"].to_list(),
            "y": data_df_species["temperature"].to_list()
        }
    elif QUESTION == "Q3_B":
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
    
    return data
    
def construct_model_function(cols = ["a", "b"]):
    def model_function(x, params):
        a, b = params[cols[0]], params[cols[1]]
        return a + b * x
    return model_function


if __name__ == "__main__":
    main()