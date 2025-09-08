# Updated: 05/2024
# Jordan Holdorf
# Run to create all Markov Transition Matrices

import numpy as np
import os
import sys

# Ensure required input/output folders exist
required_dirs = [
    "../Data",
    f"../Data/SDP_Outputs"
]

for d in required_dirs:
    os.makedirs(d, exist_ok=True)

# Get the absolute path to the "Scripts" directory
current_dir = os.path.dirname(os.path.abspath(__file__))  #
scripts_dir = os.path.dirname(current_dir)  # Moves up to "Scripts"
sys.path.append(scripts_dir)  # Add "Scripts" to sys.path

from Analytic_Solution import analytic_Markov_Matrix

from SDP_Model_Run import run_SDP_model

from Comparison_of_Simulation_Avs import comparison_simulation

from SDP_Comparison import comparison_of_SDP_data

from Simulation_using_investment_plan import Simulation_Model_Investment as SMI

# Run Transition data model:

analytic_markov = True

# Run SDP Model:
new_sdp_data = True

# Run Simulation of using investment plans
new_simulations = True


file_start = "../Data"
file_start_SDP = f"{file_start}/SDP_Outputs"

# Variables:
max_time = 25  # Max time for investment

# Create Array of climate return times
min_between_climate_events = 2
max_between_climate_events = 100
climate_step = 2
climate_value = np.arange(start=min_between_climate_events, stop=(max_between_climate_events+1), step=climate_step)

# Input Variables
rate = 6.32
price_of_carbon = 66

number_states = (3, 5, 7, 9, 15)

area_per_state = [200 / (num_stat-1) for num_stat in number_states]

# Determine Investment cost
inital_cost = 920.50
cost_per_year = 118.20
cost_per_hectare = inital_cost + cost_per_year*max_time
cost_full_conservation = cost_per_hectare* 200
investment_cost = [area * cost_per_hectare for area in area_per_state]
investment_cost = [round(value, 0) for value in investment_cost]


print("\nCost of Investing per Hectare is: $", cost_per_hectare,
      f"\nWhich includes the establishment fee of ${inital_cost} + mantainance fee ${cost_per_year} / per year. \n")
print(f"The total investment for to restore all 200 hectares is ${cost_full_conservation}. \n"
      f"The Cost of investing in the other different number of states is:\n", investment_cost, "\n")

lost_carbon_per_hectare = 0.5
lost_carbon = [area * lost_carbon_per_hectare for area in area_per_state]
lost_carbon = [round(value, 0) for value in lost_carbon]

print(f"The Amount of carbon lost to restore the different number of states is:\n{lost_carbon}")


# Create Discount Array
min_discount = 3
max_discount = 15
discounts_1 = [int(i) + 0.5 for i in range(min_discount, max_discount)]
discounts_2 = [int(i) for i in range(min_discount, (max_discount+1))]
discounts = discounts_1 + discounts_2
discount_rate = sorted(discounts)


# Loops
loop_simulaions = 10

# ANALYTIC SOLUTION FOR MARKOV MATRIX
if analytic_markov:
    print(f"Starting Markov Matrix")
    analytic_Markov_Matrix(file_path=file_start,
                           max_time = max_time,
                           climate_value=climate_value,
                           num_states=number_states,
                           max_area=200)

if new_sdp_data:
    print(f"Starting SDP Model")
    run_SDP_model(file_path_import=file_start,
                  file_path_save=file_start_SDP,
                  max_time = max_time,
                  climate_value=climate_value,
                  num_states=number_states,
                  max_area=200,
                  rate=rate,
                  price_carbon=price_of_carbon,
                  investment_cost=investment_cost,
                  carbon_lost=lost_carbon,
                  discount=discount_rate)

    comparison_of_SDP_data(file_path_import=file_start,
                           file_path_save=file_start,
                           max_time = max_time,
                           discount_options=discount_rate,
                           climate_options=climate_value,
                           num_states=number_states)


if new_simulations:
    print(f"Starting Simulations")

    SMI(file_path= file_start,
        years=max_time,
        loop=loop_simulaions,
        num_state=number_states,
        discount=discount_rate,
        climate_values=climate_value,
        investment=investment_cost,
        max_area=200,
        price_carbon=price_of_carbon,
        sequestration_rate=rate,
        carbon_lost_per_state=lost_carbon,
        cost_per_hectare=cost_per_hectare,
        lost_carbon_per_hectare=lost_carbon_per_hectare
        )
