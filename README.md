# Adaptive Temporal Conservation Investment Model

Code for modelling adaptive investment strategies in ecological restoration under climate risk.

This repository implements a stochastic dynamic programming (SDP) framework and simulations of applying adaptive, year-by-year investment plans. These are compared with traditional “invest once” strategies under uncertain climate disturbances. The mangrove restoration case example evaluates outcomes in terms of Net Present Value (NPV), Profitability Index (PI), and environmental performance (area and carbon).

> Note: This repository contains code only. Outputs (CSV, images, etc.) are written to `Data/` and are excluded from version control via `.gitignore`.

---

## Requirements

Python 3.9+ is recommended.

Core packages:
- numpy  
- pandas  
- pymdptoolbox (aka mdptoolbox, for solving the stochastic dynamic program)  
- matplotlib (for any plotting)

To set up, create and activate a virtual environment, then install the above packages.

---

## Repository Structure

All code is contained in the `Scripts/` folder:

- `Analytic_Solution.py` – analytic baseline solution for comparison  
- `Comparison_of_Simulation_Avs.py` – utilities to compare simulated vs. analytic solutions  
- `SDP_Model_Run.py` – runs the stochastic dynamic programming (SDP) optimisation  
- `SDP_Comparison.py` – compares optimised (SDP) strategies with traditional approaches  
- `Simulation_using_investment_plan.py` – simulation engine that applies investment plans over time  
- `Temporal_model.py` – main entry point for running temporal investment strategies  

Outputs (CSV, figures) are saved to `/Data/` (created automatically if missing).  
This folder is excluded from version control.

---

## Usage

1. Clone the repository and set up the environment as described above.  
2. Navigate to the `Scripts/` folder.  
3. Run `Temporal_model.py` to generate outputs.  

Outputs will be written under `/Data/Using_investment_plan/`, organised by strategy, discount rate, and climate return period.

---

## Configuration and Parameters

The model requires ecological, financial, and climatic parameters to run. In the manuscript analysis, the following baseline values were used:

- **Project time horizon**: 25 years
- **Target restoration area**: 200 hectares
- **Investment Cost**: 25 years
- **Restoration cost per hectare**: $3,875.50 USD per hectare (includes establishment and annual maintenance costs)
- **Carbon sequestration rate**: 6.32 Tonnes of carbon per hectare per year
- **Carbon Emissions due to restoration activities**: 0.5 Tonnes of carbon per hectare
- **Carbon price**: 66 USD tCO₂⁻¹ (constant)  
- **Cyclone/climate return periods**: 2–100 years (varied across senarios)
- **Discount rates**: 3%–15%  (varied across senarios)
- **Restoration units ("states")**: varies (e.g., 3, 5, 7, 9, 15)

These values were chosen to represent realistic but stylised conditions based on published mangrove restoration data.

### Adjusting Parameters
To adapt the model to a new context, users will need to edit the input values in the scripts:

- `SDP_Model_Run.py`: controls stochastic dynamic programming runs and state definitions.  
- `Simulation_using_investment_plan.py`: applies investment plans with specified discount rates, climate return periods, and restoration costs.  
- `Temporal_model.py`: main entry point where project horizon, parameter ranges, and file paths are set.  

Outputs automatically adjust to the parameters chosen.

---


## Outputs

Each run generates CSV files including:
- Net Position (`Net_position.csv`)  
- Net Present Value (`npv.csv`)  
- Profitability Index (`Ratio.csv`)  
- Investment Over Time (`Investment_overtime.csv`)  
- Carbon and Area Sequestration (`Carbon.csv`, `Area.csv`)  
- Summary statistics (`Average_Information.csv`)  

---

## Citation

If you use this code, please cite the code.  
Citation metadata is provided in the [`CITATION.cff`](./CITATION.cff) file, which GitHub also uses to generate ready-to-use citations (APA, BibTeX, etc.) via the “Cite this repository” button.

---

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
