# Updated: 05/2024
# Name: Jordan Holdorf
import os


def Simulation_Model_Investment(file_path,
                                years,
                                loop,
                                num_state,
                                discount,
                                climate_values,
                                investment,
                                max_area,
                                price_carbon,
                                sequestration_rate,
                                carbon_lost_per_state,
                                cost_per_hectare,
                                lost_carbon_per_hectare):
    """
    :param years: Max Number of years
    :param loop: Number of simulation loops
    :param num_state: Number of states
    :param discount: Discount Rates
    :param climate_values: Climate Return Values
    :param investment: Cost of Investment
    :param max_area: Maximum area that can be conserved
    :param price_carbon: Price of carbon 
    :param sequestration_rate: Rate of Carbon Sequestration per hectare
    :param carbon_lost: Amount of carbon lost due to investment
    :return:
    """

    import pandas as pd
    import numpy as np
    import random as rnd
    import string

    def _ensure_parent_dir(path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _to_csv(df, path: str, mode='w', **kwargs):
        _ensure_parent_dir(path)
        df.to_csv(path, mode=mode, **kwargs)

    rnd.seed(23)

    invest_once = [1]
    invest_once.extend(num_state)
    num_state = invest_once

    def find_state_index(current_area, states_areas):
        # states_areas is assumed sorted ascending
        if current_area <= 0:
            return 0
        # walk thresholds and pick the first upper bound exceeded
        for i in range(len(states_areas) - 1):
            if states_areas[i] < current_area <= states_areas[i + 1]:
                return i + 1
        return len(states_areas) - 1  # clamp to last bucket

    for climate in climate_values:
        climate_probability = 1 / climate
        climate_event_effect = pd.DataFrame(np.zeros((years, loop), dtype=float))

        for i in range(loop):
            for j in range(years):
                probability = rnd.random()
                if probability > climate_probability:
                    climate_event_effect.iloc[j, i] = 1
                else:
                    percent_drop = rnd.random()

                    climate_event_effect.iloc[j, i] = percent_drop

        os.makedirs(f"{file_path}/Using_investment_plan", exist_ok=True)

        climate_event_effect.to_csv(
            f"{file_path}/Using_investment_plan/Climate_return_{climate}_Seed_data.csv",mode='w')
        print("\nClimate seed values have updated. \n")

        for dis in discount:
            for i in range(len(num_state)):
                val_state = num_state[i]
                start_point = 0
                end_point = max_area

                # Initialise the dataframes
                action = np.zeros(shape=(years, loop), dtype="U100")
                states_overtime = np.zeros(shape=(years, loop), dtype="U100")
                investment_overtime = np.zeros(shape=(years, loop), dtype="float")
                area_overtime = np.zeros(shape=(years, loop), dtype="float")
                Cumulative_area_overtime = np.zeros(shape=(years, loop), dtype="float")
                Carbon_storage = np.zeros(shape=(years, loop), dtype="float")
                Cumulative_Carbon_storage = np.zeros(shape=(years, loop), dtype="float")
                net_position = np.zeros(shape=(years, loop), dtype="float")
                npv = np.zeros(shape=(years, loop), dtype="float")
                pv = np.zeros(shape=(years, loop), dtype="float")
                present_investment = np.zeros(shape=(years, loop), dtype="float")
                profitability_index = np.zeros(shape=(years, loop), dtype="float")

                if val_state == 1:
                    investment_cost = cost_per_hectare * max_area
                    lost_carbon = lost_carbon_per_hectare * max_area
                    for k in range(loop):
                        current_area = 0
                        for year in range(years):
                            discount_factor = (1 + (dis / 100)) ** (year)
                            if year == 0:
                                investment_overtime[year, k] = investment_cost
                                present_investment[year, k] = investment_cost / discount_factor

                                current_area = max_area * climate_event_effect.iloc[year, k]
                                current_area = float(round(current_area, 2))

                                area_overtime[year, k] = current_area
                                Cumulative_area_overtime[year, k] = current_area

                                carbon_sequestered = current_area * sequestration_rate - lost_carbon
                                carbon_price = price_carbon * carbon_sequestered
                                net_price = carbon_price - investment_cost

                                Carbon_storage[year, k] = carbon_sequestered
                                Cumulative_Carbon_storage[year, k] = carbon_sequestered

                                net_position[year, k] = net_price
                                pv[year, k] = carbon_price / discount_factor
                                npv[year, k] = net_price / discount_factor
                                profitability_index[year, k] = (
                                    0 if present_investment[year, k] == 0
                                    else pv[year, k] / present_investment[year, k]
                                )

                            if year > 0:
                                # carry forward investment (no new spend in invest-once case)
                                investment_overtime[year, k] = investment_overtime[year - 1, k]
                                present_investment[year, k] = present_investment[year - 1, k]

                                # evolve area with climate shock
                                current_area = round(current_area * climate_event_effect.iloc[year, k], 2)

                                # benefits (no new investment cost in this branch after year 0)
                                carbon_sequestered = current_area * sequestration_rate
                                carbon_price = price_carbon * carbon_sequestered  # = net benefit this year
                                net_price = carbon_price

                                # state updates
                                area_overtime[year, k] = current_area
                                Cumulative_area_overtime[year, k] = current_area + Cumulative_area_overtime[year - 1, k]
                                Carbon_storage[year, k] = carbon_sequestered
                                Cumulative_Carbon_storage[year, k] = carbon_sequestered + Cumulative_Carbon_storage[
                                    year - 1, k]

                                # finance metrics (add once)
                                net_position[year, k] = net_position[year - 1, k] + net_price
                                pv[year, k] = pv[year - 1, k] + carbon_price / discount_factor
                                npv[year, k] = npv[year - 1, k] + net_price / discount_factor

                                # PI
                                profitability_index[year, k] = (
                                    0 if present_investment[year, k] == 0
                                    else pv[year, k] / present_investment[year, k]
                                )

                    net_position_df = pd.DataFrame(net_position)
                    area_overtime_df = pd.DataFrame(area_overtime)
                    npv_df = pd.DataFrame(npv)
                    pv_df = pd.DataFrame(pv)
                    profitability_index_df = pd.DataFrame(profitability_index)
                    investment_present_df = pd.DataFrame(present_investment)
                    investment_overtime_df = pd.DataFrame(investment_overtime)
                    Action_df = pd.DataFrame(action)

                    average_values_df = pd.DataFrame()
                    average_values_df['Mean_Profit'] = net_position.mean(axis=1)
                    average_values_df['Mean_Total_Investment'] = investment_overtime.mean(axis=1)
                    fifth_percentile_invest = np.percentile(investment_overtime, 5, axis=1)
                    nintyfifth_percentile_invest = np.percentile(investment_overtime, 95, axis=1)
                    average_values_df['Invest_fifth_percentile'] = fifth_percentile_invest
                    average_values_df['Invest_nintyfifth_percentile'] = nintyfifth_percentile_invest

                    average_values_df['Mean_NPV'] = npv.mean(axis=1)
                    average_values_df["Max_NPV"] = npv.max(axis=1)
                    average_values_df["Min_NPV"] = npv.min(axis=1)
                    fifth_percentile_npv = np.percentile(npv, 5, axis=1)
                    nintyfifth_percentile_npv = np.percentile(npv, 95, axis=1)
                    average_values_df['NPV_fifth_percentile'] = fifth_percentile_npv
                    average_values_df['NPV_nintyfifth_percentile'] = nintyfifth_percentile_npv

                    average_values_df['PV'] = pv.mean(axis=1)
                    fifth_percentile_pv = np.percentile(pv, 5, axis=1)
                    nintyfifth_percentile_pv = np.percentile(pv, 95, axis=1)
                    average_values_df['PV_fifth_percentile'] = fifth_percentile_pv
                    average_values_df['PV_nintyfifth_percentile'] = nintyfifth_percentile_pv

                    average_values_df["Present_Investment"] = present_investment.mean(axis=1)
                    average_values_df["Max_Present_Investment"] = present_investment.max(axis=1)
                    average_values_df["Min_Present_Investment"] = present_investment.min(axis=1)
                    fifth_percentile_present_investment = np.percentile(present_investment, 5, axis=1)
                    nintyfifth_percentile_present_investment = np.percentile(present_investment, 95, axis=1)
                    average_values_df['Present_Investment_fifth_percentile'] = fifth_percentile_present_investment
                    average_values_df['Present_Investment_nintyfifth_percentile'] = nintyfifth_percentile_present_investment

                    average_values_df['Profitability_index'] = profitability_index.mean(axis=1)
                    fifth_percentile_index = np.percentile(profitability_index, 5, axis=1)
                    nintyfifth_percentile_index = np.percentile(profitability_index, 95, axis=1)
                    average_values_df['Profitability_index_fifth_percentile'] = fifth_percentile_index
                    average_values_df['Profitability_index_nintyfifth_percentile'] = nintyfifth_percentile_index

                    average_values_df["Count_Negatives"] = (npv < 0).sum(axis=1)
                    average_values_df["Percentage_Negatives"] = average_values_df["Count_Negatives"] / loop

                    average_values_df["Count_Positives"] = (npv > 0).sum(axis=1)
                    average_values_df["Percentage_Positive"] = average_values_df["Count_Positives"] / loop

                    average_values_df["Count_Invest"] = (action == "Invest Single").sum(axis=1)
                    average_values_df["Percentage_Invest"] = average_values_df["Count_Invest"] / loop

                    average_values_df['Mean_Area'] = area_overtime.mean(axis=1)
                    fifth_percentile_area = np.percentile(area_overtime, 5, axis=1)
                    nintyfifth_percentile_area = np.percentile(area_overtime, 95, axis=1)
                    average_values_df['Area_fifth_percentile'] = fifth_percentile_area
                    average_values_df['Area_nintyfifth_percentile'] = nintyfifth_percentile_area

                    average_values_df['Mean_Cumulative_Area'] = Cumulative_area_overtime.mean(axis=1)
                    fifth_percentile_c_area = np.percentile(Cumulative_area_overtime, 5, axis=1)
                    nintyfifth_percentile_c_area = np.percentile(Cumulative_area_overtime, 95, axis=1)
                    average_values_df['Cumulative_Area_fifth_percentile'] = fifth_percentile_c_area
                    average_values_df['Cumulative_Area_nintyfifth_percentile'] = nintyfifth_percentile_c_area

                    average_values_df['Mean_Carbon_Sequestration'] = Carbon_storage.mean(axis=1)
                    average_values_df['Mean_Cumulative_Carbon_Sequestration'] = Cumulative_Carbon_storage.mean(axis=1)
                    fifth_percentile_carbon = np.percentile(Cumulative_Carbon_storage, 5, axis=1)
                    nintyfifth_percentile_carbon = np.percentile(Cumulative_Carbon_storage, 95, axis=1)
                    average_values_df['Carbon_fifth_percentile'] = fifth_percentile_carbon
                    average_values_df['Carbon_nintyfifth_percentile'] = nintyfifth_percentile_carbon

                    # Build a base folder for this discount × climate combo
                    base_dir = os.path.join(
                        file_path, "Using_investment_plan", "Invest_Once_States",
                        f"Discount_{dis}", f"Climate_{climate}"
                    )

                    # Write outputs (parents will be created automatically)
                    _to_csv(pd.DataFrame(net_position), os.path.join(base_dir, "Net_position.csv"))
                    _to_csv(pd.DataFrame(area_overtime), os.path.join(base_dir, "Area.csv"))
                    _to_csv(pd.DataFrame(npv), os.path.join(base_dir, "npv.csv"))
                    _to_csv(pd.DataFrame(pv), os.path.join(base_dir, "pv.csv"))
                    _to_csv(pd.DataFrame(profitability_index), os.path.join(base_dir, "Ratio.csv"))
                    _to_csv(pd.DataFrame(present_investment), os.path.join(base_dir, "present_investment.csv"))
                    _to_csv(pd.DataFrame(investment_overtime), os.path.join(base_dir, "Investment_overtime.csv"))
                    _to_csv(pd.DataFrame(action), os.path.join(base_dir, "Action_taken.csv"))
                    _to_csv(average_values_df, os.path.join(base_dir, "Average_Information.csv"))

                    print(f"Run for {climate} Year Climate Return, {dis}%, Discount Rate and Invest Once. \n")

                if val_state != 1:
                    investment_cost = investment[i-1]
                    carbon_lost = carbon_lost_per_state[i-1]
                    thresholds_unrounded = np.linspace(start_point, end_point, val_state)
                    thresholds = np.round(thresholds_unrounded).astype(int)
                    states_areas = np.sort(thresholds)

                    # States
                    alphabet_string_lower = string.ascii_lowercase
                    alphabet_list_lower = list(alphabet_string_lower)
                    states_list_lower = alphabet_list_lower[:-(len(alphabet_list_lower) - val_state)]
                    states = np.array(states_list_lower, dtype=str)

                    # States Index
                    alphabet_string = string.ascii_uppercase
                    alphabet_list = list(alphabet_string)
                    states_list = alphabet_list[:-(len(alphabet_list) - (len(thresholds)))]
                    states_index = np.array(states_list, dtype=str)

                    states = states

                    investment_plan = pd.read_csv(
                        f"{file_path}/SDP_Outputs/{val_state}_States/Discount_{dis}/Policy_output_{climate}.csv")

                    investment_plan.drop(columns=["Unnamed: 0", "Value"], axis="columns", inplace=True)

                    for k in range(loop):
                        state_loc_options = np.arange(start=0, stop=len(states), step=1)
                        current_area = 0
                        for year in range(years):
                            discount_factor = (1 + (dis / 100)) ** (year)

                            threshold_index = find_state_index(current_area, states_areas)

                            current_state = states[threshold_index]
                            states_overtime[year, k] = current_state
                            strategy = investment_plan[(investment_plan["State"] == current_state) & (investment_plan["Years"] == (year+1))]
                            current_action = strategy.iat[0, 2]
                            action[year, k] = current_action

                            if current_action != "No Investment" and threshold_index < len(states):

                                investment_index = current_action.index('_') + 1
                                Num_invest = int(current_action[investment_index:])

                                if year == 0:
                                    investment_overtime[year, k] = (investment_cost*Num_invest)
                                    present_investment[year, k] = (investment_cost*Num_invest) / discount_factor
                                    state_position = threshold_index + Num_invest
                                    state_position = min(state_position, len(states_areas) - 1)  # clamp
                                    current_area = states_areas[state_position] * climate_event_effect.iloc[year, k]
                                    current_area = round(current_area, 2)
                                    carbon_sequestered = current_area * sequestration_rate - (carbon_lost*Num_invest)
                                    carbon_price = price_carbon * carbon_sequestered
                                    net_price = carbon_price - investment_cost*Num_invest

                                    area_overtime[year, k] = current_area
                                    Cumulative_area_overtime[year, k] = current_area

                                    Carbon_storage[year, k] = carbon_sequestered
                                    Cumulative_Carbon_storage[year, k] = carbon_sequestered

                                    net_position[year, k] = net_price
                                    pv[year, k] = carbon_price / discount_factor
                                    npv[year, k] = net_price / discount_factor

                                else:
                                    investment_overtime[year, k] = investment_overtime[year - 1, k] + \
                                                                   investment_cost*Num_invest
                                    present_investment[year, k] = present_investment[year - 1, k] + \
                                                                  ((investment_cost*Num_invest) / discount_factor)
                                    state_position = threshold_index + Num_invest
                                    state_position = min(state_position, len(states_areas) - 1)  # clamp
                                    current_area = states_areas[state_position] * climate_event_effect.iloc[year, k]
                                    current_area = round(current_area, 2)

                                    carbon_sequestered = current_area * sequestration_rate - (carbon_lost*Num_invest)
                                    carbon_price = price_carbon * carbon_sequestered

                                    net_price = carbon_price - (investment_cost*Num_invest)

                                    area_overtime[year, k] = current_area
                                    Cumulative_area_overtime[year, k] = current_area + \
                                                                        Cumulative_area_overtime[year - 1, k]

                                    Carbon_storage[year, k] = carbon_sequestered
                                    Cumulative_Carbon_storage[year, k] = carbon_sequestered + \
                                                                         Cumulative_Carbon_storage[year - 1, k]

                                    net_position[year, k] = net_position[year - 1, k] + net_price
                                    pv[year, k] = pv[year - 1, k] + carbon_price / discount_factor
                                    npv[year, k] = npv[year - 1, k] + net_price / discount_factor

                            else:
                                investment_overtime[year, k] = investment_overtime[year - 1, k]
                                present_investment[year, k] = present_investment[year - 1, k]
                                state_position = threshold_index
                                state_position = min(state_position, len(states_areas) - 1)  # clamp
                                current_area = current_area * climate_event_effect.iloc[year, k]
                                current_area = round(current_area, 2)

                                carbon_sequestered = current_area * sequestration_rate
                                net_price = price_carbon * carbon_sequestered

                                net_position[year, k] = net_price + net_position[year - 1, k]

                                area_overtime[year, k] = current_area
                                Cumulative_area_overtime[year, k] = current_area + Cumulative_area_overtime[year - 1, k]

                                Carbon_storage[year, k] = carbon_sequestered
                                Cumulative_Carbon_storage[year, k] = carbon_sequestered + \
                                                                     Cumulative_Carbon_storage[year - 1, k]

                                pv[year, k] = pv[year - 1, k] + net_price / discount_factor
                                npv[year, k] = npv[year - 1, k] + net_price / discount_factor

                            profitability_index[year, k] = (
                                0 if present_investment[year, k] == 0
                                else pv[year, k] / present_investment[year, k]
                            )

                    net_position_df = pd.DataFrame(net_position)
                    npv_df = pd.DataFrame(npv)
                    pv_df = pd.DataFrame(pv)
                    profitability_index_df = pd.DataFrame(profitability_index)
                    investment_present_df = pd.DataFrame(present_investment)
                    investment_overtime_df = pd.DataFrame(investment_overtime)
                    Action_df = pd.DataFrame(action)
                    area_overtime_df = pd.DataFrame(area_overtime)

                    average_values_df = pd.DataFrame()
                    average_values_df['Mean_Profit'] = net_position.mean(axis=1)
                    average_values_df['Mean_Total_Investment'] = investment_overtime.mean(axis=1)
                    fifth_percentile_invest = np.percentile(investment_overtime, 5, axis=1)
                    nintyfifth_percentile_invest = np.percentile(investment_overtime, 95, axis=1)
                    average_values_df['Invest_fifth_percentile'] = fifth_percentile_invest
                    average_values_df['Invest_nintyfifth_percentile'] = nintyfifth_percentile_invest

                    average_values_df['Mean_NPV'] = npv.mean(axis=1)
                    average_values_df["Max_NPV"] = npv.max(axis=1)
                    average_values_df["Min_NPV"] = npv.min(axis=1)
                    fifth_percentile_npv = np.percentile(npv, 5, axis=1)
                    nintyfifth_percentile_npv = np.percentile(npv, 95, axis=1)
                    average_values_df['NPV_fifth_percentile'] = fifth_percentile_npv
                    average_values_df['NPV_nintyfifth_percentile'] = nintyfifth_percentile_npv

                    average_values_df['PV'] = pv.mean(axis=1)
                    fifth_percentile_pv = np.percentile(pv, 5, axis=1)
                    nintyfifth_percentile_pv = np.percentile(pv, 95, axis=1)
                    average_values_df['PV_fifth_percentile'] = fifth_percentile_pv
                    average_values_df['PV_nintyfifth_percentile'] = nintyfifth_percentile_pv

                    average_values_df["Present_Investment"] = present_investment.mean(axis=1)
                    average_values_df["Max_Present_Investment"] = present_investment.max(axis=1)
                    average_values_df["Min_Present_Investment"] = present_investment.min(axis=1)
                    fifth_percentile_present_investment = np.percentile(present_investment, 5, axis=1)
                    nintyfifth_percentile_present_investment = np.percentile(present_investment, 95, axis=1)
                    average_values_df['Present_Investment_fifth_percentile'] = fifth_percentile_present_investment
                    average_values_df['Present_Investment_nintyfifth_percentile'] = nintyfifth_percentile_present_investment

                    average_values_df['Profitability_index'] = profitability_index.mean(axis=1)
                    fifth_percentile_index = np.percentile(profitability_index, 5, axis=1)
                    nintyfifth_percentile_index = np.percentile(profitability_index, 95, axis=1)
                    average_values_df['Profitability_index_fifth_percentile'] = fifth_percentile_index
                    average_values_df['Profitability_index_nintyfifth_percentile'] = nintyfifth_percentile_index

                    average_values_df["Count_Negatives"] = (npv < 0).sum(axis=1)
                    average_values_df["Percentage_Negatives"] = average_values_df["Count_Negatives"] / loop

                    average_values_df["Count_Positives"] = (npv > 0).sum(axis=1)
                    average_values_df["Percentage_Positive"] = average_values_df["Count_Positives"] / loop

                    average_values_df["Count_Invest"] = (action == "Invest Single").sum(axis=1)
                    average_values_df["Percentage_Invest"] = average_values_df["Count_Invest"] / loop

                    average_values_df['Mean_Area'] = area_overtime.mean(axis=1)
                    fifth_percentile_area = np.percentile(area_overtime, 5, axis=1)
                    nintyfifth_percentile_area = np.percentile(area_overtime, 95, axis=1)
                    average_values_df['Area_fifth_percentile'] = fifth_percentile_area
                    average_values_df['Area_nintyfifth_percentile'] = nintyfifth_percentile_area

                    average_values_df['Mean_Cumulative_Area'] = Cumulative_area_overtime.mean(axis=1)
                    fifth_percentile_c_area = np.percentile(Cumulative_area_overtime, 5, axis=1)
                    nintyfifth_percentile_c_area = np.percentile(Cumulative_area_overtime, 95, axis=1)
                    average_values_df['Cumulative_Area_fifth_percentile'] = fifth_percentile_c_area
                    average_values_df['Cumulative_Area_nintyfifth_percentile'] = nintyfifth_percentile_c_area

                    average_values_df['Mean_Carbon_Sequestration'] = Carbon_storage.mean(axis=1)
                    average_values_df['Mean_Cumulative_Carbon_Sequestration'] = Cumulative_Carbon_storage.mean(axis=1)
                    fifth_percentile_carbon = np.percentile(Cumulative_Carbon_storage, 5, axis=1)
                    nintyfifth_percentile_carbon = np.percentile(Cumulative_Carbon_storage, 95, axis=1)
                    average_values_df['Carbon_fifth_percentile'] = fifth_percentile_carbon
                    average_values_df['Carbon_nintyfifth_percentile'] = nintyfifth_percentile_carbon

                    base_dir = os.path.join(
                        file_path, "Using_investment_plan", f"{val_state}_States",
                        f"Discount_{dis}", f"Climate_{climate}"
                    )

                    _to_csv(net_position_df, os.path.join(base_dir, "Net_position.csv"))
                    _to_csv(npv_df, os.path.join(base_dir, "npv.csv"))
                    _to_csv(pv_df, os.path.join(base_dir, "pv.csv"))
                    _to_csv(profitability_index_df, os.path.join(base_dir, "Ratio.csv"))
                    _to_csv(investment_present_df, os.path.join(base_dir, "present_investment.csv"))
                    _to_csv(investment_overtime_df, os.path.join(base_dir, "Investment_overtime.csv"))
                    _to_csv(Action_df, os.path.join(base_dir, "Action_taken.csv"))
                    _to_csv(area_overtime_df, os.path.join(base_dir, "Area_overtime.csv"))
                    _to_csv(average_values_df, os.path.join(base_dir, "Average_Information.csv"))

                    print(f"Run for {climate} Year Climate Return, {dis}%, Discount Rate and {val_state} States. \n")

                    count_states = pd.DataFrame()

                    for state in states_index:
                        Count_value = (states_overtime == state.lower()).sum(axis=1)
                        count_states[f"{state}_Percentage"] = Count_value / loop

                        _to_csv(count_states, os.path.join(base_dir, "Count_States.csv"))
