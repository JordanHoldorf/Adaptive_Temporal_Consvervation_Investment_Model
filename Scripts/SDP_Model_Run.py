# Jordan Holdorf
# 21/11/2023
import mdptoolbox.util


def run_SDP_model(file_path_import, file_path_save, max_time, climate_value, num_states, max_area, rate, price_carbon,
                  investment_cost, carbon_lost, discount):

    """
    :param max_time: Maximum number of years for run time
    :param climate_value: Vector of all Climate risks being investigated (Number of years between cyclones)
    :param num_states: Number of states
    :param max_area: Maximum Amount fo area for the system
    :param rate: Rate of Carbon sequestration (Known value)
    :param price_carbon: Price of carbon (Known value)
    :param investment_cost: Cost of investing in Restoration (known value)
    :param carbon_lost: Carbon lost due to investing (Known Value)
    :param discount: Discount rate (as a percentage - code coverts to decimal value) (vector of possible)
    :return:
    """

    # Import Packages:
    import numpy as np
    import pandas as pd
    import mdptoolbox as mdp
    import string
    import os

    for i_num in range(len(num_states)):

        val_states = num_states[i_num]
        start_point = 0
        end_point = max_area

        thresholds_unrounded = np.linspace(start_point, end_point, val_states)
        thresholds = np.round(thresholds_unrounded).astype(int)

        # States
        alphabet_string_lower = string.ascii_lowercase
        alphabet_list_lower = list(alphabet_string_lower)
        states_list_lower = alphabet_list_lower[:-(len(alphabet_list_lower) - val_states)]
        states = np.array(states_list_lower, dtype=str)

        # States Index
        alphabet_string = string.ascii_uppercase
        alphabet_list = list(alphabet_string)
        states_list = alphabet_list[:-(len(alphabet_list) - (len(thresholds)))]
        states_index = np.array(states_list, dtype=str)

        # Variables:
        # Number of years
        Nmax = max_time

        # Area of the states
        Mangrove_area = thresholds

        # Price of Carbon
        P_Carbon = price_carbon

        # Rate of Sequestration:
        R_Carbon = rate

        # Carbon lost due to investment
        I_Carbon = carbon_lost[i_num]
        # I_Carbon_2 = carbon_lost*1.75

        # Cost of investment
        Investment_Cost_1 = investment_cost[i_num]
        #print(f"Investment cost per state is ${Investment_Cost_1}. \n"
              #f"Carbon Lost Per Investment is {I_Carbon} Hectares.")

        for j in range(len(discount)):
            #print(f"Discount {discount[j]}%")
            discount_rate = discount[j]
            discount_factor = 1 / (1 + discount_rate * 0.01)

            os.makedirs(f"{file_path_save}/{val_states}_States/Discount_{discount_rate}", exist_ok=True)

            for i in climate_value:
                no_investment = pd.DataFrame()
                invest_1 = pd.DataFrame()

                cyclone_years = i  # years between events

                no_investment = pd.read_csv(
                    f"{file_path_import}/Markov_Matrix_{Nmax}Years/{val_states}_States/"
                    f"Markov_Matrix_Standard_Climate_{cyclone_years}.csv")
                no_investment.drop("Unnamed: 0", axis="columns", inplace=True)
                no_investment.columns = [np.arange(0, no_investment.shape[1])]
                no_investment.columns = [np.arange(0, no_investment.shape[1])]
                no_investment = no_investment.to_numpy()
                actions = ['No Investment']

                invest_arrays = []

                for l in range(val_states - 1):
                    investment = l + 1
                    invest = pd.read_csv(
                        f"{file_path_import}/Markov_Matrix_{Nmax}Years/{val_states}_States/"
                        f"Markov_Matrix_Invest_{investment}_Climate_{cyclone_years}.csv")
                    invest.drop("Unnamed: 0", axis="columns", inplace=True)

                    invest.columns = [np.arange(0, invest.shape[1])]
                    invest_array = invest.to_numpy()
                    invest_arrays.append(invest_array)

                    actions.append(f"Invest_{investment}")

                invest_3d_array = np.stack(invest_arrays, axis=0)
                no_investment_3d = np.expand_dims(no_investment, axis=0)

                transitions = np.concatenate((no_investment_3d, invest_3d_array), axis=0)

                value = np.zeros([len(actions), len(states), len(states)])

                for a in range(len(actions)):
                    if a == 0:
                        for b in range(len(states)):
                            for c in range(len(states)):
                                carbon = Mangrove_area[c] * R_Carbon
                                value[a, b, c] = carbon * P_Carbon
                    if a > 0:
                        for b in range(len(states)):
                            for c in range(len(states)):
                                carbon = Mangrove_area[c] * R_Carbon - (I_Carbon * a)
                                value[a, b, c] = carbon * P_Carbon - (Investment_Cost_1 * a)

                mdp.util.check(P=transitions, R=value)

                sdp = mdp.mdp.FiniteHorizon(transitions=transitions, reward=value, discount=discount_factor, N=Nmax)

                sdp.run()
                value_df = pd.DataFrame(sdp.V)
                value_df.insert(0, "State", states)
                value_df = pd.melt(value_df, id_vars='State', var_name="Years", value_name="Value")

                policy_df = pd.DataFrame(sdp.policy)
                policy_df.insert(0, "State", states)
                policy_df = pd.melt(policy_df, id_vars='State', var_name="Years", value_name="Policy")

                policy_df = pd.merge(value_df, policy_df)

                # Ensure that the 'Policy' column is of type string (This is the first change)
                policy_df['Policy'] = policy_df['Policy'].astype(str)

                for j in range(policy_df.shape[0]):
                    policy_df.loc[j, 'Years'] = policy_df.loc[j, 'Years'] + 1

                for j in range(policy_df.shape[0]):
                    for p in range(val_states):
                        if policy_df.loc[j, 'Policy'] == str(p):  # Convert 'p' to string (This is the second change)
                            policy_df.loc[j, 'Policy'] = actions[p]

                # Save Dataframe
                if Nmax == 25:
                    policy_df.to_csv(f"{file_path_save}/{val_states}_States/Discount_{discount_rate}/"
                                     f"Policy_output_{cyclone_years}.csv")

            for i in climate_value:
                no_investment = pd.DataFrame()
                invest_1 = pd.DataFrame()

                cyclone_years = i  # years between events

                no_investment = pd.read_csv(
                    f"{file_path_import}/Markov_Matrix_{Nmax}Years/{val_states}_States/"
                    f"Markov_Matrix_Standard_Climate_Cyclone_to_zero_{cyclone_years}.csv")
                no_investment.drop("Unnamed: 0", axis="columns", inplace=True)
                no_investment.columns = [np.arange(0, no_investment.shape[1])]
                no_investment.columns = [np.arange(0, no_investment.shape[1])]
                no_investment = no_investment.to_numpy()
                actions = ['No Investment']

                invest_arrays = []

                for l in range(val_states - 1):
                    investment = l + 1
                    invest = pd.read_csv(
                        f"{file_path_import}/Markov_Matrix_{Nmax}Years/{val_states}_States/"
                        f"Markov_Matrix_Invest_{investment}_Climate_Cyclone_to_zero_{cyclone_years}.csv")
                    invest.drop("Unnamed: 0", axis="columns", inplace=True)

                    invest.columns = [np.arange(0, invest.shape[1])]
                    invest_array = invest.to_numpy()
                    invest_arrays.append(invest_array)

                    actions.append(f"Invest_{investment}")

                invest_3d_array = np.stack(invest_arrays, axis=0)
                no_investment_3d = np.expand_dims(no_investment, axis=0)

                transitions = np.concatenate((no_investment_3d, invest_3d_array), axis=0)

                value = np.zeros([len(actions), len(states), len(states)])

                for a in range(len(actions)):
                    if a == 0:
                        for b in range(len(states)):
                            for c in range(len(states)):
                                carbon = Mangrove_area[c] * R_Carbon
                                value[a, b, c] = carbon * P_Carbon
                    if a > 0:
                        for b in range(len(states)):
                            for c in range(len(states)):
                                carbon = Mangrove_area[c] * R_Carbon - (I_Carbon * a)
                                value[a, b, c] = carbon * P_Carbon - (Investment_Cost_1 * a)

                mdp.util.check(P=transitions, R=value)

                sdp = mdp.mdp.FiniteHorizon(transitions=transitions, reward=value, discount=discount_factor, N=Nmax)

                sdp.run()
                value_df = pd.DataFrame(sdp.V)
                value_df.insert(0, "State", states)
                value_df = pd.melt(value_df, id_vars='State', var_name="Years", value_name="Value")

                policy_df = pd.DataFrame(sdp.policy)
                policy_df.insert(0, "State", states)
                policy_df = pd.melt(policy_df, id_vars='State', var_name="Years", value_name="Policy")

                policy_df = pd.merge(value_df, policy_df)
                policy_df['Policy'] = policy_df['Policy'].astype(str)

                for j in range(policy_df.shape[0]):
                    policy_df.loc[j, 'Years'] = policy_df.loc[j, 'Years'] + 1

                for j in range(policy_df.shape[0]):
                    for p in range(val_states):
                        if policy_df.loc[j, 'Policy'] == str(p):  # Convert 'p' to string (This is the second change)
                            policy_df.loc[j, 'Policy'] = actions[p]

                # Save Dataframe
                if Nmax == 25:
                    policy_df.to_csv(f"{file_path_save}/{val_states}_States/Discount_{discount_rate}/Policy_output_{cyclone_years}_Go_to_zero.csv")