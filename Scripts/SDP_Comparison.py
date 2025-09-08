def comparison_of_SDP_data(file_path_import, file_path_save, discount_options, num_states, climate_options, max_time):
    """
    :param discount_options: Discount Rates to be Compared
    :param num_states: Possible number of states
    :param climate_options: Climate return times to be compared
    :param max_time: Maximum amount of time
    :return:
    """

    import os
    import numpy as np
    import string
    import pandas as pd


    def process_data(max_time, val_states, discount, climate_value):
        alphabet_string_lower = string.ascii_lowercase
        alphabet_list_lower = list(alphabet_string_lower)
        states_list_lower = alphabet_list_lower[:-(len(alphabet_list_lower) - val_states)]
        states = np.array(states_list_lower, dtype=str)

        # Define paths for input and output directories
        # Input Data
        sdp_path = f"{file_path_import}/SDP_Outputs_{max_time}Years/{val_states}_States"

        # Output Data
        comparison_path_discount = f"{file_path_save}/Comparison_Discount_Rates_{max_time}/{val_states}_States"
        comparison_path_climate = f"{file_path_save}/Comparison_Climate_{max_time}/{val_states}_States"

        # Initialize DataFrame for discount rate comparison

        # Process discount rates comparison

        for alpha in climate_value:
            discount_comparison = pd.DataFrame(index=discount, columns=states)
            for beta, discount_value in enumerate(discount):
                try:
                    # Construct path to data file
                    data_path = f"{sdp_path}/Discount_{discount_value}/Policy_output_{alpha}.csv"
                    # Read data and prepare for comparison
                    data = pd.read_csv(data_path, index_col=['Years', 'State'])
                    data.drop(columns=['Unnamed: 0', 'Value'], inplace=True)
                    data = data.unstack(level='State')

                    last_investment_years = {}

                    # Iterate over unique state labels
                    for state in data.columns.get_level_values('State').unique():
                        state_data = data[('Policy', state)]

                        if any(state_data != 'No Investment'):
                            # Find the last index where investment is not 'No Investment'
                            last_index = state_data[state_data != 'No Investment'].index[-1]
                            # Get the corresponding year
                            last_year = data.index[last_index]

                            discount_comparison.loc[discount_value, state] = last_year
                        else:
                            # Handle case where all values are 'No Investment'
                            discount_comparison.loc[discount_value, state] = 0 # or any other appropriate handling

                except FileNotFoundError:
                    print(f"File not found: {data_path}")

            # Create output directory if it doesn't exist
            os.makedirs(comparison_path_discount, exist_ok=True)
            # Save discount comparison results to CSV
            discount_comparison.to_csv(f"{comparison_path_discount}/Climate_{alpha}.csv")

            print(f"Comparison of Discount Rates has been completed for {alpha} Year Climate Return.")

        # Initialize DataFrame for climate return times comparison

        # Process climate return times comparison
        for discount_value in discount:
            climate_comparison = pd.DataFrame(index=climate_value, columns=states)
            for kappa, climate in enumerate(climate_value):
                try:
                    # Construct path to data file
                    data_path = f"{sdp_path}/Discount_{discount_value}/Policy_output_{climate}.csv"
                    # Read data and prepare for comparison
                    data = pd.read_csv(data_path, index_col=['Years', 'State'])
                    data.drop(columns=['Unnamed: 0', 'Value'], inplace=True)
                    data = data.unstack(level='State')

                    last_investment_years = {}

                    # Iterate over unique state labels
                    for state in data.columns.get_level_values('State').unique():
                        state_data = data[('Policy', state)]

                        if any(state_data != 'No Investment'):
                            # Find the last index where investment is not 'No Investment'
                            last_index = state_data[state_data != 'No Investment'].index[-1]
                            # Get the corresponding year
                            last_year = data.index[last_index]

                            climate_comparison.loc[climate, state] = last_year
                        else:
                            # Handle case where all values are 'No Investment'
                            climate_comparison.loc[climate, state] = 0  # or any other appropriate handling

                except FileNotFoundError:
                    print(f"File not found: {data_path}")

            # Create output directory if it doesn't exist
            os.makedirs(comparison_path_climate, exist_ok=True)
            # Save climate comparison results to CSV
            climate_comparison.to_csv(f"{comparison_path_climate}/Discount_{discount_value}.csv")

            print(f"Comparison of Climate Return times has been completed for {discount_value}% Discount.")


    for num in num_states:
        process_data(max_time, num, discount_options, climate_options)
        print(f"Run for {num} States. \n")

    print("All Completed")