import os


def analytic_Markov_Matrix(file_path, max_time, climate_value, num_states, max_area):
    import numpy as np
    import pandas as pd
    import string

    for val_states in num_states:

        os.makedirs(f"{file_path}/Markov_Matrix_{max_time}Years/{val_states}_States/", exist_ok=True)

        start_point = 0
        end_point = max_area

        thresholds_unrounded = np.linspace(start_point, end_point, val_states)
        thresholds = np.round(thresholds_unrounded).astype(int)

        print(f"The Thresholds in a {val_states} State System are:\n", thresholds, "\n")

        # States
        states = list(string.ascii_lowercase[:val_states])
        states_index = list(string.ascii_uppercase[:val_states])

        print("\nStart Uniform distribution Climate Model")

        for Cyclone_return in climate_value:
            threshold_gaps = np.diff(thresholds)
            Matrix = np.zeros((val_states, val_states))

            for row in range(val_states):
                for col in range(val_states):
                    if row == 0 and col == 0:
                        Matrix[row, col] = 1
                    elif 0 < col <= row:
                        Matrix[row, col] = (threshold_gaps[row - 1]) / (thresholds[row] + 1)
                    elif col == 0:
                        Matrix[row, col] = 1 / (thresholds[row] + 1)

            P = np.eye(val_states) * (1 - 1 / Cyclone_return) + Matrix * (1 / Cyclone_return)
            df = pd.DataFrame(P).div(P.sum(axis=1), axis=0)
            df.columns = states
            df.index = states_index

            if df.sum(axis=1).between(0.999, 1.001).all():
                print(f"\nMarkov Matrix Average return climate event = {Cyclone_return} Year/s.\nStandard Markov Matrix is Stochastic")
            else:
                print(f"Markov Matrix Average return climate event = {Cyclone_return} Year/s.\nStandard Matrix is not Stochastic")

            outpath = f"{file_path}/Markov_Matrix_{max_time}Years/{val_states}_States"
            df.to_csv(f"{outpath}/Markov_Matrix_Standard_Climate_{Cyclone_return}.csv")

            invest_markov_matrix = df.copy()
            for invest in range(1, val_states):
                invest_markov_matrix = invest_markov_matrix.drop(index=invest_markov_matrix.index[0])
                invest_markov_matrix = pd.concat([invest_markov_matrix, df.iloc[[-1]]], ignore_index=True)

                if invest_markov_matrix.sum(axis=1).between(0.999, 1.001).all():
                    print(f"Invest {invest} Markov Matrix is Stochastic")
                else:
                    print(f"Invest {invest} Markov Matrix is not Stochastic")

                invest_markov_matrix.to_csv(
                    f"{outpath}/Markov_Matrix_Invest_{invest}_Climate_{Cyclone_return}.csv"
                )

        print("\nStart Always goes to zero Climate Model")

        for Cyclone_return in climate_value:
            Matrix = np.zeros((val_states, val_states))
            for row in range(val_states):
                Matrix[row, 0] = 1

            P = np.eye(val_states) * (1 - 1 / Cyclone_return) + Matrix * (1 / Cyclone_return)
            df = pd.DataFrame(P).div(P.sum(axis=1), axis=0)
            df.columns = states
            df.index = states_index

            if df.sum(axis=1).between(0.999, 1.001).all():
                print(f"\nMarkov Matrix Average return climate event = {Cyclone_return} Year/s.\nStandard Markov Matrix is Stochastic")
            else:
                print(f"Markov Matrix Average return climate event = {Cyclone_return} Year/s.\nStandard Matrix is not Stochastic")

            df.to_csv(f"{outpath}/Markov_Matrix_Standard_Climate_Cyclone_to_zero_{Cyclone_return}.csv")

            invest_markov_matrix = df.copy()
            for invest in range(1, val_states):
                invest_markov_matrix = invest_markov_matrix.drop(index=invest_markov_matrix.index[0])
                invest_markov_matrix = pd.concat([invest_markov_matrix, df.iloc[[-1]]], ignore_index=True)

                if invest_markov_matrix.sum(axis=1).between(0.999, 1.001).all():
                    print(f"Invest {invest} Markov Matrix is Stochastic")
                else:
                    print(f"Invest {invest} Markov Matrix is not Stochastic")

                invest_markov_matrix.to_csv(
                    f"{outpath}/Markov_Matrix_Invest_{invest}_Climate_Cyclone_to_zero_{Cyclone_return}.csv"
                )
