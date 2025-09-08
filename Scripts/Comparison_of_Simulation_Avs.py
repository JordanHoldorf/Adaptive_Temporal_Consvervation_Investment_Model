def comparison_simulation(climate, discount_rate, num_states, maxtime):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    title_font = {'fontname': 'Arial', 'size': '50', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '40', 'color': 'black', 'weight': 'normal'}
    lim_font = {'fontname': 'Arial', 'size': '40', 'color': 'black', 'weight': 'normal'}

    discount_rate = discount_rate

    for val in range(len(num_states)):
        val_state = num_states[val]
        print(val_state)

        for j in discount_rate:

            # Initialise Dataframes
            dis = j
            percentage_positive = pd.DataFrame()
            mean_npv = pd.DataFrame()
            mean_present_value_investment = pd.DataFrame()
            percentage_invest = pd.DataFrame()
            for rep in climate:

                sim_data = pd.read_csv(f"../Data/Using_investment_plan/{val_state}_States/Discount_{dis}/Climate_{rep}/Average_Information.csv")

                percentage_positive[f"{rep}"] = sim_data.loc[:, "Percentage_Positive"]
                mean_npv[f"{rep}"] = sim_data.loc[:, "NPV"]
                mean_present_value_investment[f"{rep}"] = sim_data.loc[:, "Present_Investment"]
                percentage_invest[f"{rep}"] = sim_data.loc[:, "Percentage_Invest"]

            percentage_invest.to_csv(
                f"../Data/Using_investment_plan/{val_state}_States/Discount_{dis}/Percentage_invest.csv")

            mean_npv.to_csv(
                f"../Data/Using_investment_plan/{val_state}_States/Discount_{dis}/mean_npv.csv")

            final_percentage = percentage_positive.iloc[(maxtime-1), :].to_numpy()
            total_npv = mean_npv.iloc[(maxtime-1), :].to_numpy()
            total_present_value_invest = mean_present_value_investment.iloc[(maxtime-1), :].to_numpy()

            x_axis = climate
            print(x_axis)

            fig = plt.figure(figsize=(25, 20))
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(x_axis, final_percentage, marker='o')  # Changed to plot
            ax1.set_xlabel("Return period of Climate Events (years)", **axis_font)
            ax1.set_ylabel("Percentage (%)", **axis_font)
            ax1.set_title(f"Percentage of positive NPV in final year", **title_font)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(x_axis, total_present_value_invest, marker='o', label="Amount Invested")  # Changed to plot
            ax2.plot(x_axis, total_npv, marker='o', label="NPV")  # Changed to plot
            ax2.set_xlabel("Return period of Climate Events (years)", **axis_font)
            ax2.set_ylabel("Present Value ($)", **axis_font)
            ax2.ticklabel_format(style="plain", axis="y")
            ax2.set_title(f"NPV compared to investment", **title_font)
            plt.legend(title_fontsize=60, fontsize="50")
            plt.suptitle(f"Discount Rate of {dis}%")
            plt.tight_layout()
            plt.savefig(f"../Figures/Simulation_of_investment_plan/{val_state}_States/Discount_{dis}"
                        f"/Profit_and_Investment_and_Percentage_positive_{maxtime}.png")
            plt.clf()

            fig2, ax = plt.subplots(figsize=(25, 20))  # Create a single subplot for the entire figure

            # Plot the second dataset (NPV)
            npv_line, = ax.plot(x_axis, total_npv, marker='o', color='black')

            # Fill the area between the second dataset and x-axis
            ax.fill_between(x_axis, total_npv, 0, color='blue', alpha=0.5)

            # Set labels, title, and legend
            ax.set_xlabel("Return period of Climate Events (years)", **axis_font)
            ax.set_ylabel("Net Present Value ($)", **axis_font)
            ax.ticklabel_format(style="plain", axis="y")
            ax.set_title("NPV compared to investment", **title_font)
            plt.tight_layout()

            fig2.savefig(f"../Figures/Simulation_of_investment_plan/{val_state}_States/Discount_{dis}"
                        f"/NPV.png")
            plt.close(fig2)

        print(f"Code run to completion for {val_state} States")
    print(f"Code run to completion")