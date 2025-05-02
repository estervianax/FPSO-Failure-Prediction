import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataExplorator():
    def get_failure_events(self,df, fail_col ='fail'):
        """
        Calculates failure statistics from a time-ordered dataframe.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing a boolean column indicating failure states.
        fail_col : str
            Name of the column that indicates whether a failure occurred (default is 'fail').

        Returns:
        -------
        overall_failures : int
            Total number of cycles where the system was in a failed state (fail == True).
        shift_to_failures : int
            Number of distinct failure events, defined as transitions from non-failure to failure.
        """
        overall_failures = df[(df[fail_col] == True)]
        shift_to_failures = df[(df[fail_col] == True) & (df['fail_continued'] == False)]
        print(f'Overall failures: {overall_failures.shape[0]}')
        print(f'Shift to failures: {shift_to_failures.shape[0]}')
    
    def plot_target_balance(self, df, target = 'fail'):
        """
        Plots the distribution of the target variable with color intensity 
        representing the frequency (darker = more frequent).

        Parameters:
        - df (pd.DataFrame): The input dataframe containing a boolean 'fail' column and at least one categorical column.
        - category_column (str): The name of the categorical column to group by.

        Returns:
        - Plot
        """
        offer_counts = df[target].value_counts()
        sorted_counts = offer_counts.sort_values(ascending=True)
        colors = sns.color_palette("magma", len(sorted_counts))
        color_map = {label: color for label, color in zip(sorted_counts.index, colors)}

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=offer_counts.index,
            y=offer_counts.values,
            palette=[color_map[x] for x in offer_counts.index]
        )

        for p in plt.gca().patches:
            plt.gca().annotate(
                format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points'
            )

        plt.title('Distribution of Fail vs. Not Fail', fontsize=16)
        plt.xlabel('Failure Status', fontsize=12)
        plt.ylabel('Quantity', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_failure_rate_by_category(self,df, category_column: str):
        """
        Plots the failure rate as a percentage for each category in the specified column.

        This function is useful for analyzing categorical variables and their relationship
        with the failure target. It calculates the percentage of failure (`fail == True`)
        within each category and displays the result as a bar plot.

        Parameters:
        - df (pd.DataFrame): The input dataframe containing a boolean 'fail' column and at least one categorical column.
        - category_column (str): The name of the categorical column to group by.

        Returns:
        - Plot
        """
        df['fail'] = df['fail'].astype(int)

        plot_data = df.groupby(category_column)['fail'].mean().reset_index()
        plot_data['Failure Rate (%)'] = plot_data['fail'] * 100

        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        ax = sns.barplot(data=plot_data, x=category_column, y='Failure Rate (%)', palette='magma')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)

        plt.title(f'Failure Rate by {category_column.replace("_", " ").title()}', fontsize=14)
        plt.xlabel(category_column.replace("_", " ").title(), fontsize=12)
        plt.ylabel('Failure Rate (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    def plot_column_distribution(self,df, column:str):
        """
        Plots the distribution of a specified column in the dataset.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data to be plotted.
        column : str
            The name of the column for which the distribution will be plotted.

        Returns:
        -------
        None
            Displays a bar plot showing the distribution of the specified column.
        """
        df = df[column].value_counts().reset_index()
        df.columns = [column, 'count']

        sorted_counts = df.sort_values('count',ascending=True)
        colors = sns.color_palette("magma", len(sorted_counts))
        color_map = {label: color for label, color in zip(sorted_counts.index, colors)}

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x=column, y='count', palette=[color_map[x] for x in df.index])

        plt.title(f'{column} Distribution')
        plt.xlabel(f'{column}')
        plt.ylabel('Number of times used')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    def get_top_preset_combinations_before_failure(self, df):
        """
        Identifies and prints the most common combinations of preset_1 and preset_2 
        that appear immediately before a failure event.

        This function filters for rows where a failure is predicted to occur in the next cycle 
        (`fail_next == 1`) but has not yet occurred (`fail == 0`). It then counts the frequency 
        of each (preset_1, preset_2) combination and reports their relative frequencies as percentages.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the columns 'fail', 'fail_next', 'preset_1', and 'preset_2'.

        Returns:
        -  Prints the top preset combinations by percentage.
        """
        presets_before_failure = df[(df['fail_next'] == True) & (df['fail'] == 0)]['preset_combined']

        preset_percent = presets_before_failure.value_counts(normalize=True).reset_index(name='percentage')

        preset_percent['percentage'] = (preset_percent['percentage'] * 100).round(2)

        print("Most common preset combinations before a failure (in %):")
        print(preset_percent.head())



    def evaluate_preset_transition_failure_impact(self, df, preset_combined_seq_1, preset_combined_seq_2):
        """
        Evaluates how often a specific transition between two preset combinations is followed by a failure.

        This function checks how many times a given sequence of preset transitions 
        (from `preset_seq_1` to `preset_seq_2`) immediately precedes a failure event.
        It then calculates and prints the absolute count and percentage relative to all failure events.

        Parameters:
        - df (pd.DataFrame): DataFrame that includes the 'preset_combined' and 'fail' columns.
        - preset_seq_1 (str): The first preset combination in the sequence.
        - preset_seq_2 (str): The second preset combination that follows the first.
        
        Returns:
        - None: Prints the number and percentage of failures preceded by the specified sequence.
        """
        df['combo_lag1'] = df['preset_combined'].shift(1)
        mask = (
            (df['combo_lag1'] == preset_combined_seq_1) & 
            (df['preset_combined'] == preset_combined_seq_2) & 
            (df['fail'] == 1)
        )
        count = mask.sum()
        total_fails = df['fail'].sum()
        percentage = round(100 * count / total_fails, 2)

        print(f"Sequence {preset_combined_seq_1} → {preset_combined_seq_2} → failure occurred {count} times.")
        print(f"This represents {percentage}% of all failure events.")


    def plot_correlation(self,df):
        """
        Plots a correlation heatmap for the given DataFrame.

        This method computes the correlation matrix for the numeric columns
        in the provided DataFrame and visualizes it as a heatmap using Seaborn.

        Parameters:
        df : pandas.DataFrame
            The input DataFrame containing the data for which the correlation
            heatmap will be generated. Only numeric columns are considered.

        Returns:
        - Plot
        """
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.set(style='white')

        heatmap = sns.heatmap(
            corr,
            annot=True,      
            fmt=".2f",         
            cmap='magma',  
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8}
                            )

        plt.title('Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_box_plots_by_failure(self,df,column,fail_col = 'fail'):
        """
        Plots box plots for a specified column grouped by failure status.

        This function creates a box plot to visualize the distribution of values
        in a specified column, grouped by the failure status column. It uses the
        seaborn library for plotting.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to be plotted.
            column (str): The name of the column to plot on the y-axis.
            fail_col (str, optional): The name of the column indicating failure status.
                Defaults to 'fail'.

        Returns:
        - Plot
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=fail_col, y=column, data=df, palette='magma')
        plt.title(f'{column} by Fail Stauts')
        plt.xlabel('Fail')
        plt.ylabel(column.capitalize())
        plt.show()

    def calculate_overall_and_before_variable_mean(self,df,column):
        """
        Calculate and display the mean of a specified column for the entire dataset 
        and for the subset of data where 'fail_next' equals 1.

        Returns:
            df (pd.DataFrame): The input DataFrame containing the data.
            column (str): The name of the column for which the mean is to be calculated.
        """
        before_failure = df[df['fail_next'] == True]
        overall_mean = df[column].mean()
        before_fail_mean = before_failure[column].mean()

        summary = pd.DataFrame({
            'Mean': ['Overall', 'Before Fail'],
            column: [overall_mean, before_fail_mean]
        })

        print(f"\n Variable: {column}")
        print(summary)

    def plot_variables_time_series_with_failures(self, df, variable: str):
        """
        Plots the time series of a specified variable and highlights failure events.

        This function creates a line plot of the specified variable over time and marks
        the points where failures occurred. It also includes a horizontal line indicating
        the mean value of the variable.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data to be plotted. Must include the specified
            variable and a 'fail' column indicating failure events.
        variable : str
            The name of the variable to be plotted on the y-axis.

        Returns:
        -------
        None
            Displays a time series plot with failure events highlighted.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df[variable], label=variable, color= '#febb81')
        failures = df[df['fail'] == 1]
        plt.scatter(failures.index, failures[variable], color='#d3436e', label='Fail', zorder=5)
        mean_value = df[variable].mean()
        plt.axhline(mean_value, color='#982d80', linestyle='--', label=f'Mean: {mean_value:.2f}')
        plt.title(f'{variable} Over Time with Failure Events', fontsize=14)
        plt.xlabel('Index', fontsize=12)
        plt.ylabel(variable, fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # def test_t_for_average(self,df,column):
    #     from scipy.stats import ttest_ind

    #     '''Calculate the T-test for the means of *two independent* samples of scores.

    #     This is a test for the null hypothesis that 2 independent samples
    #     have identical average (expected) values. This test assumes that the
    #     populations have identical variances by default.'''
    #     print(ttest_ind(df[df.fail == 1][column], df[df.fail == 0][column]))

    def find_thresholds(self,df, var, n_splits=20):
        """
            Analyzes how failure rate changes across different thresholds of a numeric variable.

            Parameters:
                df (pd.DataFrame): Input dataframe containing a numeric variable and binary 'fail' column.
                var (str): Name of the numeric variable to analyze.
                n_splits (int): Number of equally spaced threshold splits to evaluate.

            Returns:
                pd.DataFrame: A DataFrame containing threshold values, failure rates above and below each threshold,
                            the difference in failure rates, and the sample sizes (counts) in each group.
        """
        thresholds = np.linspace(df[var].min(), df[var].max(), n_splits)
        results = []

        for t in thresholds:
            above = df[df[var] >= t]
            below = df[df[var] < t]
            fail_rate_above = above['fail'].mean()
            fail_rate_below = below['fail'].mean()
            results.append({
                'threshold': t,
                'fail_rate_above': fail_rate_above,
                'fail_rate_below': fail_rate_below,
                'delta_fail_rate': fail_rate_above - fail_rate_below,
                'count_above': len(above),
            })

        return pd.DataFrame(results)
    
    def average_failure_duration_cycles(self, df):
        """
        Calculates the average duration (in cycles) that the equipment remains in a continuous failure state.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing a boolean 'fail_continued' column indicating 
                            whether the failure state is ongoing at each cycle.

        Returns:
            float: The average number of cycles per continuous failure period.
        """
        df_copy = df.copy()
        df_copy['fail_continued'] = df_copy['fail_continued'].astype(bool)
        df_copy['fail_group'] = (df_copy['fail_continued'] != df_copy['fail_continued'].shift()).cumsum()

        fail_durations = df_copy[df_copy['fail_continued']].groupby('fail_group').size()
        average_duration = fail_durations.mean()

        print(f"Average duration of continuous failure: {average_duration:.2f} cycles")