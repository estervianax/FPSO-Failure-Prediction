class DataPreprocessor(object):
    def __init__(self, df):
        self.df = df

    def transform(self):
        df_transformed = self.__lower_columns_names(self.df)
        df_transformed = self.__set_cycle_as_index(df_transformed)
        df_transformed = self.__add_fail_continued_column(df_transformed)
        df_transformed = self.__add_fail_next_colum(df_transformed)
        df_transformed = self.__add_preset_combined_colum(df_transformed)
        df_transformed, categorical_cols, numerical_cols = (
            self.__separate_columns_groups(df_transformed)
        )
        return df_transformed, categorical_cols, numerical_cols

    def create_time_based_columns(
        self, df, numerical_features_list, all_features_list, set_window=5
    ):
        df_times_based = self.__create_rolling_mean_columns(
            df, numerical_features_list, set_window
        )
        df_times_based = self.__create_lag1_columns(df_times_based, all_features_list)
        df_times_based = df_times_based.dropna().reset_index(drop=True)
        df_times_based, times_based_categorical_cols, times_based_numerical_cols = (
            self.__separate_columns_groups(df_times_based)
        )
        return df_times_based, times_based_categorical_cols, times_based_numerical_cols

    def __lower_columns_names(self, df):
        df.columns = map(str.lower, df.columns)
        return df

    def __set_cycle_as_index(self, df):
        df = df.set_index("cycle")
        return df

    def __add_fail_continued_column(self, df):
        df["fail_continued"] = df["fail"].shift(True, fill_value=False)
        return df

    def __add_fail_next_colum(self, df):
        df["fail_next"] = df["fail"].shift(-1)
        return df

    def __add_preset_combined_colum(self, df):
        df["preset_combined"] = (
            df["preset_1"].astype(str) + "_" + df["preset_2"].astype(str)
        )
        return df

    def __create_rolling_mean_columns(self, df, numerical_columns_list, set_window):
        for column in numerical_columns_list:
            df[f"{column}_rolling_{set_window}"] = (
                df[column].rolling(window=set_window).mean()
            )
        return df

    def __create_lag1_columns(self, df, columns_list):
        for column in columns_list:
            df[f"{column}_lag1"] = df[column].shift(1)
        return df

    def __separate_columns_groups(self, df):
        categorical_cols = []
        numerical_cols = []

        for col in df.columns:
            if col.startswith("preset"):
                df[col] = df[col].astype("category")
                categorical_cols.append(col)
            elif (
                col.startswith("temperature")
                or col.startswith("pressure")
                or col.startswith("frequency")
                or col.startswith("vibration")
            ):
                df[col] = df[col].astype("float32")
                numerical_cols.append(col)

        return df, categorical_cols, numerical_cols
