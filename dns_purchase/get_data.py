import pandas as pd


# Функция, возвращающая выборку по недельным стоимостям без НДС
def get_sales(df):
    data = df.copy()

    col_to_drop = ['Unnamed: 0', 'category_id', 'category_name', 'responsible_team_id', 'delivery_channel_id']
    data.drop(columns=[col for col in col_to_drop if col in data.columns], inplace=True)

    data['product_count'] = data['product_cost']
    data = data.drop(['product_cost'], axis=1, inplace=False)
    return data

# Функция, возвращающая выборку по недельным количествам товаров
def get_count(df):
    data = df.copy()

    col_to_drop = ['Unnamed: 0', 'category_id', 'category_name', 'responsible_team_id', 'delivery_channel_id']
    data.drop(columns=[col for col in col_to_drop if col in data.columns], inplace=True)

    data = data.drop(['product_cost'], axis=1, inplace=False)
    return data

# Функция, заполняющая пропущенные даты
def get_full_data(data):
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    start_date = df["date"].min()
    end_date = df["date"].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='MS')

    full_df = pd.DataFrame({"date": all_dates})
    full_df = full_df.merge(df, on="date", how="left")
    full_df["product_count"] = full_df["product_count"].fillna(0).astype(int)
    data_full = pd.DataFrame(full_df)
    return data_full