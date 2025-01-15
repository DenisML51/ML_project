from utils import ch_fcs
import pandas as pd

# Запрос на получение недельных данных по закупкам (количество товаров и сумма без НДС)
def get_data():
    df = ch_fcs().execute_to_df('''
        SELECT
            date_trunc('week', schet.`Период`) as date,
            product.category_4_id as category_id,
            product.category_4_name as category_name,
            product.responsible_team_id as responsible_team_id,
            product.delivery_channel_id as delivery_channel_id,
            sum(schet.`Количество`) as product_count,
            sum(schet.`СуммаБезНДС`) as product_sale
        FROM
            RN.Schet_45 as schet
        INNER JOIN
                dict.product as product ON product.id = schet.`Номенклатура`
        WHERE
            (equals(schet.`Регистратор_ТипСсылки`, '00000e05-0000-0000-0000-000000000000')
            OR equals(schet.`Регистратор_ТипСсылки`, '00000028-0000-0000-0000-000000000000'))
            AND greaterOrEquals(extract(year from schet.`Период`), 2018)
            AND notEquals(category_id,'00000000-0000-0000-0000-000000000000')
            AND equals(schet.`ВидДвижения`, 0)
            AND product.delivery_channel_id <> '00000000-0000-0000-0000-000000000000'
            AND product.responsible_team_id <> '00000000-0000-0000-0000-000000000000'
        GROUP BY
            date_trunc('week', schet.`Период`),
            product.category_4_id,
            product.category_4_name,
            product.responsible_team_id,
            product.delivery_channel_id;
        ''')
    return df


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