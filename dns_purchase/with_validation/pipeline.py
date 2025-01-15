import pandas as pd

from get_data import get_sales, get_count, get_full_data
from version_Transformers import run_pipeline

# data = get_data()
data = pd.read_csv('data_month.csv')

freq = 'MS'
predict_df = pd.DataFrame(
    columns=[f'{freq}_date', 'category_id', 'final_count_forecast', 'final_cost_forecast', 'date_load']
)

for i in data.category_id.unique():
    print(f'Обрабатывается набор категория {i}')
    df = data.copy()
    df = df.where((df.category_id == i)).dropna()

    data_count = get_full_data(get_count(df), freq)
    print(data_count.shape)
    data_cost =  get_full_data(get_sales(df), freq)
    print(data_cost.shape)

    print('Моделируется count')
    if len(data_count.date) > 50:
        print(data_count.head())
        model, forecast_df = run_pipeline(
            data_count,
            freq=freq)
    else:
        print(f'Недостаточно данных для набора категория {i}')


    print('Моделируется cost')
    if len(data_cost.date) > 20:
        print(data_cost.head())
        model_2, forecast_df_cost = run_pipeline(data_cost, freq)
    else:
        print(f'Недостаточно данных для набора категория {i}')
    count_df = {
        f'{freq}_date': forecast_df['Date'],
        'category_id': i,
        'final_count_forecast': forecast_df['Predicted'],
        'final_cost_forecast': forecast_df_cost['Predicted'],
        'date_load': pd.Timestamp.now().strftime("%Y-%m-%d")
    }
    count_df = pd.DataFrame(count_df)

    predict_df = pd.concat([predict_df, count_df])





# predict_df.to_csv('data_month_dep.csv', index=False)
#
# pg_conn.insert_df(
#     df =
# )