import pandas as pd
pd.set_option("display.max_columns", None)

from get_data import get_sales, get_count, get_full_data
from LSTM_model import run_pipeline


data = pd.read_csv('C:/Users/Rusinov.DS/PycharmProjects/ML_project/dns_purchase/dataset/data_month.csv')
predict_df = pd.DataFrame(
    columns=['month_date', 'category_id', 'final_count_forecast', 'final_cost_forecast', 'date_load']
)



for i in data.category_id.unique():
    print(f'⚙️Обрабатывается категория {i}')
    df = data.copy()
    df = df.where((df.category_id == i)).dropna()

    data_count = get_full_data(get_count(df))
    data_cost =  get_full_data(get_sales(df))

    if len(data_count.date) > 20:
        model, forecast_df = run_pipeline(data_count, category=i)
        print(f"    ✅Прогноз количества построен для категории {i}")
    else:
        print(f'    ❌Недостаточно данных количества для категории {i}')

    if len(data_cost.date) > 20:
        model_2, forecast_df_cost = run_pipeline(data_cost, category=i)
        print(f"    ✅Прогноз затрат построен для категории {i}")
    else:
        print(f'    ❌Недостаточно данных затрат для категории {i}')
    count_df = {
        'month_date': forecast_df['Date'],
        'category_id': i,
        'final_count_forecast': forecast_df['Predicted'],
        'final_cost_forecast': forecast_df_cost['Predicted'],
        'date_load': pd.Timestamp.now().strftime("%Y-%m-%d")
    }
    count_df = pd.DataFrame(count_df)
    print(count_df.head())

    predict_df = pd.concat([predict_df, count_df])
    print(f"✅Прогноз построен для категории {i}")


predict_df.to_csv('lstm_predict.csv', index=False)