import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from datetime import date
from dateutil.relativedelta import relativedelta
import joblib

st.title("команда НЕайтишники")
st.header("краткосрочное потребление газа в регионах")

region = st.selectbox("Регион", range(1, 64))
longivity = st.selectbox("На сколько нужно сделать расчет", range(1, 8))
temperature = st.text_input("Температура", help="Введите температуру одной строкой, разделяя значения пробелами")
day_type = st.text_input("Тип дня", help="Введите тип дня одной строкой, разделяя значения пробелами (по умолчанию - 0)")

# Обработка данных
if st.button("Рассчитать"):
    
    reg = int(region) - 1
    long = int(longivity)

    if temperature != "":
        temp = list(map(float, temperature.split()))

        if day_type != "":
            days = list(map(int, day_type.split()))
        else:
            days = [0] * long

        data = np.array([list(i) for i in zip(temp, days)])

        model = joblib.load(f"sarima{reg}.joblib")


        forecast = model.get_prediction(1, long, exog=data)
        predicts = forecast.prediction_results.forecasts.flatten()
        confs = np.array(forecast.conf_int())

        st.write(f"Прогноз потребления на близжайшие {long} дней:")
        for i in range(1, long + 1):
            st.write(f"{date.today() + relativedelta(days=i)}: {round(predicts[i-1], 2)} кубометров газа, доверительный интервал: от {round(confs[i-1][0], 2)} до {round(confs[i-1][1], 2)}")

        dates = [date.today() + relativedelta(days=i) for i in range(1, len(predicts) + 1)]

        # Преобразование дат в формат matplotlib
        dates = [d.strftime('%Y-%m-%d') for d in dates]  # Преобразование в строку 

        # Построение графика
        fig, ax = plt.subplots()  # Создаем фигуру и ось
        ax.plot(dates, predicts)  # Построение графика
        ax.set_xlabel("Дата")  # Подпись оси X
        ax.set_ylabel("Потребление")  # Подпись оси Y
        ax.set_title("График потребления газа по датам")  # Заголовок

        for i in range(len(confs)):
            plt.fill_between(
                [i, i + 1],
                confs[i][0],
                confs[i][1],
                color='lightblue',
                alpha=0.5,
    )

        st.pyplot(fig)  # Отображение графика в Streamlit


    

    else:
        st.write("Убедитесь, что вы корректно ввели данные!")



        
