import os

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


class LoadData:
    cwd = os.getcwd()

    def load_data(self, ticker, n_days):
        file_to_load = os.path.join(self.cwd, "History", 'stock_market_data-%s.csv' % ticker)
        if not os.path.exists(file_to_load):
            print("This ticker is not supported")
            return
        else:
            df = pd.read_csv(file_to_load, index_col=False)
            df['Date'] = df['Date'].astype('datetime64[ns]')
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            df = df.sort_values('Date')

            '''high_prices = df.loc[:, 'High'].to_numpy()
            low_prices = df.loc[:, 'Low'].to_numpy()
            mid_prices = (high_prices + low_prices) / 2.0'''

            scaler = MinMaxScaler(feature_range=(0, 1))
            model = self.model_load(self, ticker)

            close_days = df.filter(['Date'])
            last_day = close_days[-1:].values[0][0]
            close_prices = df.filter(['Close'])
            pred_days = []
            for days in range(n_days):
                x_input = self.get_input(self, close_prices, scaler)
                #print(x_input)
                pred_price = self.predict_price(self, model, x_input, scaler)
                close_prices.loc[len(close_prices)] =pred_price
                close_days.loc[len(close_prices)] =str(last_day) + " + " + str(days + 1)
                close_days.reset_index()

            chart_y_input = close_prices[-n_days*2:].values
            chart_x_input_tem = close_days[-n_days*2:].values

            chart_x_input=[]
            for x in chart_x_input_tem:
                chart_x_input.append(str(x[0]))

            return chart_x_input, chart_y_input

    def get_input(self, close_prices, scaler):
        last_60_days = close_prices[-60:].values
        mid_prices = scaler.fit_transform(last_60_days)
        x_input = []
        x_input.append(mid_prices)
        return x_input

    '''def get_next_days(self, n_days, x_input, temp_input, ticker):
        lst_output = []
        n_steps = 60
        i = 0

        model = self.model_load(self, ticker)

        while (i < n_days):
            if len(temp_input) > 60:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape(1, n_steps, 1)
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i += 1
            else:
                x_input = x_input.reshape(1, n_steps, 1)
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i += 1

        return lst_output'''

    def model_load(self, ticker):
        model_to_load = os.path.join(self.cwd, "Models",'%s.h5' % ticker)
        print(model_to_load)
        model = load_model(model_to_load)
        return model

    def predict_price(self, model, x_input, scaler):
        x_input = np.array(x_input)
        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))
        pred_price = model.predict(x_input)
        pred_price = scaler.inverse_transform(pred_price)
        return [pred_price[0][0]]
