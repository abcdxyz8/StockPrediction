from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.tab import MDTabsBase
from kivy.uix.floatlayout import FloatLayout
from LoadData import LoadData

# import datetime as dt
# import urllib.request, json
# import os

# import tensorflow as tf # This code has been tested with TensorFlow 1.6




class Tab(MDBoxLayout, MDTabsBase):
    pass


class GraphLayout(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MainWidget(MDBoxLayout):
    row_data = []
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        lst = LoadData.load_data(LoadData, "QQQ", 7)

        dates = lst[0]
        values = lst[1]

        n = len(dates)

        for i in range(n):
            self.row_data.append((dates[i], "{:.2f}".format(values[i][0])))

        self.data_tables = MDDataTable(
            use_pagination=True,
            rows_num=20,
            column_data=[
                ("Date", dp(30)),
                ("Close Value", dp(30)),
            ],
            row_data=self.row_data
        )

        box = self.ids.box
        box.add_widget(self.data_tables)
        #box.height = dp(250)


class StockPredictionApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"  # "Gray"  #
        self.theme_cls.accent_palette = "Gray"
        # self.theme_cls.primary_hue = "900"
        # self.icon = "images/icon.ico"

        return MainWidget()


# Window.maximize()

if __name__ == '__main__':
    StockPredictionApp().run()
