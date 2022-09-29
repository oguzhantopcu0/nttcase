
import architecture
from data_location import data_core, data_core2, x_in3, x_in2, x_in, df_4, expect

model2 = architecture.weather(data=data_core, prediction_data=x_in, epoch=1000,
                              batch_size=50, steps=30, date_scala='day', expectetion=expect)

model2.tensorize()
model2.model_weather()
history = model2.fitter1()
architecture.plotter(history, metric='root_mean_squared_error', val_metric='val_root_mean_squared_error')
model2.predicter()
