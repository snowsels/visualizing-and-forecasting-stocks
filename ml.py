import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd
import  pandas_datareader as web
import datetime as dt
#from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense,Dropout,LSTM

"""
SOME POPULAR COMPANIES STOCKS 1.amazon-AMZN 2.Facebook-FB 3.Goldman sach 4.infosys-INFY 5.Apple-AAPL 6.Google-GOOGL 7.HP-HPQ 8.Dell-DELL
9.Acer-ACER 10.Adobe-ADBE 11.Microsoft-MSFT 12.Uber-UBER 13.Oracle-ORCL 14.Snapchat-SNAP 15.Twitter-TWTR 16.walmarts-WMT   """

 #load data
company='UBER'
start=dt.datetime(2013,1,1)
end=dt.datetime(2021,2,1)
data=web.DataReader(company,'yahoo',start,end)

 #prepare data

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data['Close'].values.reshape(-1,1))
prediction_days=60
x_train=[]
y_train=[]
for x in range(prediction_days,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])   #it gives the values (0,0),(1,0),(2,0)(3,0)....(60,0) at a 1 times and so on (1,0),(2,0)....(61,0)
    y_train.append(scaled_data[x,0])    #append values from (60,0) to (2019,0)



x_train=np.array(x_train)
y_train=np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))   #(1959,60,1)


 #build the model
model=Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(units=50))
model.add(Dropout(.2))
model.add(Dense(units=1))  #prediction of next closing value
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=25,batch_size=32)#Fit function adjusts weights according to data values so that better accuracy can be achieved.


'''Test the model accuracy on existing data'''

 #load test data
test_start=dt.datetime(2021,2,1)
test_end=dt.datetime.now()

test_data=web.DataReader(company,'yahoo',test_start,test_end)

actual_prices=test_data['Close'].values

total_dataset=pd.concat((data['Close'],test_data['Close']),axis=0)    #total_data

model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs=model_inputs.reshape(-1,1)
model_inputs=scaler.transform(model_inputs)


#make prediction on test data

x_test=[]
for x in range(prediction_days,len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_prices=model.predict(x_test)
predicted_prices=scaler.inverse_transform(predicted_prices)


 #predict next day



real_data=[model_inputs[len(model_inputs)-prediction_days:len(model_inputs+1),0]]
real_data=np.array(real_data)
real_data=np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))
prediction_next_day=model.predict(real_data)
prediction_next_day=scaler.inverse_transform(prediction_next_day)
print(f"Prediction of next day:{prediction_next_day}")



#plot the test predictions:::
#temp= make_interp_spline(actual_prices, predicted_prices)
#actual_prices_1=np.linspace(actual_prices.min(),actual_prices.max())
#predicted_prices_1=temp(actual_prices_1)




#plot the actual and prediction data

plt.plot(actual_prices,color="blue",label=f" Actaul {company} Price")
plt.plot(predicted_prices,color='tomato',label=f"Predicted {company} Price")
plt.title(f'{company} Share Price')
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.grid(color = 'green', linewidth = 0.5)
plt.show()