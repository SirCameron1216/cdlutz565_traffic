import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics


st.title('Traffic Volume Prediction')
st.image('traffic_image.gif')

# DT Pickle
dt_pickle = open('dt_traffic.pickle', 'rb') 
dt_model = pickle.load(dt_pickle) 
dt_pickle.close()

# RF Rickle
rf_pickle = open('rf_traffic.pickle','rb')
rf_model = pickle.load(rf_pickle)
rf_pickle.close()

#ADA Pickle
ada_pickle = open('ada_traffic.pickle', 'rb') 
ada_model = pickle.load(ada_pickle) 
ada_pickle.close()

# XG Pickle
xg_pickle = open('xg_traffic.pickle', 'rb') 
xg_model = pickle.load(xg_pickle) 
xg_pickle.close()


#default_df = pd.read_csv('Traffic_Volume.csv')
#default_df['weather_description'].replace(to_replace='Sky is Clear',value='sky is clear',inplace=True)
default_df = pd.read_csv('traffic_clean.csv')
default_df


with st.form('user_inputs'):
    
    holiday = st.selectbox('Holiday? If so, which one?',options=(default_df['holiday'].unique()))
    temp = st.number_input('Temperature in Kelvin?')
    rain_1h = st.number_input('How many millimeters of rain has fallen in the last hour?')
    snow_1h = st.number_input('How many millimeters of snow has falling in the last hour?')
    clouds_all = st.slider('What percent cloud coverage is there right now?',1,100)
    weather_main = st.selectbox('Which best describes the current weather conditions?', 
                                       options=(default_df['weather_main'].unique()))
    month = st.selectbox('What month is it?', options=(default_df['month'].unique()))
    weekday = st.selectbox('What day of the week is it? 0 = Monday', options=(default_df['weekday'].unique()))
    time = st.selectbox('What hour of the day is it? 0 = Midnight', options=(default_df['time'].unique()))
    ml_model = st.selectbox('Select Machine Learning Model for Prediction:', 
                            options=['Choose an option','Decision Tree','Random Forest','ADABoost','XGBoost'])
    st.form_submit_button()

# R2 and RMSE table needed
encode_df = default_df.copy()
encode_df = encode_df.drop(columns=['traffic_volume'])
encode_df.loc[len(encode_df)] = [holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,month,weekday,time]
cat_var = ['holiday','weather_main','month','weekday','time']
encode_dummy_df = pd.get_dummies(encode_df,columns=cat_var)
user_encoded_df = encode_dummy_df.tail(1)

table = pd.read_csv('ML_model_score.csv')


if ml_model == 'Decision Tree' :
        new_pred_dt = dt_model.predict(user_encoded_df)
        st.write('Decision Tree Prediction: {}'.format(*new_pred_dt))
        st.subheader('Prediction Performance')
        tab1, tab2 = st.tabs(['R2 & RMSE','Feature Importance'])
        with tab1:
            st.write(table)
        with tab2:
            st.image('dt_traffic_feature_imp.svg')


if ml_model == 'Random Forest':
        new_pred_rf = rf_model.predict(user_encoded_df)
        st.write('Random Forest Prediction: {}'.format(*new_pred_rf))
        st.subheader('Prediction Performance')
        tab1, tab2 = st.tabs(['R2 & RMSE','Feature Importance'])
        with tab1:
            st.write(table)
        with tab2:
            st.write('rf_feature_traffic_imp.svg failed to save. Could have retrieved but would be wasteful')

if ml_model == 'ADABoost':
        new_pred_ada = ada_model.predict(user_encoded_df)
        st.write('ADA Boost Prediction: {}'.format(*new_pred_ada))
        st.subheader('Prediction Performance')
        tab1, tab2 = st.tabs(['R2 & RMSE','Feature Importance'])
        with tab1:
            st.write(table)
        with tab2:
            st.image('ada_traffic_feature_imp.svg')

if ml_model == 'XGBoost':
        new_pred_xg = xg_model.predict(user_encoded_df)
        st.write('XG Boost Prediction: {}'.format(*new_pred_xg))
        st.subheader('Prediction Performance')
        tab1, tab2 = st.tabs(['R2 & RMSE','Feature Importance'])
        with tab1:
            st.write(table)
        with tab2:
            st.image('xg_traffic_feature_imp.svg')
else:
       st.write('Please submit form to generate prediction!')