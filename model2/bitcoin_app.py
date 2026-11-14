import streamlit as st
import yfinance as yf
import joblib
import pandas as pd
import datetime

st.set_page_config(page_title="Bitcoin Price Predictor", page_icon="₿", layout="wide")
st.title("₿ Bitcoin Next-Day Price Predictor")
st.write("This website take todays data to predict tommorows closing price.")

# 1. Saved model load karo
model = joblib.load("model2/bitcoin_model.pkl")#ab usska naam badal diya model bas hoga ab

# 2. Internet se khud live data download karo

live_df = yf.download("BTC-USD", period="10d", interval="1d", multi_level_index=False)
#yesterdays data nikalenge only.

today=datetime.date.today()
yesterday=today-datetime.timedelta(days=1)#timedelta ek din ki date create kardeta taaki agar date format mein subtract hoga toh bas ek din subtract hosake
yesterday_data=yf.download("BTC-USD",start=yesterday,end=today,interval="1d",multi_level_index=False)#here start is included and end is excluded.
# Note : issue abhi bhi hai agar data upload hi na hua ho toh fir empty hi rehjayega or error aajayega na fir kya karoge

#3. Yesterday ka completed data uthao (.iloc[[-1]]) — exact wahi 4 columns aur wahi order!
feature_cols = ['High', 'Low', 'Open', 'Volume']
# yesterday_input = live_df[feature_cols].iloc[[-1]] actually mein inse data sahi se nahi aaraha tha matlab agar aaj hai toh kal ka data uploaded hai per agar aaj ka bhi upload kardiya toh aaj ka hi data use hojayega or ab kal ka prediction dega balki vo data abhi sahi hai hi nahi vo toh 12 am per correct upload hoga 
# yesterday_date = live_df.index[-2].strftime('%Y-%m-%d')# thats why ham command dekar only yesterdays data mangenge iloc ka issue hat jayega 
if not yesterday_data.empty:#agar empty nahi hoga toh yesterday ki value aajayegi
    yesterday_input = yesterday_data[feature_cols]
    yesterday_date = yesterday_data.index[0].strftime('%Y-%m-%d')
    last_close = float(yesterday_data['Close'].iloc[0])
else:#agar empty hua ho toh fir live_df wala 10 days ka data hai usske last element use hojayega
    st.warning("⚠️ Yesterday's data is not uploaded yet! Using the latest available data instead.")
    yesterday_input = live_df[feature_cols].iloc[[-1]]
    yesterday_date = live_df.index[-1].strftime('%Y-%m-%d')
    last_close = float(live_df['Close'].iloc[-1])
# Screen par graph aur automatic input table dikhao
st.subheader("📈 Pichle 10 Din ka Closing Price")
st.line_chart(live_df['Close'])

st.subheader(f"📊 Automatic Input Data (Completed Candle: {yesterday_date})")
st.dataframe(yesterday_input)

# 4. Button dabate hi prediction dikhao
if st.button("Predict Next Closing Price 🚀"):
    try :
        input_array = yesterday_input.values.astype(float)#array banadiya dataframe se
        predicted_price = float(model.predict(input_array)[0])
        diff = predicted_price - last_close
        col1, col2 = st.columns(2)
        col1.metric(label=f"Last Completed Close ({yesterday_date})", value=f"${last_close:,.2f}")
        col2.metric(label="Predicted Next Close", value=f"${predicted_price:,.2f}", delta=f"${diff:,.2f}")
    except Exception as e:
        # print("Yesterday's Data is not uploaded yet") NOTE: print statement se bas vs code ke terminal mein aajata hai website per nahi aata
        st.error(f"Prediction error : {e}")
#to run this website from wsl commandline open it in vs code first then paste below code
#source ~/bitcoin_env/bin/activate
#cd /mnt/c/Machinelearningfullpractise
#streamlit run model2/bitcoin_app.py