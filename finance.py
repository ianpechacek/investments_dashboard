#######################
# Import libraries
import streamlit as st
import yfinance as yf

import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
# For plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image
import base64
import io
import os

MARGIN = dict(l=0,r=10,b=10,t=25)

#######################
# Page configuration
st.set_page_config(
    page_title="Investments Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#############################

ticker_list = ['TSLA', 'META', 'GOOGL', 'NVDA']
period_list = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
interval_list = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

#######################
# Sidebar
with st.sidebar:
    st.title('📈 Investments')
    
    selected_ticker = st.selectbox('Select ticker', ticker_list)
    selected_period = st.selectbox('Select period', period_list, index=4)

    if selected_period in ['1d']:
        interval_list = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
    elif selected_period in ['5d']:
        interval_list = ['2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d']
    elif selected_period in ['1mo']:
        interval_list = ['2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk']
    elif selected_period in ['3mo']:
        interval_list = ['1d', '5d', '1wk', '1mo']
    else: 
        interval_list = ['1d', '5d', '1wk', '1mo', '3mo']

    selected_interval = st.selectbox('Select interval', interval_list, index=0)
    # st.write("Selected interval", selected_interval)

    data1 = yf.download(selected_ticker, period=selected_period, interval=selected_interval) 
    df = pd.DataFrame(data1['Adj Close'])
    df2 = pd.DataFrame(data1)
    # data1 = data1.reset_index()
    yf_data = yf.Ticker(selected_ticker)
    st.session_state.yf_data = yf_data.info
   

#######################
# FUNCTIONS
#######################

# Load the data
def load_data():
    df = pd.read_csv("final.csv")
    return df

data = load_data().copy()
st.session_state.cap_data = data

def image_to_base64(img_path, output_size=(64, 64)):
    # Check if the image path exists
    if os.path.exists(img_path):
        with Image.open(img_path) as img:
            img = img.resize(output_size)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    return ""

def header_logo(ticker):
    output_dir = 'downloaded_logos'
    logo_dict = {
        'META': 'Meta Platforms (Facebook).png',
        'NVDA': 'NVIDIA.png',
        'TSLA': 'Tesla.png',
        'GOOGL': 'Alphabet (Google).png'
    }
    logo_name = logo_dict.get(ticker)  # Safely get the name in case the ticker is not in the dictionary
    if logo_name:
        path = os.path.join(output_dir, f'{logo_name}')
        # return image_to_base64(path, output_size=(64, 64))
        return path
    else:
        print(f"Logo for ticker '{ticker}' not found.")


# Plots

# Simple linechart
def simple_line_chart(df, tickers):
    data = np.round(df[tickers],2)
    st.line_chart(data, y_label='US Dollars')

def simple_bar_chart(df, tickers):
    # colors = ['#B03A2E' if row['Open'][tickers] - row['Close'][tickers] >= 0 
        # else '#27AE60' for index, row in df.iterrows()]
    data = df['Volume'][tickers] *10e-9
    st.bar_chart(data, y_label='Volume (Billions)')

def signum_sign(x):
    if x >= 0:
        return "+"
    else:
        return ""
    
def metric_close(df, tickers):
    data = np.round(df[tickers][-1])
    st.metric("", '$ '+str(data))

    

def metric_change(df, tickers):
    data = np.round(df[tickers],2)
    difference = np.round(data[-1] - data[0],2)
    delta = str(round(100*difference / data[0]))+'%'  
    st.metric(" ", signum_sign(difference)+' '+ str(difference) + '$', delta=delta)

########################
# LOGO
#######################

# If 'Logo' column doesn't exist, create one with path to the logos
if 'Logo' not in data.columns:
    output_dir = 'downloaded_logos'
    data['Logo'] = data['Name'].apply(lambda name: os.path.join(output_dir, f'{name}.png'))

# Convert image paths to Base64
data["Logo"] = data["Logo"].apply(image_to_base64)
data["Market Cap"] = data["Market Cap"].str.replace(' ', '', regex=False)  # Remove spaces
data["Market Cap"] = pd.to_numeric(data["Market Cap"], errors='coerce').astype('float64')  # Convert to integer

image_column = st.column_config.ImageColumn(label="")
nazev_column = st.column_config.TextColumn(label="Company")
market_cap_column = st.column_config.NumberColumn("Market Cap ($B)", format="$%dB")

# Adjust the index to start from 1 and display only the first 25 companies
data.reset_index(drop=True, inplace=True)
data = data.head(25)
data.index = data.index + 1

data = data[['Logo', 'Name', 'Market Cap']]

#########################################################
# Dashboard Main Panel


st.session_state.page_subheader = '{0} ({1})'.format(yf_data.info['shortName'], yf_data.info['symbol'])

st.image(header_logo(selected_ticker))
st.subheader(st.session_state.page_subheader)



metric_change(df, selected_ticker)



col = st.columns((1,1, 4.5, 2.8), gap='medium')

with col[0]:
    
    st.text('Current price')
    st.text(yf_data.info['currentPrice'])
    st.divider()

    st.text('Open')
    st.text(yf_data.info['open'])
    st.divider()

    st.text('previousClose')
    st.text(yf_data.info["previousClose"])
    st.divider()

    st.text('Recommendation')
    st.text(yf_data.info["recommendationKey"].upper())
    st.divider()

    st.text('Last Split Factor')
    st.text(yf_data.info.get("lastSplitFactor", "n/a"))
    # st.divider()


with col[1]:
    
    st.text('Day Low')
    st.text(yf_data.info['dayLow'])
    st.divider()

    st.text('Day High')
    st.text(yf_data.info['dayHigh'])
    st.divider()

    st.text('50 Day Average')
    st.text(yf_data.info["fiftyDayAverage"])
    st.divider()

    st.text('52 Week Change')
    # st.text('{0:.2f} x {1}'.format(yf_data.info['ask'], yf_data.info['askSize']))
    if isinstance(yf_data.info.get("52WeekChange", "n/a"), (int, float)):
        st.text('{:.2f}'.format(yf_data.info.get("52WeekChange", "n/a")))
    else:
        st.text("n/a")
    st.divider()

    st.text('ROI')
    # st.text('{0:.2f} x {1}'.format(yf_data.info['ask'], yf_data.info['askSize']))
    st.text('{:.2f}'.format(yf_data.info["returnOnEquity"]))
    # st.divider()

with col[2]:
    st.markdown('#### Market')
    # Simple linechart
    simple_line_chart(df, selected_ticker)
    
    simple_bar_chart(df2, selected_ticker)

    # Adds the volume as a bar chart
    # layout = go.Layout(title='Price, MA and Volume', height=500, margin=MARGIN, marker_color=colors)
    # fig.update_layout(layout)
    # st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    

with col[3]:
    st.markdown('#### Top Companies')

    st.dataframe(data, height=600, column_config={"Logo": image_column,"Name":nazev_column, "Market Cap":market_cap_column})
    
