import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



image = Image.open('Pennstate logo.png')
st.image(image, use_column_width = True)


# Set title
st.title('ERP Simulation Dashboard')
file = st.sidebar.file_uploader("Please upload the sales order report")



if file is not None:
    st.sidebar.success('File uploaded Succesfully')
    df = pd.read_excel(file)
    df['Profit'] = df['Value'] - df['Cost']
    def dem_product(data_f):
        chart = px.histogram(data_f, x = 'Material description', y = 'Qty', color = 'Area', 
                             barmode = 'group', facet_row = 'Round', height = 600)
        chart.update_xaxes(None)
        chart.update_yaxes(title = 'Total Demand')
        
        return chart
    
    def profit_product(data_f):
        st.subheader('Profit by Region')
        if len(data_f['Round'].unique()) == 1:
            chart_2 = go.Figure(data=[go.Pie(labels=data_f['Area'], values=df['Profit'], hole=.6)])
            chart_2.update_layout(annotations=[dict(text='Round 1', x=0.5, y=0.5, font_size=20, showarrow=False)])
        elif len(data_f['Round'].unique()) == 2:
            df_1 = data_f.loc[data_f['Round'] == 1]
            df_2 = data_f.loc[data_f['Round'] == 2]
            chart_2 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
            chart_2.add_trace(go.Pie(labels=df_1['Area'], values=df_1['Profit']),
              1, 1)
            chart_2.add_trace(go.Pie(labels=df_2['Area'], values=df_2['Profit']),
              1, 2)
            chart_2.update_traces(hole=.6, hoverinfo="label+percent")

            chart_2.update_layout(annotations=[dict(text='Round 1', x=0.1, y=0.5, font_size=20, showarrow=False),
                                               dict(text='Round 2', x=0.3, y=0.5, font_size=20, showarrow=False)])
        else:
            df_1 = data_f.loc[data_f['Round'] == 1]
            df_2 = data_f.loc[data_f['Round'] == 2]
            df_3 = data_f.loc[data_f['Round'] == 3]
            chart_2 = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])
            chart_2.add_trace(go.Pie(labels=df_1['Area'], values=df_1['Profit']),1, 1)
            chart_2.add_trace(go.Pie(labels=df_2['Area'], values=df_2['Profit']),1, 2)
            chart_2.add_trace(go.Pie(labels=df_3['Area'], values=df_3['Profit']),1, 3)

            chart_2.update_traces(hole=.6, hoverinfo="label+percent")
            chart_2.update_layout(annotations=[dict(text='Round 1', x=0.08, y=0.5, font_size=20, showarrow=False),
                                               dict(text='Round 2', x=0.5, y=0.5, font_size=20, showarrow=False),
                                               dict(text='Round 3', x=0.9, y=0.5, font_size=20, showarrow=False)])


        return chart_2
    
    

    
    re_ord = st.sidebar.radio("Did you reorder?", ("Yes", "No"), index = 1)
    
    if re_ord == "Yes":
        st.sidebar.subheader('Reorder Quantity Information')
        ml_5_spritz = st.sidebar.number_input('Quantity of 500mL Spritz reordered', min_value=1, step=1)
        ml_5_lemspritz = st.sidebar.number_input('Quantity of 500mL Lemon Spritz reordered', min_value=1, step=1)
        ml_5_pure = st.sidebar.number_input('Quantity of 500mL ClearPure reordered', min_value=1, step=1)
        l_1_spritz = st.sidebar.number_input('Quantity of 1L Spritz reordered', min_value=1, step=1)
        l_1_lemspritz = st.sidebar.number_input('Quantity of 1L Lemon Spritz reordered', min_value=1, step=1)
        l_1_pure = st.sidebar.number_input('Quantity of 1L ClearPure reordered', min_value=1, step=1)
        
        st.sidebar.subheader('Delivery Day')
        
        day_5_spritz = st.sidebar.number_input('Scheduled Delivery of 500 mL Spritz', min_value = 1, max_value = 20, step = 1)
        day_5_lemspritz = st.sidebar.number_input('Scheduled Delivery of 500 mL Lemon Spritz', min_value = 1, max_value = 20, step = 1)
        day_5_pure = st.sidebar.number_input('Scheduled Delivery of 500 mL ClearPure', min_value = 1, max_value = 20, step = 1)
        day_1_spritz = st.sidebar.number_input('Scheduled Delivery of 1L Spritz', min_value = 1, max_value = 20, step = 1)
        day_1_lemspritz = st.sidebar.number_input('Scheduled Delivery of 1L Lemon Spritz', min_value = 1, max_value = 20, step = 1)
        day_1_pure = st.sidebar.number_input('Scheduled Delivery of 1L ClearPure', min_value = 1, max_value = 20, step = 1)
       
        new_data.loc[day_5_spritz, '500mL Spritz'] = new_data.loc[day_5_spritz, '500mL Spritz'] + (ml_5_spritz*24)
        new_data.loc[day_5_lemspritz, '500mL Lemon Spritz'] = new_data.loc[day_5_lemspritz, '500mL Lemon Spritz'] + (ml_5_lemspritz*24)
        new_data.loc[day_5_pure, '500mL ClearPure'] = new_data.loc[day_5_pure, '500mL ClearPure'] + (ml_5_pure*24)
        new_data.loc[day_1_spritz, '1L Spritz'] = new_data.loc[day_1_spritz, '1L Spritz'] + (l_1_spritz*12)
        new_data.loc[day_1_lemspritz, '1L Lemon Spritz'] = new_data.loc[day_1_lemspritz, '1L Lemon Spritz'] + (l_1_lemspritz*12)
        new_data.loc[day_1_pure, '1L ClearPure'] = new_data.loc[day_1_pure, '1L ClearPure'] + (l_1_pure*12)
        
        st.line_chart(new_data)
        st.plotly_chart(dem_product(df))
        st.plotly_chart(profit_product(df))
        
        
        
    else:
        st.plotly_chart(dem_product(df))
        st.plotly_chart(profit_product(df))
        

    
    
