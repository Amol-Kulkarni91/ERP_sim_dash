import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
        st.subheader('Demand by Region')
        chart = px.histogram(data_f, x ='Material description', y = 'Qty', color = 'Area', barmode = 'group', 
                             template = 'seaborn', facet_row = 'Round', height = 600)
        chart.update_xaxes(title=None)
        chart.update_yaxes(title='Total Demand')
        return chart

    def profit_product(data_f):
        data_f['Profit'] = data_f['Value'] - data_f['Cost']
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
            chart_2.update_traces(hole=.4, hoverinfo="label+percent")

            chart_2.update_layout(annotations=[dict(text='Round 1', x=0.18, y=0.5, font_size=20, showarrow=False),
                                               dict(text='Round 2', x=0.82, y=0.5, font_size=20, showarrow=False)])
        else:
            df_1 = data_f.loc[data_f['Round'] == 1]
            df_2 = data_f.loc[data_f['Round'] == 2]
            df_3 = data_f.loc[data_f['Round'] == 3]
            chart_2 = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])
            chart_2.add_trace(go.Pie(labels=df_1['Area'], values=df_1['Profit']),1, 1)
            chart_2.add_trace(go.Pie(labels=df_2['Area'], values=df_2['Profit']),1, 2)
            chart_2.add_trace(go.Pie(labels=df_3['Area'], values=df_3['Profit']),1, 3)

            chart_2.update_traces(hole=.4, hoverinfo="label+percent+name")
            chart_2.update_layout(annotations=[dict(text='Round 1', x=0.18, y=0.5, font_size=20, showarrow=False),
                                               dict(text='Round 2', x=0.82, y=0.5, font_size=20, showarrow=False),
                                               dict(text='Round 3', x=1.46, y=0.5, font_size=20, showarrow=False)])


        return chart_2
    
    def round_first(data_f):
        st.subheader('Inventory')
        sorted_df = data_f.sort_values(by='Day')
        sorted_df = sorted_df.groupby(['Material description', 'Day'])['Qty'].sum().reset_index()
        sp5 = 1000
        lsp5 = 1000
        p5 = 1000
        sp1 = 1000
        lsp1 = 1000
        p1 = 1000
        
        inventory = {'500mL Spritz': [1000]}
        inventory_1 = {'500mL Lemon Spritz' : [1000]}
        inventory_2 = {'500mL ClearPure' : [1000]}
        inventory_3 = {'1L Spritz' : [1000]} 
        inventory_4 = {'1L Lemon Spritz' : [1000]}
        inventory_5 = {'1L ClearPure': [1000]}
           
            
        
        for i in sorted_df.iterrows():
            if (i[1]['Material description'] == '500mL Spritz') and (i[1]['Day'] == 1):
                sp5 = sp5 - i[1]['Qty']
                inventory['500mL Spritz'].append(sp5)
            elif (i[1]['Material description'] == '500mL Spritz') and (i[1]['Day'] > 1):
                sp5 = sp5 -  i[1]['Qty']
                inventory['500mL Spritz'].append(sp5)
            elif (i[1]['Material description'] == '500mL Lemon Spritz') and (i[1]['Day'] == 1):
                lsp5 = lsp5 - i[1]['Qty']
                inventory_1['500mL Lemon Spritz'].append(lsp5)
            elif (i[1]['Material description'] == '500mL Lemon Spritz') and (i[1]['Day'] > 1):
                lsp5 = lsp5 - i[1]['Qty']
                inventory_1['500mL Lemon Spritz'].append(lsp5)
            elif (i[1]['Material description'] == '500mL ClearPure') and (i[1]['Day'] == 1):
                p5 = p5 - i[1]['Qty']
                inventory_2['500mL ClearPure'].append(p5)
            elif (i[1]['Material description'] == '500mL ClearPure') and (i[1]['Day'] > 1):
                p5 = p5 - i[1]['Qty']
                inventory_2['500mL ClearPure'].append(p5)
            elif (i[1]['Material description'] == '1L Spritz') and (i[1]['Day'] == 1):
                sp1 = sp1 - i[1]['Qty']
                inventory_3['1L Spritz'].append(sp1)
            elif (i[1]['Material description'] == '1L Spritz') and (i[1]['Day'] > 1):
                sp1 = sp1 - i[1]['Qty']
                inventory_3['1L Spritz'].append(sp1)
            elif (i[1]['Material description'] == '1L Lemon Spritz') and (i[1]['Day'] == 1):
                lsp1 = lsp1 - i[1]['Qty']
                inventory_4['1L Lemon Spritz'].append(lsp1)
            elif (i[1]['Material description'] == '1L Lemon Spritz') and (i[1]['Day'] > 1):
                lsp1 = lsp1 - i[1]['Qty']
                inventory_4['1L Lemon Spritz'].append(lsp1)
            elif (i[1]['Material description'] == '1L ClearPure') and (i[1]['Day'] == 1):
                p1 = p1 - i[1]['Qty']
                inventory_5['1L ClearPure'].append(p1)
            elif (i[1]['Material description'] == '1L ClearPure') and (i[1]['Day'] > 1):
                p1 = p1 - i[1]['Qty']
                inventory_5['1L ClearPure'].append(p1)

        inventory = pd.DataFrame.from_dict(inventory)
        inventory_1 = pd.DataFrame.from_dict(inventory_1)
        inventory_2 = pd.DataFrame.from_dict(inventory_2)
        inventory_3 = pd.DataFrame.from_dict(inventory_3)
        inventory_4 = pd.DataFrame.from_dict(inventory_4)
        inventory_5 = pd.DataFrame.from_dict(inventory_5)
        
        chart_data = pd.concat([inventory,inventory_1, inventory_2, inventory_3, inventory_4, inventory_5], axis = 1).fillna(0)
        chart_data[chart_data < 0] = 0  
        return chart_data
    
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
        
        new_data = round_first(df)
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
        st.line_chart(round_first(df))
        st.plotly_chart(dem_product(df))
        st.plotly_chart(profit_product(df))
        

    
    
