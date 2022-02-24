import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt

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
        selection = alt.selection_multi(fields=['Material description'], bind='legend')
        chart = alt.Chart(data_f).mark_bar().encode(x = alt.X('Material description:N', title = 'Products'), 
                                y = alt.Y('sum(Qty):Q', title = 'Demand'), column = 'Area:N', 
                                color = 'Material description:N', tooltip = ['sum(Qty):Q'],
                               opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(
        selection).properties(width = 200, height = 200)
        return chart

    def profit_product(data_f):
        selection = alt.selection_multi(fields=['Material description'], bind='legend')
        st.subheader('Profit by Region')
        chart_2 = alt.Chart(data_f).mark_line().encode(x = 'Day:O', 
                                 y = alt.Y('sum(Profit):Q', title = 'Profit by Product'), column = 'Area:N',
                                 color = 'Material description:N', tooltip = ['sum(Profit):Q'], 
                                 opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(
                                selection).properties(width = 200, height = 200)
        return chart_2
    
    def round_first(re_ord, data_f):
        sort_data_f = data_f.sort_values(by='Day')
        spritz_5 = [1000]
        lemspritz_5 = [1000]
        pure_5 = [1000]
        spritz_1 = [1000]
        lemspritz_1 = [1000]
        pure_1 = [1000]
        j = 0 
        for i in sort_data_f.iterrows():
            if (i[1]['Material description'] == '500mL Spritz') and (i[1]['Day'] == 1):
                spritz_5.append(spritz_5[j] - i[1]['Qty'])
                j += 1
            else:
                spritz_5.append(spritz_5[j] - i[1]['Qty'])
                j+= 1
                               
        j = 0
        for i in sort_data_f.iterrows():
            if (i[1]['Material description'] == '500mL Lemon Spritz') and (i[1]['Day'] == 1):
                lemspritz_5.append(lemspritz_5[j] - i[1]['Qty'])
                j += 1
            else:
                lemspritz_5.append(lemspritz_5[j] - i[1]['Qty'])
                j += 1
        j = 0
        for i in sort_data_f.iterrows():
            if (i[1]['Material description'] == '500mL ClearPure') and (i[1]['Day'] == 1):
                pure_5.append(pure_5[j] - i[1]['Qty'])
                j += 1
            else:
                pure_5.append(pure_5[j] - i[1]['Qty'])
                j+=1
                
        j = 0
        for i in sort_data_f.iterrows():
            if (i[1]['Material description'] == '1L Spritz') and (i[1]['Day'] == 1):
                spritz_1.append(spritz_1[j] - i[1]['Qty'])
                j += 1
            else:
                spritz_1.append(spritz_1[j] - i[1]['Qty'])
                j+=1
                
       j = 0
        for i in sort_data_f.iterrows():
            if (i[1]['Material description'] == '1L Lemon Spritz') and (i[1]['Day'] == 1):
                lemspritz_1.append(lemspritz_1[j] - i[1]['Qty'])
                j += 1
            else:
                lemspritz_1.append(lemspritz_1[j] - i[1]['Qty'])
                j+=1         
                
       j = 0
        for i in sort_data_f.iterrows():
            if (i[1]['Material description'] == '1L ClearPure') and (i[1]['Day'] == 1):
                pure_1.append(pure_1[j] - i[1]['Qty'])
                j += 1
            else:
                pure_1.append(pure_1[j] - i[1]['Qty'])
                j+=1        
                
        
        if re_ord == 'Yes':
            
        else:
            chart_data = pd.dataframe([spritz_5, lemspritz_5, pure_5, spritz_1, lemspritz_1, pure_1]).transpose()
            chart_data.columns = ['500mL Spritz', '500mL Lemon Spritz', '500mL ClearPure', '1L Spritz', '1L Lemon Spritz', '1L ClearPure']
            
            
        return chart_data
    
    re_ord = st.sidebar.radio("Did you reorder?", ("Yes", "No"))
    
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
        
    else:
        st.line_chart(round_first('No', df))
        st.altair_chart(dem_product(df), use_container_width = False)
        st.altair_chart(profit_product(df), use_container_width = False)
        

    
    
