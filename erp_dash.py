import openpyxl
import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt

image = Image.open('Pennstate logo.png')
st.image(image, use_column_width = True)

# Set title
st.title('ERP Simulation Dashboard')

file = st.sidebar.file_uploader("Please choose a file to upload")

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

    st.altair_chart(dem_product(df), use_container_width = False)
    st.altair_chart(profit_product(df), use_container_width = False)
