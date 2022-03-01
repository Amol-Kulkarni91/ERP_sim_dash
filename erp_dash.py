import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
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
		st.subheader("Demand by Region")
		chart = px.histogram(data_f, x = 'Material description', y = 'Qty', color = 'Area', 
				     barmode = 'group', facet_row = 'Round', height = 600, template = 'seaborn')
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

			chart_2.update_layout(annotations=[dict(text='Round 1', x=0.16, y=0.5, font_size=20, showarrow=False),
						       dict(text='Round 2', x=0.85, y=0.5, font_size=20, showarrow=False)])
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
						       dict(text='Round 3', x=0.93, y=0.5, font_size=20, showarrow=False)])


		return chart_2
	
	def update_data(data_f1, data_f2):
		
		temp_df1 = data_f2.loc[data_f2['Material description'] == '500mL Spritz']
		temp_df2 = data_f2.loc[data_f2['Material description'] == '500mL Lemon Spritz']
		temp_df3 = data_f2.loc[data_f2['Material description'] == '500mL ClearPure']
		temp_df4 = data_f2.loc[data_f2['Material description'] == '1L Spritz']
		temp_df5 = data_f2.loc[data_f2['Material description'] == '1L Lemon Spritz']
		temp_df6 = data_f2.loc[data_f2['Material description'] == '1L ClearPure']
		
		for i in data_f1.iterrows():
			if i[1]["Material Description"] == "500mL Spritz":
				if i[1]['Goods'][0] == 1:
					temp_df1.loc[(temp_df1['Round'] == 1) & (temp_df1['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 2:
					temp_df1.loc[(temp_df1['Round'] == 2) & (temp_df1['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 3:
					temp_df1.loc[(temp_df1['Round'] == 3) & (temp_df1['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				else:
					temp_df1.loc[(temp_df1['Round'] == temp_df1['Round'].unique().max()) & (temp_df1['Day'] == 1), 'Qty'] -= i[1]['Quantity']
					
			elif i[1]["Material Description"] == "500mL Lemon Spritz":
				if i[1]['Goods'][0] == 1:
					temp_df2.loc[(temp_df2['Round'] == 1) & (temp_df2['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 2:
					temp_df2.loc[(temp_df2['Round'] == 2) & (temp_df2['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 3:
					temp_df2.loc[(temp_df2['Round'] == 3) & (temp_df2['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				else:
					temp_df2.loc[(temp_df2['Round'] == temp_df2['Round'].unique().max()) & (temp_df2['Day'] == 1), 'Qty'] -= i[1]['Quantity']
					
			elif i[1]["Material Description"] == "500mL ClearPure":
				if i[1]['Goods'][0] == 1:
					temp_df3.loc[(temp_df3['Round'] == 1) & (temp_df3['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 2:
					temp_df3.loc[(temp_df3['Round'] == 2) & (temp_df3['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 3:
					temp_df3.loc[(temp_df3['Round'] == 3) & (temp_df3['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				else:
					temp_df3.loc[(temp_df3['Round'] == temp_df3['Round'].unique().max()) & (temp_df3['Day'] == 1), 'Qty'] -= i[1]['Quantity']
					
			elif i[1]["Material Description"] == "1L Spritz":
				if i[1]['Goods'][0] == 1:
					temp_df4.loc[(temp_df4['Round'] == 1) & (temp_df4['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 2:
					temp_df4.loc[(temp_df4['Round'] == 2) & (temp_df4['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 3:
					temp_df4.loc[(temp_df4['Round'] == 3) & (temp_df4['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				else:
					temp_df4.loc[(temp_df4['Round'] == temp_df4['Round'].unique().max()) & (temp_df4['Day'] == 1), 'Qty'] -= i[1]['Quantity']
				
			elif i[1]["Material Description"] == "1L Lemon Spritz":
				if i[1]['Goods'][0] == 1:
					temp_df5.loc[(temp_df5['Round'] == 1) & (temp_df5['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 2:
					temp_df5.loc[(temp_df5['Round'] == 2) & (temp_df5['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 3:
					temp_df5.loc[(temp_df5['Round'] == 3) & (temp_df5['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				else:
					temp_df5.loc[(temp_df5['Round'] == temp_df5['Round'].unique().max()) & (temp_df5['Day'] == 1), 'Qty'] -= i[1]['Quantity']
					
			elif i[1]["Material Description"] == "1L ClearPure":
				if i[1]['Goods'][0] == 1:
					temp_df6.loc[(temp_df6['Round'] == 1) & (temp_df6['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				elif i[1]['Goods'][0] == 2:
					temp_df6.loc[(temp_df6['Round'] == 2) & (temp_df6['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']		
				elif i[1]['Goods'][0] == 3:
					temp_df6.loc[(temp_df6['Round'] == 3) & (temp_df6['Day'] == int(i[1]['Goods'][2:4])), 'Qty'] -= i[1]['Quantity']
				else:
					temp_df6.loc[(temp_df6['Round'] == temp_df6['Round'].unique().max()) & (temp_df6['Day'] == 1), 'Qty'] -= i[1]['Quantity']
		
		data_f = pd.concat([temp_df1, temp_df2, temp_df3, temp_df4, temp_df5, temp_df6])
		return data_f
	
	def wide_data(data_f):
		
		if len(data_f.loc[:, 'Round'].unique()) == 1:			
			data_f = data_f.sort_values(by = 'Day')
			data_f = data_f.groupby(['Day', 'Material description'])['Qty'].sum().reset_index()
			data_f = pd.pivot_table(data_f, index = 'Day', columns = ['Material description'], values= 'Qty')
			wdf = data_f.fillna(0)
		
		elif len(data_f.loc[:, 'Round'].unique()) == 2:
			tdf_1 = data_f.loc[data_f['Round'] == 1]
			tdf_2 = data_f.loc[data_f['Round'] == 2]
			tdf_1 = tdf_1.sort_values(by = 'Day')
			tfd_1 = tdf_1.groupby(['Day', 'Material description'])['Qty'].sum().reset_index()
			tdf_1 = pd.pivot_table(tdf_1, index = 'Day', columns = ['Material description'], values= 'Qty')
			tdf_2 = tdf_2.sort_values(by = 'Day')
			tfd_2 = tdf_2.groupby(['Day', 'Material description'])['Qty'].sum().reset_index()
			tdf_2 = pd.pivot_table(tdf_2, index = 'Day', columns = ['Material description'], values= 'Qty')
			data_f = pd.concat([tdf_1, tdf_2]).reset_index().drop('Day', axis = 1)
			wdf = data_f.fillna(0)
		else:
			tdf_1 = wide_data(data_f.loc[data_f['Round'] == 1])
			tdf_2 = wide_data(data_f.loc[data_f['Round'] == 2])
			tdf_3 = wide_data(data_f.loc[data_f['Round'] == 3])
			tdf_1 = tdf_1.sort_values(by = 'Day')
			tfd_1 = tdf_1.groupby(['Day', 'Material description'])['Qty'].sum().reset_index()
			tdf_1 = pd.pivot_table(tdf_1, index = 'Day', columns = ['Material description'], values= 'Qty')
			tdf_2 = tdf_2.sort_values(by = 'Day')
			tfd_2 = tdf_2.groupby(['Day', 'Material description'])['Qty'].sum().reset_index()
			tdf_2 = pd.pivot_table(tdf_2, index = 'Day', columns = ['Material description'], values= 'Qty')
			tdf_3 = tdf_1.sort_values(by = 'Day')
			tfd_3 = tdf_1.groupby(['Day', 'Material description'])['Qty'].sum().reset_index()
			tdf_3 = pd.pivot_table(tdf_3, index = 'Day', columns = ['Material description'], values= 'Qty')
			data_f = pd.concat([tdf_1, tdf_2, tdf_3]).reset_index().drop('Day', axis = 1)
			wdf = data_f.fillna(0)
			

		return wdf
	
	def inv_calc(data_f):
		data = data_f.to_numpy().tolist()
		data.insert(0, [1000] * len(data_f.columns))
		data_f = pd.DataFrame(data, index = [0] + data_f.index.tolist(), columns = data_f.columns)
		data_f.columns.name = ''
		for col in range(0, data_f.shape[1]):
			x = 1000
			for row in range(0, data_f.shape[0]):
				if row != 0:
					x = x - data_f.iloc[row, col]
					data_f.iloc[row, col] = x		
		return data_f
	
	def inv_chart(data_f):
		st.subheader('Inventory')
		data_f['Day'] = list(np.arange(1, len(data_f) + 1))
		data_f.rename(columns = {'Day' : 'Day', '' : 'Products', 'value' : 'value'}, inplace = True)
		data_f = pd.melt(data_f, id_vars = ['Day'], value_vars = ['1L ClearPure', '1L Lemon Spritz','1L Spritz',
							  '500mL Spritz', '500mL Lemon Spritz', '500mL ClearPure'])
		data_f.rename(columns = {'Day' : 'Day', '' : 'Products', 'value' : 'value'}, inplace = True)
		chart_3 = px.line(data_f, x = 'Day', y = 'value', color = 'Products', template = 'seaborn')
		chart_3.update_yaxes(title = 'Units in Inventory')

		return chart_3
	
	re_ord = st.sidebar.radio("Did you reorder?", ("Yes", "No"), index = 1)
    
	if re_ord == "Yes":
		file1 = st.sidebar.file_uploader("Please upload the Purchase order tracking report")
		if file1 is not None:
			st.sidebar.success('File uploaded Succesfully')

			df1 = pd.read_excel(file1)
			new_data = inv_calc(wide_data(update_data(df1, df)))
			st.plotly_chart(inv_chart(new_data))
			st.plotly_chart(profit_product(df))
			st.plotly_chart(dem_product(df))

	else:
# 		x=0
		new_data = inv_calc(wide_data(df))
		st.plotly_chart(inv_chart(new_data))
		st.plotly_chart(profit_product(df))
		st.plotly_chart(dem_product(df))
        
        

    
    
