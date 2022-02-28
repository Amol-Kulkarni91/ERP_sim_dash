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
	
	if 'spritz5ml_q' not in st.session_state:
		st.session_state['spritz5ml_q'] = []
	if 'lemspritz5ml_q' not in st.session_state:
		st.session_state['lemspritz5ml_q'] = []
	if 'pure5ml_q' not in st.session_state:
		st.session_state['pure5ml_q'] = []
	if 'spritz_q' not in st.session_state:
		st.session_state['spritz_q'] = []
	if 'lemspritz_q' not in st.session_state:
		st.session_state['lemspritz_q'] = []
	if 'pure_q' not in st.session_state:
		st.session_state['pure_q'] = []
	if 'spritz5ml_d' not in st.session_state:
		st.session_state['spritz5ml_d'] = []
	if 'lemspritz5ml_d' not in st.session_state:
		st.session_state['lemspritz5ml_d'] = []
	if 'pure5ml_d' not in st.session_state:
		st.session_state['pure5ml_d'] = []
	if 'spritz_d' not in st.session_state:
		st.session_state['spritz_d'] = []
	if 'lemspritz_d' not in st.session_state:
		st.session_state['lemspritz_d'] = []
	if 'pure_d' not in st.session_state:
		st.session_state['pure_d'] = []
		
	df = pd.read_excel(file)
	df['Profit'] = df['Value'] - df['Cost']
	def dem_product(data_f):
		st.subheader("Demand by Region")
		chart = px.histogram(data_f, x = 'Material description', y = 'Qty', color = 'Area', 
				     barmode = 'group', facet_row = 'Round', height = 600, template = 'seaborn')
		chart.update_xaxes(None)
		chart.update_yaxes(title = 'Total Demand')

		return chart
    
	def wide_data(data_f):
		data_f = data_f.sort_values(by = 'Day')
		data_f = data_f.groupby(['Day', 'Material description'])['Qty'].sum().reset_index()
		data_f = pd.pivot_table(data_f, index = 'Day', columns = ['Material description'], values= 'Qty')
		data_f = data_f.fillna(0)

		return data_f
    
	def first_round(data_f):
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
    
	def second_round(data_f):
		df_1 = wide_data(data_f.loc[data_f['Round'] == 1])
		df_2 = wide_data(data_f.loc[data_f['Round'] == 2])
		wdf = pd.concat([df_1, df_2]).reset_index().drop('Day', axis = 1)
		data_f = wdf

		return wdf
                             
	def third_round(data_f):
		df_1 = data_f.loc[data_f['Round'] == 1]
		df_2 = data_f.loc[data_f['Round'] == 2]
		df_3 = data_f.loc[data_f['Round'] == 3]
		wdf_1 = wide_data(df_1)
		wdf_2 = wide_data(df_2)
		wdf_3 = wide_data(df_3)

		wdf = pd.concat([wdf_1, wdf_2, wdf_3]).reset_index().drop('Day', axis = 1)
		data_f = wdf

		return wdf

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
	
	def update_data(sp5_q, lsp5_q, cp5_q, sp1_q, lsp1_q, cp1_q, sp5_d, lsp5_d, cp5_d, sp1_d, lsp1_d, cp1_d, data_f):
		
		for i in range(0, len(sp5_d)):
			data_f.loc[sp5_d[i] - 1,'500mL Spritz'] = data_f.loc[sp5_d[i] - 1,'500mL Spritz'] - (sp5_q[i]*24)
		for j in range(0, len(lsp5_d)):
			data_f.loc[lsp5_d[j] - 1,'500mL Lemon Spritz'] = data_f.loc[lsp5_d[j] - 1,'500mL Spritz'] - (lsp5_q[j]*24)
		for k in range(0, len(cp5_d)):
			data_f.loc[cp5_d[k] - 1,'500mL ClearPure'] = data_f.loc[cp5_d[k] - 1,'500mL Lemon Spritz'] + (cp5_q[k]*24)
		for l in range(0, len(sp1_d)):
			data_f.loc[sp1_d[l] - 1,'1L Spritz'] = data_f.loc[sp1_d[l] - 1,'500mL ClearPure'] - (sp1_q[l]*12)
		for m in range(0, len(lsp1_d)):
			data_f.loc[lsp1_d[m] - 1,'1L Lemon Spritz'] = data_f.loc[lsp1_d[m] - 1,'1L Lemon Spritz'] + (lsp1_q[m]*12)
		for n in range(0, len(cp1_d)):
			data_f.loc[cp1_d[n] - 1,'1L ClearPure'] = data_f.loc[cp1_d[n] - 1,'1L ClearPure'] - (cp1_q[n]*12)

		return data_f
    
	re_ord = st.sidebar.radio("Did you reorder?", ("Yes", "No"), index = 1)
    
	if re_ord == "Yes":
		st.sidebar.subheader('Reorder Quantity Information')
		with st.sidebar.form(key='my_form'):

			ml_5_spritz = st.sidebar.number_input('Quantity of 500mL Spritz reordered', min_value=0, step=1)
			ml_5_lemspritz = st.sidebar.number_input('Quantity of 500mL Lemon Spritz reordered', min_value=0, step=1)
			ml_5_pure = st.sidebar.number_input('Quantity of 500mL ClearPure reordered', min_value=0, step=1)
			l_1_spritz = st.sidebar.number_input('Quantity of 1L Spritz reordered', min_value=0, step=1)
			l_1_lemspritz = st.sidebar.number_input('Quantity of 1L Lemon Spritz reordered', min_value=0, step=1)
			l_1_pure = st.sidebar.number_input('Quantity of 1L ClearPure reordered', min_value=0, step=1)

			st.sidebar.subheader('Delivery Day')

			day_5_spritz = st.sidebar.number_input('Scheduled Delivery of 500 mL Spritz', min_value = 1, max_value = 60, step = 1)
			day_5_lemspritz = st.sidebar.number_input('Scheduled Delivery of 500 mL Lemon Spritz', min_value = 1, max_value = 60, step = 1)
			day_5_pure = st.sidebar.number_input('Scheduled Delivery of 500 mL ClearPure', min_value = 1, max_value = 60, step = 1)
			day_1_spritz = st.sidebar.number_input('Scheduled Delivery of 1L Spritz', min_value = 1, max_value = 60, step = 1)
			day_1_lemspritz = st.sidebar.number_input('Scheduled Delivery of 1L Lemon Spritz', min_value = 1, max_value = 60, step = 1)
			day_1_pure = st.sidebar.number_input('Scheduled Delivery of 1L ClearPure', min_value = 1, max_value = 60, step = 1)
			submit_button = st.form_submit_button(label='Submit')
			
		if submit_button:
			st.session_state['spritz5ml_q'].append(ml_5_spritz)
			st.session_state['lemspritz5ml_q'].append(ml_5_lemspritz)
			st.session_state['pure5ml_q'].append(ml_5_pure)
			st.session_state['spritz_q'].append(l_1_spritz)
			st.session_state['lemspritz_q'].append(l_1_lemspritz)
			st.session_state['pure_q'].append(l_1_pure)
			st.session_state['spritz5ml_d'].append(day_5_spritz)
			st.session_state['lemspritz5ml_d'].append(day_5_lemspritz)
			st.session_state['pure5ml_d'].append(day_5_pure)
			st.session_state['spritz_d'].append(day_1_spritz)
			st.session_state['lemspritz_d'].append(day_1_lemspritz)
			st.session_state['pure_d'].append(day_1_pure)
			

			if len(df['Round'].unique()) == 1:
				new_data = wide_data(df)
				new_data = update_data(st.session_state['spritz5ml_q'], st.session_state['lemspritz5ml_q'], st.session_state['pure5ml_q'], 
						       st.session_state['spritz_q'], st.session_state['lemspritz_q'], st.session_state['pure_q'], 
						       st.session_state['spritz5ml_d'], st.session_state['lemspritz5ml_d'], st.session_state['pure5ml_d'], 
						       st.session_state['spritz_d'], st.session_state['lemspritz_d'], st.session_state['pure_d'], new_data)
				st.plotly_chart(inv_chart(first_round(new_data)))

			elif len(df['Round'].unique()) == 2:
				new_data = second_round(df)
				new_data = update_data(ml_5_spritz, ml_5_lemspritz, ml_5_pure, l_1_spritz, l_1_lemspritz, l_1_pure, 
						       day_5_spritz, day_5_lemspritz, day_5_pure, day_1_spritz, day_1_lemspritz, day_1_pure, new_data)
				st.plotly_chart(inv_chart(first_round(new_data)))
			else:
				new_data = third_round(df)
				new_data = update_data(ml_5_spritz, ml_5_lemspritz, ml_5_pure, l_1_spritz, l_1_lemspritz, l_1_pure, 
						       day_5_spritz, day_5_lemspritz, day_5_pure, day_1_spritz, day_1_lemspritz, day_1_pure, new_data)
				st.plotly_chart(inv_chart(first_round(new_data)))

			st.plotly_chart(profit_product(df))
			st.plotly_chart(dem_product(df))

	else:
		if len(df['Round'].unique()) == 1:
			st.plotly_chart(inv_chart(first_round(wide_data(df))))

		elif len(df['Round'].unique()) == 2:
			st.plotly_chart(inv_chart(first_round(second_round(df))))
		else:
			st.plotly_chart(inv_chart(first_round(third_round(df))))

		st.plotly_chart(profit_product(df))
		st.plotly_chart(dem_product(df))
        
        

    
    
