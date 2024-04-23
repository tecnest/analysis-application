#!/usr/bin/env python
# coding: utf-8


#------------------------------------loading data--------------------------------------------------#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


df= pd.read_excel("https://github.com/tecnest/General-review/raw/main/ITEMS%26COLOR(1).xlsx", header=None)

new_header = df.iloc[2]  # Grab the third row for the header, remember it's index 2 because of zero-indexing
df = df[3:]  # Take the data less the header row
df.columns = new_header  # Set the header row as the DataFrame header

# Reset the index of the DataFrame if necessary
df.reset_index(drop=True, inplace=True)


numeric_columns = ['TY.Sales', 'oh', 'A_SellThru', 'TY.Qty', 'A_Days']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

#------------------------------------cleaning data--------------------------------------------------#
import numpy as np

# Replace 'inf' with 'NaN'
df.replace([np.inf, -np.inf], np.nan, inplace=True)
pd.DataFrame(df)

# Drop rows where 'TY.sales' is NaN
df.dropna(subset=['TY.Sales'], inplace=True)

# Drop rows with 'NaN'
df.dropna(inplace=True)
#pd.DataFrame(df)


#-----------------------------------drop columns  &  group_bys  -----------------------------------------------------#

import locale
locale.setlocale(locale.LC_ALL, '')

st.title('General review')
st.header('Boy & Men Review')


df1=df.copy()

columns_to_drop = ['itemid', 'EailiestDate', 'TY.Disc%', 'A_AvgQSoldD-MinDate']  # List of columns to drop
df1.drop(columns_to_drop, axis=1, inplace=True)

agg_funcs = {
    'TY.Qty': 'sum',
    'TY.Sales': 'sum',
    'oh':'sum',
    'A_SellThru': 'mean',
}

# Group by parameter and apply custom aggregation functions
gender = df1.groupby('Gender').agg(agg_funcs)
Cat=df1.groupby('Cat2').agg(agg_funcs)
Color=df1.groupby('Color').agg(agg_funcs)
men = df1[df1['Gender'] == 'Men']
boys = df1[df1['Gender'] == 'Boys']


#-----------------------------gender--------------------------------------#

#gender.drop('A_Days', axis=1, inplace=True)
total_OH = gender['oh'].sum()
total_sales=gender['TY.Sales'].sum()
gender['Sales%']=(gender['TY.Sales']/ total_sales) * 100
gender['Stock%'] = (gender['oh'] / total_OH) * 100

desired_order = ['TY.Sales', 'Sales%', 'oh', 'Stock%', 'TY.Qty', 'A_SellThru']
# Reindex the DataFrame with the desired order of columns
gender = gender.reindex(columns=desired_order)

def format_numbers(x):
    if isinstance(x, (int, float)):
        return '{:,.2f}'.format(x).replace(',', ' ')
    return x

gender=gender.applymap(format_numbers)
gender



#-------------------------------------category------------------------------#

# <div style="text-align: left; color: orange;"><h2>Category Review</h2></div>
 

# In[3]:


total_OH = Cat['oh'].sum()
total_sales=Cat['TY.Sales'].sum()
Cat['Sales%']=(Cat['TY.Sales']/ total_sales) * 100
Cat['Stock%'] = (Cat['oh'] / total_OH) * 100

desired_order = ['TY.Sales', 'Sales%', 'oh', 'Stock%', 'TY.Qty', 'A_SellThru']
# Reindex the DataFrame with the desired order of columns
Cat = Cat.reindex(columns=desired_order)
Cat = Cat.sort_values(by='TY.Sales', ascending=False)
Cat1=Cat.copy()

Cat=Cat.applymap(format_numbers)
    
st.header('Category Review')
pd.DataFrame(Cat)
Cat


# In[7]:
#-----------------------------visualisation----------------------------------#


import streamlit as st
import plotly.graph_objs as go
import pandas as pd


fig = go.Figure()
fig.add_trace(go.Bar(x=Cat1.index, y=Cat1['TY.Sales'], hoverinfo='y', marker_color='royalblue'))
fig.update_layout(width=1100, height=550, yaxis_title='TY.Sales', template='plotly_white')
st.plotly_chart(fig)  # Display the figure in Streamlit



# In[6]:


fig1 = go.Figure()
fig1.add_trace(go.Pie(labels=Cat1.index, values=Cat1['Sales%'], hole=0.3))
fig1.update_layout(title='Sales Percentage by Category', width=1000, height=1000,legend=dict(
        x=1.5,  # Position the legend outside the pie chart on the right
        y=1,  # Align the legend with the top of the pie chart
        xanchor='left',  # Anchor the legend to the left of the x position
        bgcolor='rgba(255, 255, 255, 0.5)',  # Optional: background color with transparency
        bordercolor='Black',  # Optional: border color
        borderwidth=1  # Optional: border width
    ), template='plotly_white')


# Display the interactive plots
st.plotly_chart(fig1)



#--------------------------------Colors------------------------------#
st.title('Color review')

total_OH = Color['oh'].sum()
total_sales=Color['TY.Sales'].sum()
Color['Sales%']=(Color['TY.Sales']/ total_sales) * 100
Color['Stock%'] = (Color['oh'] / total_OH) * 100


Color = Color.reindex(columns=desired_order)
Color = Color.sort_values(by='TY.Sales', ascending=False)
Color1 = Color.copy()

Color=Color.applymap(format_numbers)

st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable deprecation warning
pd.set_option('display.max_rows', 200)

# Get the first 200 rows
first_50_rows = Color.head(200)

# Display the result
st.write(first_50_rows)


num_categories = len(Color1.index)
num_figures = num_categories // 50 + 1  # Adjust the number as needed
categories_per_figure = 50
start_idx = 0

st.header('Sales by color')

max_ty_sales = Color1['TY.Sales'].max()
for i in range(num_figures):
    end_idx = min(start_idx + categories_per_figure, num_categories)
    subset_df = Color1[start_idx:end_idx]
    
    fig3 = go.Figure(go.Bar(x=subset_df.index, y=subset_df['TY.Sales'], name='TY.Sales'))
    fig3.update_layout(width=1100, height=550,
        title=f'TY.Sales by Category (Part {i+1})',
        xaxis_title='Category',
        yaxis_title='TY.Sales',
        yaxis=dict(range=[0, max_ty_sales + (0.1 * max_ty_sales)])  # Extend y-axis slightly above max
    )
    
    # Display the figure in the notebook
    st.plotly_chart(fig3)
    
    start_idx += categories_per_figure

st.header('Sales percentages by colors')

# Pie chart creation
fig2 = go.Figure()
fig2.add_trace(go.Pie(labels=Color1.index, values=Color1["Sales%"], hole=0.3, textinfo='none'))
fig2.update_layout(
    title='Sales Percentage by Category',
    width=800,
    height=800,
    legend=dict(
        x=1.5,  # Position the legend outside the pie chart on the right
        y=1,
        xanchor='left',
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor='Black',
        borderwidth=1
    ),
    template='plotly_white'
)

# Display the pie chart in the notebook
st.plotly_chart(fig2)


#------------------------------size------------------------------------#

ds=pd.read_excel("https://github.com/tecnest/General-review/raw/main/ITEMS%26SIZE.xlsx", header=None)
new_header = ds.iloc[2]
ds = ds[3:]  # Take the data less the header row
ds.columns = new_header  # Set the header row as the DataFrame header
ds.reset_index(drop=True, inplace=True)

ds.replace([np.inf, -np.inf], np.nan, inplace=True)
ds.dropna(subset=['TY.Sales'], inplace=True)
# Drop rows with 'NaN'


st.title('Size review')

ds1=ds.copy()

columns_to_drop = ['itemid', 'EailiestDate', 'TY.Disc%', 'A_AvgQSoldD-MinDate']  # List of columns to drop
ds1.drop(columns_to_drop, axis=1, inplace=True)

agg_funcs = {
    'TY.Qty': 'sum',
    'TY.Sales': 'sum',
    'oh':'sum',
    'A_SellThru': 'mean',
}

# Group by parameter and apply custom aggregation functions

Size=ds1.groupby('Size').agg(agg_funcs)


total_OH = Size['oh'].sum()
total_sales=Size['TY.Sales'].sum()
Size['Sales%']=(Size['TY.Sales']/ total_sales) * 100
Size['Stock%'] = (Size['oh'] / total_OH) * 100


desired_order = ['TY.Sales', 'Sales%', 'oh', 'Stock%', 'TY.Qty', 'A_SellThru']
# Reindex the DataFrame with the desired order of columns
Size = Size.reindex(columns=desired_order)
Size = Size.sort_values(by='TY.Sales', ascending=False)

Size=Size.applymap(format_numbers)
st.dataframe(Size)

#------------------visualisation size---------------------#

size_order = Size.index.tolist() 
Size.index = pd.Categorical(Size.index, categories=size_order, ordered=True)  
fig5 = go.Figure(go.Bar(x=Size.index, y=Size['TY.Sales'], name='TY.Sales'))
fig5.update_layout(
    width=1100, height=550,
    title=f'TY.Sales by Size',
    xaxis=dict(
        title='Size',
        type='category',  # Ensure x-axis treats data as categorical
        categoryorder='array',  # Ensures categories are shown as per the order in the index
        categoryarray=desired_order,  # Ensures specific order from the array
	tickvals=Size.index,  # Specify tick values to ensure all categories are shown
        ticktext=Size.index  # Specify the text for each tick value   
 ),
    yaxis_title='TY.Sales')
    
# Display the figure in the notebook
st.plotly_chart(fig5)
    

# Pie chart creation
fig6 = go.Figure()
fig6.add_trace(go.Pie(labels=Size.index, values=Size['Sales%'], hole=0.3, textinfo='none'))
fig6.update_layout(
    title='Sales Percentage by Size',
    width=800,
    height=800,
    legend=dict(
        x=1.5,  # Position the legend outside the pie chart on the right
        y=1,
        xanchor='left',
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor='Black',
        borderwidth=1
    ),
    template='plotly_white'
)

# Display the pie chart in the notebook
st.plotly_chart(fig6, width=1000)

