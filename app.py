import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm 
import matplotlib

import seaborn as sns

import ast 

from datetime import datetime

st.set_page_config(layout="wide")
plt.style.use('dark_background')
#matplotlib.pyplot.ion


#Start of App Including Type and Raw Data
st.title('IWD Well Intelligence Dashboard')
st.set_option('deprecation.showPyplotGlobalUse', False)

file =  pd.read_csv('data.csv',parse_dates=True,index_col='DRILL_DATE',low_memory=False)
file['WELL_LOCATION_TOWN'] = file['WELL_LOCATION_TOWN'].astype(str)
file = file.rename(columns={'LATITUDE':'lat','LONGITUDE':'lon'})
locations = file['WELL_LOCATION_TOWN'].unique()

st.subheader('Data Filtering')


location = st.sidebar.multiselect('Well Location',np.sort(locations))
if location:
    filt1 = file.set_index('WELL_LOCATION_TOWN')
    file1 = filt1.loc[location]
else:
    filt1 = file.set_index('WELL_LOCATION_TOWN')
    file1 = filt1
    


well_max = file1['WELL_DEPTH_FT'].max()
well_min = file1['WELL_DEPTH_FT'].min()
wellmin = st.sidebar.number_input('Well Depth Minimum Filter',well_min,well_max,value=well_min)
wellmax = st.sidebar.number_input('Well Depth Maximum Filter',well_min,well_max,value=well_max)

yield_max = file1['WELL_YIELD_GPM'].max()
yield_min = file1['WELL_YIELD_GPM'].min()
yield_min = file1['WELL_YIELD_GPM'].min()
yield_max = file1['WELL_YIELD_GPM'].max()
yieldmin =  st.sidebar.number_input('Well Yield Maximum Filter',yield_min,yield_max,value=yield_min)
yieldmax =  st.sidebar.number_input('Well Yield Maximum Filter',yield_min,yield_max,value=yield_max)

case_min = file1['CASING_LENGTH_FT'].min()
case_max = file1['CASING_LENGTH_FT'].max()
casemin =  st.sidebar.number_input('Well Casing Maximum Filter',case_min,case_max,value=case_min)
casemax =  st.sidebar.number_input('Well Casing Maximum Filter',case_min,case_max,value=case_max)

case_max = file1['CASING_LENGTH_FT'].max()
case_min = file1['CASING_LENGTH_FT'].min()

file2 = file1[(file1['WELL_DEPTH_FT']>wellmin) & (file1['WELL_DEPTH_FT']<wellmax) &
                       (file1['WELL_YIELD_GPM']>yieldmin) & (file1['WELL_YIELD_GPM']<yieldmax) &
                       (file1['CASING_LENGTH_FT']>case_min) & (file1['CASING_LENGTH_FT']<case_max)
                       ]

hydrofrac = st.sidebar.radio("Filtering Based on Hydrofracing",('Show All','Only Show Hydrofractured'))
if hydrofrac == 'Only Show Hydrofractured':
    file3 =  file2[file2['HYDROFRACTURE']=='YES']
else:
    file3 = file2
# # with col004:
#    st_dev_filt = st.number_input('Std. Deviation Filter to Remove Values > x St. Dev',
#                         min_value = 0, 
#                         max_value = 100,
#                         value = 3,
#                         step=1)

data_table_expand = st.beta_expander("Show Data Table",expanded=False)
with data_table_expand:
    st.write(file3)
      
col1,col2,col3 = st.beta_columns(3)
#COLUMN 1

plot = sns.histplot(file3['WELL_DEPTH_FT'],kde=True)
plot.set(title='WELL DEPTH FT')
plot.set_xlim(left=wellmin,right=wellmax)
fig = plot.get_figure()
col1.write(file3['WELL_DEPTH_FT'].describe(percentiles=[]))
col1.pyplot(fig,clear_figure=True)

#COLUMN 2

plot = sns.histplot(file3['CASING_LENGTH_FT'],kde=True)
plot.set(title='CASING LENGHT FT')
plot.set_xlim(left=casemin,right=casemax)
fig = plot.get_figure()
col2.write(file3['CASING_LENGTH_FT'].describe(percentiles=[]))
col2.pyplot(fig,clear_figure=True)

#COLUMN 3   
plot = sns.histplot(file3['WELL_YIELD_GPM'],kde=True)
plot.set(title='WELL YIELD GPM')
plot.set_xlim(left=yieldmin,right=yieldmax)
fig = plot.get_figure()
col3.write(file3['WELL_YIELD_GPM'].describe(percentiles=[]),use_container_width=True)
col3.pyplot(fig,clear_figure=True)

import plotly.express as px
fig = px.scatter_mapbox(file3, lat="lat", lon="lon",     color="WELL_YIELD_GPM", size="WELL_DEPTH_FT",
                  color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)
fig.update_layout(mapbox_style="carto-darkmatter",height=750)
st.plotly_chart(fig,use_container_width=True,height=100)