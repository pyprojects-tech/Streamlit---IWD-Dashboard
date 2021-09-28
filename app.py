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

file =  pd.read_csv('data.csv',parse_dates=True,index_col='DRILL_DATE')
file['WELL_LOCATION_TOWN'] = file['WELL_LOCATION_TOWN'].astype(str)
locations = file['WELL_LOCATION_TOWN'].unique()

st.subheader('Data Filtering')

col001, col002, col003,col004  = st.beta_columns([2,2,1,1])
with col001:
    location = st.multiselect('Well Location',np.sort(locations))
if location:
    filt1 = file.set_index('WELL_LOCATION_TOWN')
    file1 = filt1.loc[location]
else:
    filt1 = file.set_index('WELL_LOCATION_TOWN')
    file1 = filt1
    

with col002:
     well_max = file1['WELL_DEPTH_FT'].max()
     well_min = file1['WELL_DEPTH_FT'].min()
     well_depth_filt = st.slider('Select Well Depth Range', well_min,well_max,(well_min,well_max))
     file2 = file1[(file1['WELL_DEPTH_FT']>well_depth_filt[0]) & (file1['WELL_DEPTH_FT']<well_depth_filt[1])]
with col003:
      hydrofrac = st.radio("Filtering Based on Hydrofracing",('Show All','Only Show Hydrofractured'))
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

      
col1,col2,col3,col4 = st.beta_columns([1,1,1,1])
with col1:
    wellmin = st.number_input('X-axis Min',well_min,well_max,value=well_min)
    wellmax = st.number_input('X-axis Max',well_min,well_max,value=well_max)
    
    plot = sns.histplot(file3['WELL_DEPTH_FT'],kde=True)
    plot.set(title='WELL DEPTH FT')
    plot.set_xlim(left=wellmin,right=wellmax)
    fig = plot.get_figure()
    st.pyplot(fig,clear_figure=True)
with col2: 
    case_min = file3['CASING_LENGTH_FT'].min()
    case_max = file3['CASING_LENGTH_FT'].max()
    casemin = st.number_input('X-axis Min',case_min,case_max,value=case_min)
    casemax = st.number_input('X-axis Max',case_min,case_max,value=case_max)
    
    plot = sns.histplot(file3['CASING_LENGTH_FT'],kde=True)
    plot.set(title='CASING LENGHT FT')
    plot.set_xlim(left=casemin,right=casemax)
    fig = plot.get_figure()
    st.pyplot(fig,clear_figure=True)
with col3: 
    yield_min = file3['WELL_YIELD_GPM'].min()
    yield_max = file3['WELL_YIELD_GPM'].max()
    yieldmin = st.number_input('X-axis Min',yield_min,yield_max,value=yield_min)
    yieldmax = st.number_input('X-axis Max',yield_min,yield_max,value=yield_max)
    
    plot = sns.histplot(file3['WELL_YIELD_GPM'],kde=True)
    plot.set(title='WELL YIELD GPM')
    plot.set_xlim(left=yieldmin,right=yieldmax)
    fig = plot.get_figure()
    st.pyplot(fig,clear_figure=True)
# with col4: 
#     plot = sns.histplot(file3['WELL_DRILLER_COMPANY'],kde=True)
#     plot.set(title='WELL DRILLER COMPANY')
#     fig = plot.get_figure()
#     st.pyplot(fig,clear_figure=True)