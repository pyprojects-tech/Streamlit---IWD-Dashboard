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

file =  pd.read_csv('data.csv',low_memory=False)
file['WELL_LOCATION_TOWN'] = file['WELL_LOCATION_TOWN'].astype(str)
file = file.rename(columns={'LATITUDE':'lat','LONGITUDE':'lon'})
locations = file['WELL_LOCATION_TOWN'].unique()

locfilt = st.sidebar.radio("Filtering Based on City or Coordinates",('Select Citie(s)','Enter Coordinates'))

if locfilt == 'Select Citie(s)':
    location = st.sidebar.multiselect('Well Location',np.sort(locations),default='BANGOR')
    if location:
        filt1 = file.set_index('WELL_LOCATION_TOWN')
        file1 = filt1.loc[location]
        boxdf = pd.DataFrame()
        boxdf = pd.DataFrame()
        dfl1 = pd.DataFrame()
        dfl2 = pd.DataFrame()
        dfl3 = pd.DataFrame()
        dfl4 = pd.DataFrame()
        dfl0 = pd.DataFrame()
    else:
        filt1 = file.set_index('WELL_LOCATION_TOWN')
        file1 = filt1
        boxdf = pd.DataFrame()
        dfl1 = pd.DataFrame()
        dfl2 = pd.DataFrame()
        dfl3 = pd.DataFrame()
        dfl4 = pd.DataFrame()
        dfl0 = pd.DataFrame()
else:
    st.sidebar.write('Enter Coordinates and Distance')
    longitude = st.sidebar.number_input('Well Longitude',value=-68.77464)
    latitude = st.sidebar.number_input('Well Latitude',value=44.803143)
    distance = st.sidebar.number_input('Distance from Coordinates to Filter (miles)',value=10)
    
    new_lat1 = latitude  + (distance / 3950) * (180 / np.pi)
    new_lon1 = longitude + (distance / 3950) * (180 / np.pi / np.cos(latitude * np.pi/180))
    new_lat2 = latitude  - (distance / 3950) * (180 / np.pi)
    new_lon2 = longitude - (distance / 3950) * (180 / np.pi / np.cos(latitude * np.pi/180))
    
    latvec = np.linspace(new_lat1,new_lat2,1000)
    lonvec= np.linspace(new_lon1,new_lon2,1000)
    
    dfl1 = pd.DataFrame(np.array([latvec,np.ones(1000)*new_lon1]).T,columns=['lat','lon'])
    dfl2 = pd.DataFrame(np.array([latvec,np.ones(1000)*new_lon2]).T,columns=['lat','lon'])
    dfl3 = pd.DataFrame(np.array([lonvec,np.ones(1000)*new_lat1]).T,columns=['lon','lat'])
    dfl4 = pd.DataFrame(np.array([lonvec,np.ones(1000)*new_lat2]).T,columns=['lon','lat'])
    dfl0 = pd.DataFrame(np.array([longitude*np.ones(2),latitude*np.ones(2)]).T,columns=['lon','lat'])
    
       
    file1 = file.set_index('WELL_LOCATION_TOWN')
    
st.sidebar.write('Enter Numerical Filter Values for Dataset')    
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

if locfilt == 'Select Citie(s)':
    file2 = file1[(file1['WELL_DEPTH_FT']>wellmin) & (file1['WELL_DEPTH_FT']<wellmax) &
                           (file1['WELL_YIELD_GPM']>yieldmin) & (file1['WELL_YIELD_GPM']<yieldmax) &
                           (file1['CASING_LENGTH_FT']>case_min) & (file1['CASING_LENGTH_FT']<case_max)
                           ]
else:
    file2 = file1[(file1['WELL_DEPTH_FT']>wellmin) & (file1['WELL_DEPTH_FT']<wellmax) &
                           (file1['WELL_YIELD_GPM']>yieldmin) & (file1['WELL_YIELD_GPM']<yieldmax) &
                           (file1['CASING_LENGTH_FT']>case_min) & (file1['CASING_LENGTH_FT']<case_max) &
                           (file1['lon']>new_lon2) & (file1['lon']<new_lon1) &
                           (file1['lat']>new_lat2) & (file1['lat']<new_lat1)                                                
                           ]   

hydrofrac = st.sidebar.radio("Filtering Based on Hydrofracing",('Show All','Only Show Hydrofractured'))
if hydrofrac == 'Only Show Hydrofractured':
    file3 =  file2[file2['HYDROFRACTURE']=='YES']
else:
    file3 = file2

data_table_expand = st.beta_expander("Show Data Table",expanded=False)
with data_table_expand:
    st.write(file3)
      
col1,col2,col3 = st.beta_columns(3)
file3['DRILL_DATE']= pd.to_datetime(file3['DRILL_DATE'])
#COLUMN 1

wmax =file3['WELL_DEPTH_FT'].max()
wmin =file3['WELL_DEPTH_FT'].min()
plot = sns.histplot(file3['WELL_DEPTH_FT'],kde=True)
plot.set(title='WELL DEPTH FT')
plot.set_xlim(left=wmin,right=wmax)
fig = plot.get_figure()
col1.write(file3['WELL_DEPTH_FT'].describe(percentiles=[]))
col1.pyplot(fig,clear_figure=True)

plot = sns.scatterplot(data=file3, x='DRILL_DATE',y='WELL_DEPTH_FT')
plot.set(title='WELL DEPTH OVER TIME')
fig = plot.get_figure()
col1.pyplot(fig,clear_figure=True)
#COLUMN 2

cmax =file3['CASING_LENGTH_FT'].max()
cmin =file3['CASING_LENGTH_FT'].min()
plot = sns.histplot(file3['CASING_LENGTH_FT'],kde=True)
plot.set(title='CASING LENGHT FT')
plot.set_xlim(left=cmin,right=cmax)
fig = plot.get_figure()
col2.write(file3['CASING_LENGTH_FT'].describe(percentiles=[]))
col2.pyplot(fig,clear_figure=True)

plot = sns.scatterplot(data=file3, x='DRILL_DATE',y='CASING_LENGTH_FT')
plot.set(title='CASING LENGTH OVER TIME')
fig = plot.get_figure()
col2.pyplot(fig,clear_figure=True)

#COLUMN 3  
ymax =file3['WELL_YIELD_GPM'].max()
ymin =file3['WELL_YIELD_GPM'].min() 
plot = sns.histplot(file3['WELL_YIELD_GPM'],kde=True)
plot.set(title='WELL YIELD GPM')
plot.set_xlim(left=ymin,right=ymax)
fig = plot.get_figure()
col3.write(file3['WELL_YIELD_GPM'].describe(percentiles=[]),use_container_width=True)
col3.pyplot(fig,clear_figure=True)


plot = sns.scatterplot(data=file3, x='DRILL_DATE',y='WELL_YIELD_GPM')
plot.set(title='WELL YIELD GPM OVER TIME')
fig = plot.get_figure()
col3.pyplot(fig,clear_figure=True)

# import plotly.express as px
# fig = px.scatter_mapbox(file3, lat="lat", lon="lon",     color="WELL_YIELD_GPM", size="WELL_DEPTH_FT",
#                   color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)
# fig.update_layout(mapbox_style="carto-darkmatter",height=750)
# st.plotly_chart(fig,use_container_width=True,height=100)

df = file3[['lat','lon','WELL_DEPTH_FT']].dropna()

import pydeck as pdk


layer=[
    pdk.Layer(
       'HexagonLayer',
       data=df,
       get_position='[lon, lat]',
       elevation=['WELL_DEPTH_FT'],
       elevation_scale=10,
       elevation_range=[100/3.2,1000/3.2],
       radius = 100,
       pickable=True,
       extruded=True,
       auto_highlight=True,
    ),

    
    # pdk.Layer(
    #     'ScatterplotLayer',
    #     data=dfl1,
    #     get_position='[lon, lat]',
    #     get_color='[255, 0, 0]',
    #     get_radius=200,
    # ),
    # pdk.Layer(
    #     'ScatterplotLayer',
    #     data=dfl2,
    #     get_position='[lon, lat]',
    #     get_color='[255, 0, 0]',
    #     get_radius=200,
    # ),
    # pdk.Layer(
    #     'ScatterplotLayer',
    #     data=dfl3,
    #     get_position='[lon, lat]',
    #     get_color='[255, 0, 0]',
    #     get_radius=200,
    # ),
    
    # pdk.Layer(
    #     'ScatterplotLayer',
    #     data=dfl4,
    #     get_position='[lon, lat]',
    #     get_color='[255, 0, 0]',
    #     get_radius=200,
    # ),
    # pdk.Layer(
    #     'ScatterplotLayer',
    #     data=dfl0,
    #     get_position='[lon, lat]',
    #     opacity=0.5,
    #     stroked=True,
    #     filled=True,
    #     get_line_color=['0, 0, 0'],
    #     get_fill_color='[255, 0, 0]',
    #     get_radius=500,
    # ),
    
    
]

r = pdk.Deck(
    map_style='mapbox://styles/mapbox/navigation-night-v1',
    layers = [layer],
    tooltip={"text": "{position}: {WELL_DEPTH_FT}"},
    initial_view_state=pdk.ViewState(
        latitude=file3['lat'].mean(),
        longitude=file3['lon'].mean(),
        zoom=11,
        pitch=50,
    
    ),
    )
    
st.pydeck_chart(r)