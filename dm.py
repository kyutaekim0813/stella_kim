import pandas as pd
import openpyxl
import seaborn as sns
import numpy as np
import matplotlib
import scipy.stats as stats
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime, timedelta
import json
import plotly.express as px


matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']  = False # 한글 폰트 사용시, 마이너스 글자가 깨지는 현상 방지 


# ERP_data를 불러오는 함수 
def load_dataset(path):
    return pd.read_excel(path)
     
path = 'erp_data.xlsx'
df = load_dataset(path)

material_code_list = df['자재 코드'].unique().tolist()

# JSON을 읽어 들이는 함수.
def loadJSON(path):
    f = open(path, 'r')
    res = json.load(f)
    f.close()
    return res


# 로고 Lottie와 타이틀 출력.
col1, col2 = st.columns([1,2])
with col1:
    lottie = loadJSON('Yay Jump.json')
    st_lottie(lottie, speed=1, loop=True, width=150, height=150)
with col2:
    ''
    ''
    st.subheader('1.자재코드별 작업자 비교')


#사이드바
with st.sidebar:
   st.markdown('조회할 자재코드를 입력해주세요: :tulip:')
   st.selectbox(
       '자재코드를 선택해주세요:', material_code_list, key='material_code'
   )


df['생산일자'] = df['생산일자'].astype('str')
df['생산일자'] = pd.to_datetime(df['생산일자'])
df['생산_년도'] = df['생산일자'].dt.year
df['생산_월'] = df['생산일자'].dt.month
df['생산_일'] = df['생산일자'].dt.day
df['평균외경'] = df['평균외경'].astype('float')
df['최대외경'] = df['최대외경'].astype('float')
df = df[df['자재 코드'] == st.session_state['material_code']]



# 데이터 프레임 출력 (앞 5행)
st.dataframe(df.head())

# 차트그리는 함수
y_hline = round(df['최소외경'].mean(),2)   # 수평선 추가 (최소외경)
fig = plt.figure(figsize=(10,5))
for worker in df['작업자'].unique():
    df[df['작업자'] == worker].plot(x='생산일자', y='외경검출기 실제', kind='line', label=worker, ax=plt.gca())
    plt.axhline(y= y_hline, color='r', linestyle='--')
    plt.xlabel('생산일자')
    plt.ylabel('외경검출기 실제')   
    plt.grid()
    plt.legend()
st.pyplot(fig)


st.divider()
st.subheader('2. 선형회귀모델로 회귀계수 및 절편 도출')



st.divider()
st.subheader('3. Target에 대한 최적파라미터값 도출')
