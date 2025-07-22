import pandas as pd
import openpyxl
import numpy as np
from matplotlib import font_manager,rc
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime, timedelta
import json
import platform
import matplotlib.font_manager as fm


#matplotlib.rcParams['font.family'] = 'Malgun Gothic'
#matplotlib.rcParams['axes.unicode_minus']  = False # 한글 폰트 사용시, 마이너스 글자가 깨지는 현상 방지 

#plt.rcParams['axes.unicode_minus'] = False
#plt.rc('font', family='Malgun Gothic')

font_path = r"font/NanumGothic.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font',family=font_name)
matplotlib.font_manager._rebuild()


# ERP_data를 불러오는 함수 
@st.cache_data
def load_dataset(path):
    return pd.read_excel(path)

path = 'erp_data.xlsx'
df = load_dataset(path)

# 자재코드 리스트 (NaN제거 및 정수 변환)
material_code_list = df['자재 코드'].dropna().astype(int).unique().tolist()


# 세션 상태를 초기화 한다. (데이터 로드 후에 수행)
if 'material_code' not in st.session_state:
    # 기본값이 리스트에 없으면 첫 번째 값 사용
    default_code = 500016810 if 500016810 in material_code_list else (material_code_list[0] if material_code_list else None)
    if default_code is not None:
       st.session_state['material_code'] = 500016810
    else:
        st.error('유효한 자재가 없습니다.')
        st.stop()

# 데이터 프레임 필터링 (세션 상태 초기화 후 수행)
filtered_df = df[df['자재 코드'] == st.session_state['material_code']].copy()

# 데이터 프레임 전처리 (세션 상태 초기화 후 수행 )
filtered_df['생산일자'] = filtered_df['생산일자'].astype('str')
filtered_df['생산일자'] = pd.to_datetime(filtered_df['생산일자'])
filtered_df['생산_년도'] = filtered_df['생산일자'].dt.year
filtered_df['생산_월'] = filtered_df['생산일자'].dt.month
filtered_df['생산_일'] = filtered_df['생산일자'].dt.day
filtered_df['평균외경'] = filtered_df['평균외경'].astype('float')
filtered_df['최대외경'] = filtered_df['최대외경'].astype('float')


# JSON을 읽어 들이는 함수.
def loadJSON(path):
    f = open(path, 'r')
    res = json.load(f)
    f.close()
    return res

# 로고 Lottie와 타이틀 출력.
col1, col2 = st.columns([1,3])
with col1:
    lottie = loadJSON('Yay Jump.json')
    st_lottie(lottie, speed=1, loop=True, width=150, height=150)
with col2:
    ''
    ''
    st.subheader('자재 코드에 따른 작업자간 외경 치수 비교')


# 사이드바에서 폼을 통해서 인자를 생성한다. 
with st.sidebar.form(key='chartsetting', clear_on_submit=True):
    st.markdown('조회할 자재코드를 입력해주세요: :tulip:')
    ''
    # 현재 선택된 값의 인덱스 찾기
    try:
        current_index = material_code_list.index(st.session_state['material_code'])
    except ValueError:
        current_index = 0
        st.session_state['material_code'] = material_code_list[current_index]

    selected_code =  st.selectbox('자재코드를 선택해주세요:', options = material_code_list, index= current_index)
    ''
    ''
    if st.form_submit_button(label='확인'):
        st.session_state['material_code'] = selected_code
        st.rerun()



# 데이터 프레임 출력 (앞 5행)
st.dataframe(filtered_df.head())

# 차트그리는 함수
if not filtered_df.empty:
    y_hline = round(filtered_df['최소외경'].mean(),2)   # 수평선 추가 (최소외경)
    fig, ax = plt.subplots(figsize=(10,5))

    for worker in filtered_df['작업자'].unique():
        worker_data = filtered_df[filtered_df['작업자'] == worker]
        ax.plot(worker_data['생산일자'], worker_data['외경검출기 실제'],label=worker)

    ax.axhline(y= y_hline, color='r', linestyle='--')
    ax.set_xlabel('생산일자')
    ax.set_ylabel('외경검출기 실제')   
    ax.grid()
    ax.legend()
    st.pyplot(fig)

else:
    st.warning('선택된 자재코드에 해당하는 데이터가 없습니다.')

# st.divider()
# st.subheader('2. 선형회귀모델로 회귀계수 및 절편 도출')

# st.divider()
# st.subheader('3. Target에 대한 최적파라미터값 도출')
