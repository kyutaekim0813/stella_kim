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
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer   #결측치 처리
from sklearn.preprocessing import StandardScaler  #표준화 (평균0,표준편차1)
from sklearn.pipeline import Pipeline  # 파이프라인 구성
from sklearn.model_selection import GridSearchCV   # 교차검증
from sklearn.metrics import r2_score  # 평가방법
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer   #결측치 처리
from sklearn.preprocessing import StandardScaler  #표준화 (평균0,표준편차1)
from sklearn.pipeline import Pipeline  # 파이프라인 구성
from sklearn.model_selection import GridSearchCV   # 교차검증
from sklearn.metrics import r2_score  # 평가방법


#font_path = r"font/NanumGothic.ttf"
#font_name = fm.FontProperties(fname=font_path).get_name()
#plt.rc('font',family=font_name)

# 폰트 설정
# Streamlit Cloud 환경에서 apt.txt 또는 packages.txt를 통해 나눔 폰트를 설치했다고 가정
# 설치된 폰트를 찾아서 사용합니다.
try:
    # 나눔고딕 폰트 경로를 자동으로 찾습니다.
    # 실제 시스템에 설치된 폰트 이름은 'NanumGothic'일 수도 있고,
    # 'NanumGothicOTF'일 수도 있습니다.
    # 먼저 일반적인 'NanumGothic'을 시도하고, 없으면 다른 이름을 시도하거나,
    # findSystemFonts()로 찾은 목록에서 선택할 수 있습니다.
    # 정확한 폰트 이름은 시스템에 따라 다를 수 있습니다.
    font_name = 'NanumGothic' # 또는 'NanumGothicOTF' 등 실제 설치된 폰트 이름
    fm.fontManager.addfont('font/NanumGothic.ttf') # 직접 폰트 파일 경로 추가 (시스템 폰트)
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
    st.write(f"폰트 설정 완료: {font_name}") # 디버깅용 메시지

except Exception as e:
    st.warning(f"한글 폰트 설정 중 오류 발생: {e}. 기본 폰트를 사용합니다. (배포 환경 폰트 설치 확인 필요)")
    # Fallback for local testing or if font not found in cloud
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic') # 윈도우 기본 한글 폰트
    elif platform.system() == 'Darwin': # Mac
        plt.rc('font', family='AppleGothic')
    else: # Linux (대부분의 클라우드 환경)
        # 나눔고딕이 설치되어 있지 않다면 깨질 수 있음.
        # 이 경우 apt.txt 또는 packages.txt로 폰트 설치가 필수.
        plt.rc('font', family='DejaVu Sans') # 기본 영문 폰트, 한글 깨짐 발생
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지



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
filtered_df = df[df['자재 코드'] == st.session_state['material_code']].copy().reset_index(drop=True)

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
st.dataframe(filtered_df[['생산일자','자재 코드','작업자','최소외경','외경검출기 실제']].head())

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
    ax.text( x=filtered_df['생산일자'].min(), y= y_hline+0.05 , s= f'최소외경:{y_hline}', color='r' ) # x좌표를 데이터 시작점으로
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)

else:
    st.warning('선택된 자재코드에 해당하는 데이터가 없습니다.')

st.divider()
st.subheader('2. 선형회귀모델 회귀계수 및 절편값 확인')

df_clean = df[['MCYL1', 'MCYL2', 'MCYL3', 'MCYL4', 'MCYL5', 'MNECK', 'MHEAD', 
                             'MDIE','P/O 저선기 텐션', 'T/U 저선기 텐션','다이','외경검출기 실제']].dropna()

for col_name in df_clean.columns.tolist():
    df_clean[col_name] = pd.to_numeric(df_clean[col_name], errors='coerce')

df_clean.dropna(inplace=True)

#st.dataframe(df_clean.head())

X = df_clean[['MCYL1', 'MCYL2','MCYL3', 'MCYL4', 'MCYL5', 'MNECK', 'MHEAD', 'MDIE','P/O 저선기 텐션', 'T/U 저선기 텐션','다이']]
Y = df_clean['외경검출기 실제']

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3)

pipe_list = [('impute', SimpleImputer()),
             ('scaler', StandardScaler()),
             ('model', LinearRegression())]

model_pipe = Pipeline(pipe_list)
hyper_parameter = {}
grid_model = GridSearchCV(model_pipe, param_grid = hyper_parameter, cv=3)

grid_model.fit(X_train, Y_train)
best_model = grid_model.best_estimator_

Y_train_pred = best_model.predict(X_train)
Y_test_pred =  best_model.predict(X_test)

st.write("Train r2 Score: " , round(r2_score(Y_train,Y_train_pred),3))
st.write("Test r2 Score: " , round(r2_score(Y_test,Y_test_pred),3))

# 회귀계수 확인 
#best_model['model'].coef_
df2 = pd.DataFrame(best_model['model'].coef_, index=X.columns, columns=['회귀계수'])
st.dataframe(df2)

# 절편 확인 
st.write("절편값:" , round(best_model['model'].intercept_,3))



st.divider()
st.subheader('3. Target에 대한 최적파라미터값 도출')






# st.subheader('2. 선형회귀모델로 회귀계수 및 절편 도출')

# st.divider()
# st.subheader('3. Target에 대한 최적파라미터값 도출')
