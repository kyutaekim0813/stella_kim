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
from sklearn.metrics import r2_score  # 평가지표  - 값이 1에 가까울 수록 좋다. 
from sklearn.metrics import mean_squared_error  # 평가지표 - 값이 0에 가까울 수록 좋다 (왜냐하면 규제를 하는 모델이기때문임/ 오버피팅 방지)
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge  
from sklearn.linear_model import ElasticNet
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

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

# 절대 지우지 말것 (비쥬얼 스튜디오 즉, 로컬 PC에서 작성한 코드랑 다르다) 
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


# 절대 지우지 말것 (비쥬얼 스튜디오 즉, 로컬 PC에서 작성한 코드랑 다르다) 


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

if 'regression_value' not in st.session_state:
    st.session_state['regression_value'] = '선형회귀'  # 기본값 설정

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
    st.subheader('1. 자재코드별 작업자 외경 치수 비교')


# 사이드바에서 폼을 통해서 인자를 생성한다. 
with st.sidebar.form(key='chartsetting', clear_on_submit=True):
    st.markdown('조회할 자재코드를 선택해주세요: :tulip:')
    # 현재 선택된 값의 인덱스 찾기
    try:
        current_index = material_code_list.index(st.session_state['material_code'])
    except ValueError:
        current_index = 0
        st.session_state['material_code'] = material_code_list[current_index]

    selected_code =  st.selectbox(' ',options = material_code_list, index= current_index)
    ''
    ''
    ''
    ''
    if st.form_submit_button(label='확인'):
        st.session_state['material_code'] = selected_code
        st.rerun()

''
''
''
with st.sidebar:
    st.markdown('회귀모델을 선택해주세요: :sunglasses:')
    regression_value = st.radio(label = '', options = ['선형회귀', 'LASSO','RIDGE','ELASTIC'])
    st.write('회귀모델', regression_value, '을 선택하셨습니다.')
    
    # 세션 상태 업데이트
    st.session_state['regression_value'] = regression_value


 

# 데이터 프레임 출력 (앞 5행)
st.dataframe(filtered_df[['생산일자','자재 코드','작업자','최소외경','외경검출기 실제']].head())

y_hline = round(filtered_df['최소외경'].mean(),2)  # 수평선 추가 (최소외경)

# 차트그리는 함수
# if not filtered_df.empty:
#     y_hline = round(filtered_df['최소외경'].mean(),2)   # 수평선 추가 (최소외경)
#     fig, ax = plt.subplots(figsize=(10,5))

#     for worker in filtered_df['작업자'].unique():
#         worker_data = filtered_df[filtered_df['작업자'] == worker]
#         ax.plot(worker_data['생산일자'], worker_data['외경검출기 실제'],label=worker)
        

#     ax.axhline(y= y_hline, color='r', linestyle='--')
#     ax.set_xlabel('생산일자')
#     ax.set_ylabel('외경검출기 실제')   
#     ax.grid()
#     ax.text( x=filtered_df['생산일자'].min(), y= y_hline+0.05 , s= f'최소외경:{y_hline}', color='r' ) # x좌표를 데이터 시작점으로
#     ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     st.pyplot(fig)

# else:
#     st.warning('선택된 자재코드에 해당하는 데이터가 없습니다.')

# Plotly Figure 객체 생성

fig_go = go.Figure()

# 각 작업자별 라인 추가
for worker in filtered_df['작업자'].unique():
    worker_data = filtered_df[filtered_df['작업자'] == worker]
    fig_go.add_trace(go.Scatter(
        x=worker_data['생산일자'],
        y=worker_data['외경검출기 실제'],
        mode='lines',
        name=worker,
        hoverinfo='x+y+name' # 마우스 오버 시 표시 정보
    ))

# 수평선 추가 (go.Figure.add_shape 사용)
fig_go.add_shape(
    type="line",
    x0=filtered_df['생산일자'].min(),
    y0=y_hline,
    x1=filtered_df['생산일자'].max(),
    y1=y_hline,
    line=dict(color="Red", width=2, dash="dash"),
    name=f'최소외경:{y_hline}' # shape에는 직접 label이 없으므로 annotation으로 대체
)

# 수평선 텍스트 라벨 추가 (go.Figure.add_annotation 사용)
fig_go.add_annotation(
    x=filtered_df['생산일자'].min(), # x 위치는 데이터 시작점으로
    y=y_hline + 0.05,               # y 위치는 수평선보다 약간 위로
    text=f'최소외경:{y_hline}',
    showarrow=False,                # 화살표 없음
    font=dict(color="Red", size=10),
    xanchor="left",                 # 텍스트 정렬 (왼쪽 기준)
    yanchor="bottom"                # 텍스트 정렬 (아래 기준)
)

# 레이아웃 설정 (제목, 축 라벨, 그리드, 범례, x축 글자 회전)
fig_go.update_layout(
    title={
        'text': f"{st.session_state['material_code']} 의 자재 및 작업자별 외경 사이즈 비교",
        'x': 0.5, # 제목 중앙 정렬
        'xanchor': 'center'
    },
    xaxis_title='생산일자',
    yaxis_title='외경검출기 실제',
    #xaxis_tickangle=90, # x축 글자 45도 회전
    hovermode="x unified", # 마우스 오버 시 x축 기준으로 정보 통합 표시
    legend=dict(
        orientation="v", # 범례 방향 (vertical)
        yanchor="top",   # y축 앵커 (상단)
        y=1,             # y 위치 (1은 그래프 상단)
        xanchor="left",  # x축 앵커 (왼쪽)
        x=1.02           # x 위치 (1.02는 그래프 영역 바로 밖)
    ),
    margin=dict(r=150), # 범례가 들어갈 오른쪽 여백 확보
    height=500, # 그래프 높이 설정
    width=1200,  # 그래프 너비 설정
    xaxis=dict(showgrid=True), # x축 그리드 표시
    yaxis=dict(showgrid=True)  # y축 그리드 표시
)

# Streamlit에 Plotly 그래프 표시
st.plotly_chart(fig_go, use_container_width=True) # use_container_width=True로 컨테이너 너비에 맞춤


st.divider()
st.subheader('2. 선형회귀모델 회귀계수 및 절편값 확인')

df_clean = df[['MCYL1', 'MCYL2', 'MCYL3', 'MCYL4', 'MCYL5', 'MNECK', 'MHEAD', 
                             'MDIE','P/O 저선기 텐션', 'T/U 저선기 텐션','다이','외경검출기 실제']].dropna()

for col_name in df_clean.columns.tolist():
    df_clean[col_name] = pd.to_numeric(df_clean[col_name], errors='coerce')

df_clean.dropna(inplace=True)


X = df_clean[['MCYL1', 'MCYL2','MCYL3', 'MCYL4', 'MCYL5', 'MNECK', 'MHEAD', 'MDIE','P/O 저선기 텐션', 'T/U 저선기 텐션','다이']]
Y = df_clean['외경검출기 실제']

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=3174)

# 선택된 모델에 따라 모델 인스턴스 생성
model_instance = None
if st.session_state['regression_value'] == '선형회귀':
    model_instance = LinearRegression()
elif st.session_state['regression_value'] == 'LASSO':
    model_instance = Lasso(random_state=3174)
elif st.session_state['regression_value'] == 'RIDGE':
    model_instance = Ridge(random_state=3174)
elif st.session_state['regression_value'] == 'ELASTIC':
    model_instance = ElasticNet(random_state=3174)        


pipe_list = [('impute', SimpleImputer()),
             ('scaler', StandardScaler()),
             ('model', model_instance)]

model_pipe = Pipeline(pipe_list)
hyper_parameter = {}
grid_model = GridSearchCV(model_pipe, param_grid = hyper_parameter, cv=3)

grid_model.fit(X_train, Y_train)
best_model = grid_model.best_estimator_

Y_train_pred = best_model.predict(X_train)
Y_test_pred =  best_model.predict(X_test)

# 평가지표
if st.session_state['regression_value'] == '선형회귀':
    st.write("Train Set의 r2 값 (1에 가까우면 좋다.): " , round(r2_score(Y_train,Y_train_pred),3))
    st.write("Test Set의 r2 값 (1에 가까우면 좋다.): " , round(r2_score(Y_test,Y_test_pred),3))
elif st.session_state['regression_value'] == 'LASSO':
    st.write("Train set의 MSE 값 (0에 가까우면 좋다.): " , round(mean_squared_error(Y_train,Y_train_pred),3))
    st.write("Test set의 MSE 값 (0에 가까우면 좋다.): " , round(mean_squared_error(Y_test,Y_test_pred),3))
elif st.session_state['regression_value'] == 'RIDGE':
    st.write("Train set의 MSE 값 (0에 가까우면 좋다.): " , round(mean_squared_error(Y_train,Y_train_pred),3))
    st.write("Test set의 MSE 값 (0에 가까우면 좋다.): " , round(mean_squared_error(Y_test,Y_test_pred),3))
elif st.session_state['regression_value'] == 'ELASTIC':
    st.write("Train set의 MSE 값 (0에 가까우면 좋다.): " , round(mean_squared_error(Y_train,Y_train_pred),3))
    st.write("Test set의 MSE 값 (0에 가까우면 좋다.): " , round(mean_squared_error(Y_test,Y_test_pred),3))        

# 회귀계수 확인 
#best_model['model'].coef_
df2 = pd.DataFrame(best_model['model'].coef_, index=X.columns, columns=['회귀계수'])
st.dataframe(df2)


# 절편 확인 
st.write("절편값:" , round(best_model['model'].intercept_,3))

st.divider()
st.subheader('3. Target에 대한 최적파라미터값 도출')

# 회귀모델 계수 및 절편 

coef = best_model['model'].coef_ 
intercept = best_model['model'].intercept_
Y_target = y_hline   # 목표 Y 값

# 1. 목적 함수 정의 (MSE 최소화)
def objective(x_scaled):
    # 모델은 스케일링된 데이터를 입력으로 받으므로, x_scaled를 직접 사용합니다.
    return (np.dot(x_scaled, coef) + intercept - Y_target)**2

# 2. 초기값 설정 (예: 중간값 또는 랜덤 값)
# 초기값은 스케일링된 도메인에 있어야 합니다.
# StandardScaler는 기본적으로 평균 0, 표준편차 1로 스케일링하므로 0이 좋은 초기값입니다.
x_init_scaled = np.zeros(len(coef))

# 3. 변수별 제약조건 (Bounds)
# 주의: minimize 함수에 전달되는 bounds는 objective 함수가 받는 x의 스케일과 일치해야 합니다.
# 현재 objective 함수는 스케일링된 x를 받으므로, bounds도 스케일링된 값으로 변환해야 합니다.
# 하지만 SciPy의 minimize는 bounds를 원본 스케일로 지정하고, 메서드가 내부적으로 처리하는 경우가 많습니다.
# 여기서는 원본 df의 min/max를 사용하고, minimize가 이를 잘 처리하도록 맡깁니다.
# 중요한 것은 bounds의 순서가 coef의 순서(즉, feature_cols의 순서)와 정확히 일치해야 한다는 것입니다.
bounds_original_scale = [
    (X[col].min(), X[col].max()) for col in X.columns
]


# # --- 디버깅 및 유효성 검사 추가 ---
# if len(x_init_scaled) != len(bounds_original_scale):
#     st.error(f"오류: 초기값(x0)의 길이({len(x_init_scaled)})와 제약조건(bounds)의 길이({len(bounds_original_scale)})가 일치하지 않습니다.")
#     st.error("이는 `feature_cols` 리스트가 모델 학습에 사용된 실제 특성 컬럼들과 일치하지 않기 때문일 수 있습니다.")
#     st.error(f"feature_cols: {df_clean.columns}")
#     st.error(f"coef 길이: {len(coef)}")
#     st.stop() # 앱 실행 중단


# 4. 최적화 실행
result = minimize(
    objective,
    x_init_scaled,
    bounds=bounds_original_scale,
    method='L-BFGS-B' # 최적화 메서드 지정
    
)

# 5. 결과 해석 및 역변환 (가장 중요!)
# result.x는 스케일링된 최적의 특성 값입니다.
# 이를 원래 스케일로 되돌리려면 StandardScaler의 inverse_transform을 사용해야 합니다.
# best_model['scaler']는 파이프라인 내의 StandardScaler 객체입니다.
optimal_x_scaled = result.x
optimal_x_original_scale = best_model['scaler'].inverse_transform(optimal_x_scaled.reshape(1, -1))[0]

# 6. 결과 출력
st.write(f"**최적화 성공 여부:** {result.success}")
#st.write(f"**최적화 메시지:** {result.message}")
#st.write(f"**최적의 스케일링된 X 값:** {optimal_x_scaled}")
#st.write("---")
st.write("**최적의 X 값 :**")


# # 6. 결과 출력
# st.write("Success:", result.success)  # 해의 존재성 , 제약조건이 너무 엄격하면 해가 없을 수 있다.  해가 있으면 Success ! 
# st.write("Optimal X:", result.x)
# st.write("Predicted Y:", np.dot(result.x, coef) + intercept)

# 각 특성 이름과 함께 출력하여 가독성 높이기
optimal_features_dict = dict(zip(X.columns, optimal_x_original_scale.round(2)))
st.write(optimal_features_dict)

st.write(f"**예측 Y값:** {np.dot(optimal_x_scaled, coef) + intercept:.2f}")
st.write(f"**목표 Y 값:** {Y_target}")
#st.write(f"**최소화된 목적 함수 값 (MSE):** {result.fun:.2f}")

st.info("참고: 최적화 결과는 초기값, 메서드, 제약조건, 그리고 모델의 특성에 따라 달라질 수 있습니다.")
#st.info("특히 `optimal_x_original_scale`은 모델이 목표 Y를 달성하기 위해 '가장 적합하다고 판단한' 원본 스케일의 특성 조합입니다.")
#st.info("`bounds`의 순서가 `X의 컬럼` 및 `coef`의 컬럼 순서와 정확히 일치하는지 다시 한번 확인해주세요.")
