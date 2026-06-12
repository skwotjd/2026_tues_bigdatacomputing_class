# 라이브러리 임포트: 웹 UI, 수치 연산, 모델 로드, 데이터 가공을 위한 핵심 패키지들
import streamlit as st # Streamlit 웹 어플리케이션 제작
import pandas as pd  # 표 형태의 데이터를 다루고 테이블 UI를 생성
import numpy as np # 다차원 배열 연산 및 데이터 형태 변환
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # 학습된 머신러닝 모델(.pkl 파일)을 디스크에서 불러오기

st.set_page_config(page_title="기대수명 예측 프로그램", layout="wide")
st.title("WHO 기대수명 분석 및 예측 대시보드")

try:
    m_linear = joblib.load('model_linear.pkl')
    m_poly = joblib.load('model_poly.pkl')
    m_ridge = joblib.load('model_ridge.pkl')
    df_res = joblib.load('metrics.pkl')
except:  # 만약 해당 경로에 파일이 없다면 사용자에게 경고 메시지를 출력
    st.error("⚠️ 'dog_classifier_models.pkl' 파일이 존재하지 않습니다. 먼저 모델 학습 코드를 실행해 주세요!")
    st.stop()

#사이드바 서브메뉴
st.sidebar.markdown("대시보드 메뉴")
choice = st.sidebar.radio("원하는 화면 선택 :", ["1. 모델 성능 평가 비교", "2. 실시간 수명 예측 UI"])

# 1번 화면: 표랑 막대그래프 출력 섹션
if choice == "1. 모델 성능 평가 비교":
    st.subheader("3종 회귀분석 파이프라인 결과 성능 지표")
    st.markdown("Adult Mortality, BMI, GDP")
    
    # 판다스 테이블
    st.dataframe(df_res)
    
    # Matplotlib 막대그래프
    st.subheader("결정 계수Test R2 Score 비교 시각화")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    sns.barplot(x='Model', y='Test R2', data=df_res, ax=ax, palette='pastel')
    ax.set_ylim(-1.2, 1.0)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=9)
    st.pyplot(fig)

# 2번 화면: 슬라이더 화면
elif choice == "2. 실시간 수명 예측 UI":
    st.subheader("사용자 맞춤 수치 입력 및 예측")
    st.write("슬라이더를 움직여서 관측")
    
    left_col, right_col = st.columns([1, 1.5])
    
    with left_col:
        st.markdown("독립변수 설정")
        v_mortality = st.slider("성인 사망률 (Adult Mortality)", 1, 700, 140, 1)
        v_bmi = st.slider("체질량지수 (BMI)", 1.0, 80.0, 38.0, 0.1)
        v_gdp = st.slider("국내총생산 (GDP)", 1, 95000, 5500, 100)
        
        user_model = st.selectbox("예측 알고리즘 선택:", ["Linear", "Poly", "Ridge"])
        
    with right_col:
        st.markdown("실시간 연산 출력")
        
        input_data = np.array([[v_mortality, v_bmi, v_gdp]])
        
        if user_model == "Linear":
            fit_model = m_linear
        elif user_model == "Poly":
            fit_model = m_poly
        else:
            fit_model = m_ridge
            
        #넘파이 배열로 예측
        pred_y = fit_model.predict(input_data)[0]
        
        if pred_y < 0:
            st.error(f"예측 수명이 {pred_y:.2f}세(음수)로 도출되었습니다. "
                     f"규제가 없는 3차 다항회귀(Poly) 모델이 특정 데이터 구간에서 과대적합되어 나타나는 현상입니다. "
                     f"슬라이더 값을 평균치로 바꾸거나, 규제가 포함된 모델을 골라주세요.")
        else:
            st.metric(label=f"[{user_model}] 모델 기반 최종 예상 수명", value=f"{pred_y:.2f} 세")
            st.progress(int(np.clip(pred_y, 0, 100)))
            
            if pred_y >= 75:
                st.success("높은 기대수명")
            elif pred_y >= 60:
                st.info("평균 기대수명")
            else:
                st.warning("위험 수치")
