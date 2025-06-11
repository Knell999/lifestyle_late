"""
Main content components for different pages
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

def create_home_page():
    """Create home page content"""
    st.title("🏦 KCB Grade Prediction Dashboard")
    
    # Hero section
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center; margin: 0;">
            🎯 고객 신용등급 예측 시스템
        </h2>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">
            머신러닝을 활용한 정확한 KCB 등급 예측
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                    border-left: 4px solid #007bff;">
            <h4 style="margin-top: 0; color: #007bff;">📊 데이터 특성</h4>
            <ul style="margin: 0;">
                <li>라이프스타일 변수</li>
                <li>금융 행동 변수</li>
                <li>개인 정보 변수</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                    border-left: 4px solid #28a745;">
            <h4 style="margin-top: 0; color: #28a745;">🤖 ML 모델</h4>
            <ul style="margin: 0;">
                <li>Random Forest</li>
                <li>XGBoost</li>
                <li>LightGBM</li>
                <li>Ensemble</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                    border-left: 4px solid #ffc107;">
            <h4 style="margin-top: 0; color: #ffc107;">📈 평가 지표</h4>
            <ul style="margin: 0;">
                <li>정확도 (Accuracy)</li>
                <li>F1-Score</li>
                <li>ROC-AUC</li>
                <li>교차검증</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature description
    st.markdown("## 📋 주요 기능")
    
    tab1, tab2, tab3, tab4 = st.tabs(["데이터 업로드", "모델 훈련", "예측", "시각화"])
    
    with tab1:
        st.markdown("""
        ### 📂 데이터 업로드
        - CSV 파일 업로드 지원
        - 데이터 미리보기 및 통계 정보
        - 결측값 및 이상치 탐지
        - 데이터 품질 검증
        """)
    
    with tab2:
        st.markdown("""
        ### 🎯 모델 훈련
        - 다중 알고리즘 동시 훈련
        - 하이퍼파라미터 자동 튜닝
        - 교차검증을 통한 성능 평가
        - 모델 성능 비교 및 선택
        """)
    
    with tab3:
        st.markdown("""
        ### 🔮 예측
        - 개별 고객 등급 예측
        - 배치 예측 지원
        - 예측 확률 및 신뢰도 표시
        - 예측 결과 다운로드
        """)
    
    with tab4:
        st.markdown("""
        ### 📊 시각화
        - 데이터 분포 시각화
        - 특성 중요도 분석
        - 모델 성능 비교 차트
        - 한글 폰트 지원
        """)
    
    # Getting started
    st.markdown("## 🚀 시작하기")
    st.info("""
    1. **사이드바**에서 원하는 페이지를 선택하세요
    2. **데이터 업로드** 페이지에서 CSV 파일을 업로드하세요
    3. **모델 훈련** 페이지에서 모델을 훈련시키세요
    4. **예측** 페이지에서 새로운 데이터에 대한 예측을 수행하세요
    """)

def create_main_content(config: Dict[str, Any]):
    """
    Create main content based on selected page
    
    Args:
        config: Configuration dictionary from sidebar
    """
    page = config.get('page', '홈')
    
    if page == "홈":
        create_home_page()
    elif page == "데이터 업로드":
        from .data_upload import create_data_upload_section
        create_data_upload_section()
    elif page == "모델 훈련":
        from .model_selection import create_model_training_section
        create_model_training_section(config)
    elif page == "예측":
        from .model_selection import create_prediction_section
        create_prediction_section(config)
    elif page == "시각화":
        from .visualization import create_visualization_section
        create_visualization_section(config)
    elif page == "모델 비교":
        from .results_display import create_model_comparison_section
        create_model_comparison_section(config)
