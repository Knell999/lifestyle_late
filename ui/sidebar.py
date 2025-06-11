"""
Sidebar components for Streamlit app
"""

import streamlit as st
from typing import Dict, Any

def create_sidebar() -> Dict[str, Any]:
    """
    Create sidebar with navigation and configuration options
    
    Returns:
        Dict containing user selections
    """
    st.sidebar.title("🏦 KCB Grade Prediction")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "📋 페이지 선택",
        ["홈", "데이터 업로드", "모델 훈련", "예측", "시각화", "모델 비교"]
    )
    
    
    # Configuration options
    config = {}
    config['page'] = page
    
    if page in ["모델 훈련", "예측", "모델 비교"]:
        st.sidebar.subheader("⚙️ 모델 설정")
        
        # Model selection
        config['models'] = st.sidebar.multiselect(
            "모델 선택",
            ["Random Forest", "XGBoost", "LightGBM", "Ensemble"],
            default=["Random Forest", "XGBoost"]
        )
        
        # Data mode selection
        config['data_mode'] = st.sidebar.selectbox(
            "데이터 모드",
            ["Full", "Life", "Financial"],
            help="Full: 전체 데이터, Life: 라이프스타일 데이터만, Financial: 금융 데이터만"
        )
        
        # Cross-validation settings
        config['cv_folds'] = st.sidebar.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5
        )
        
        # Advanced settings
        with st.sidebar.expander("🔧 고급 설정"):
            config['test_size'] = st.sidebar.slider(
                "테스트 데이터 비율",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05
            )
            
            config['random_state'] = st.sidebar.number_input(
                "Random State",
                min_value=1,
                max_value=1000,
                value=42
            )
    
    st.sidebar.markdown("---")
    
    # Information section
    with st.sidebar.expander("ℹ️ 프로젝트 정보"):
        st.sidebar.markdown("""
        **KCB Grade Prediction**
        
        - 🎯 목적: KCB 등급 예측
        - 📊 데이터: 라이프스타일 + 금융 데이터
        - 🤖 모델: RF, XGBoost, LightGBM, Ensemble
        - 📈 평가: 정확도, F1-Score, ROC-AUC
        """)
    
    return config
