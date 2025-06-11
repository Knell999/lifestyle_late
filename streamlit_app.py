"""
KCB Grade Prediction Streamlit App
Main application file for the web interface
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.append(str(src_path))

# Page configuration
st.set_page_config(
    page_title="KCB Grade Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Knell999/lifestyle_late',
        'Report a bug': 'https://github.com/Knell999/lifestyle_late/issues',
        'About': '''
        # KCB Grade Prediction App
        
        이 앱은 머신러닝을 활용하여 고객의 KCB 신용등급을 예측합니다.
        
        **주요 기능:**
        - 📊 데이터 업로드 및 탐색
        - 🤖 다중 ML 모델 훈련
        - 🔮 신용등급 예측
        - 📈 성능 시각화 및 분석
        
        **개발자:** KHJ
        **버전:** 1.0.0
        '''
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .stAlert > div {
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stTabs > div > div > div > div {
        padding: 1rem;
    }
    
    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Style the sidebar */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'trained_pipeline' not in st.session_state:
        st.session_state.trained_pipeline = None
    
    # Import UI components
    try:
        from ui.sidebar import create_sidebar
        from ui.main_content import create_main_content
    except ImportError as e:
        st.error(f"UI 모듈을 불러올 수 없습니다: {str(e)}")
        st.error("필요한 패키지를 설치했는지 확인하세요: pip install -r requirements.txt")
        return
    
    # Create sidebar and get configuration
    config = create_sidebar()
    
    # Create main content based on configuration
    try:
        create_main_content(config)
    except Exception as e:
        st.error(f"페이지 로딩 중 오류가 발생했습니다: {str(e)}")
        st.error("개발자에게 문의하세요.")
        
        # Show error details in expander for debugging
        with st.expander("오류 상세 정보 (개발자용)"):
            st.exception(e)

if __name__ == "__main__":
    main()
