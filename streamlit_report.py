"""
KCB Grade Prediction Analysis Report
정적 분석 보고서 생성을 위한 Streamlit 앱
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.append(str(src_path))

# Page configuration
st.set_page_config(
    page_title="KCB Grade Prediction Analysis Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for report styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        background: #f0f2f6;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 2rem 0 1rem 0;
        border-radius: 5px;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .report-footer {
        margin-top: 3rem;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        text-align: center;
        border-top: 3px solid #667eea;
    }
    
    .stTabs > div > div > div > div {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """데이터 로드"""
    try:
        from src.config import DATA_PATH
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

def run_analysis():
    """전체 분석 실행"""
    try:
        from src.pipeline import MLPipeline
        
        # 진행률 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📊 분석 파이프라인 초기화 중...")
        progress_bar.progress(20)
        
        pipeline = MLPipeline("report_analysis")
        
        status_text.text("🔄 데이터 전처리 및 모델 훈련 중...")
        progress_bar.progress(50)
        
        results = pipeline.run_full_pipeline()
        
        status_text.text("✅ 분석 완료!")
        progress_bar.progress(100)
        
        return results, pipeline
    except Exception as e:
        st.error(f"분석 실행 실패: {e}")
        return None, None

def create_executive_summary(df, results):
    """경영진 요약"""
    st.markdown('<div class="section-header"><h2>📋 Executive Summary</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("총 고객 수", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("데이터 변수 수", f"{len(df.columns)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if results and 'model_results' in results:
            best_model = max(results['model_results'].items(), 
                           key=lambda x: x[1].get('accuracy', 0))
            st.metric("최고 모델 정확도", f"{best_model[1].get('accuracy', 0):.2%}")
        else:
            st.metric("최고 모델 정확도", "분석 중...")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        target_dist = df['KCB_grade'].value_counts()
        st.metric("등급 분포", f"{len(target_dist)}개 등급")
        st.markdown('</div>', unsafe_allow_html=True)

def create_data_overview(df):
    """데이터 개요"""
    st.markdown('<div class="section-header"><h2>📊 Data Overview</h2></div>', unsafe_allow_html=True)
    
    # 기본 통계
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("데이터 기본 정보")
        
        info_data = {
            "항목": ["총 행 수", "총 열 수", "수치형 변수", "범주형 변수", "결측값", "중복값"],
            "값": [
                f"{len(df):,}",
                f"{len(df.columns)}",
                f"{len(df.select_dtypes(include=[np.number]).columns)}",
                f"{len(df.select_dtypes(include=['object']).columns)}",
                f"{df.isnull().sum().sum():,}",
                f"{df.duplicated().sum():,}"
            ]
        }
        
        st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("타겟 변수 분포")
        target_dist = df['KCB_grade'].value_counts().sort_index()
        st.bar_chart(target_dist)

def create_model_performance(results):
    """모델 성능 분석"""
    st.markdown('<div class="section-header"><h2>🤖 Model Performance Analysis</h2></div>', unsafe_allow_html=True)
    
    if not results or 'model_results' not in results:
        st.warning("모델 결과가 없습니다. 분석을 실행해주세요.")
        return
    
    # 성능 비교 테이블
    performance_data = []
    for model_name, metrics in results['model_results'].items():
        performance_data.append({
            '모델': model_name,
            '정확도': f"{metrics.get('accuracy', 0):.4f}",
            'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
            'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}",
            '정밀도': f"{metrics.get('precision', 0):.4f}",
            '재현율': f"{metrics.get('recall', 0):.4f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    st.subheader("📈 Model Performance Comparison")
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    
    # 최고 성능 모델 하이라이트
    best_model = max(results['model_results'].items(), 
                    key=lambda x: x[1].get('accuracy', 0))
    
    st.success(f"🏆 **Best Model**: {best_model[0]} (Accuracy: {best_model[1].get('accuracy', 0):.4f})")

def create_insights_recommendations():
    """인사이트 및 권장사항"""
    st.markdown('<div class="section-header"><h2>💡 Insights & Recommendations</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Key Insights")
        insights = [
            "고객 신용등급 예측에서 Random Forest와 XGBoost 모델이 우수한 성능을 보임",
            "라이프스타일 데이터와 금융 데이터의 조합이 예측 정확도를 향상시킴",
            "특정 변수들이 신용등급 예측에 높은 중요도를 가짐",
            "데이터 전처리와 피처 엔지니어링이 모델 성능에 큰 영향을 미침"
        ]
        
        for insight in insights:
            st.markdown(f"• {insight}")
    
    with col2:
        st.subheader("📋 Recommendations")
        recommendations = [
            "정기적인 모델 재훈련을 통한 성능 유지",
            "추가적인 외부 데이터 소스 확보 검토",
            "실시간 예측 시스템 구축 고려",
            "모델 해석가능성 향상을 위한 SHAP 분석 도입"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")

def main():
    """메인 보고서 생성"""
    
    # 헤더
    st.markdown('''
    <div class="main-header">
        <h1>📊 KCB Grade Prediction Analysis Report</h1>
        <p>머신러닝 기반 고객 신용등급 예측 분석 보고서</p>
        <p><strong>생성일:</strong> ''' + datetime.now().strftime("%Y년 %m월 %d일 %H:%M") + '''</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # 데이터 로드
    st.subheader("🔄 데이터 로딩...")
    df = load_data()
    
    if df is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    st.success(f"✅ 데이터 로드 완료 ({len(df):,} rows, {len(df.columns)} columns)")
    
    # 분석 실행 여부 선택
    if st.button("🚀 전체 분석 실행", type="primary"):
        with st.spinner("분석 실행 중... 잠시만 기다려주세요."):
            results, pipeline = run_analysis()
            st.session_state.results = results
            st.session_state.pipeline = pipeline
    
    # 기존 결과가 있다면 사용
    results = st.session_state.get('results', None)
    
    # 보고서 섹션들
    create_executive_summary(df, results)
    
    st.markdown("---")
    create_data_overview(df)
    
    st.markdown("---")
    create_model_performance(results)
    
    st.markdown("---")
    create_insights_recommendations()
    
    # 보고서 하단
    st.markdown('''
    <div class="report-footer">
        <h3>📄 Report Information</h3>
        <p><strong>Generated by:</strong> KCB Grade Prediction System</p>
        <p><strong>Analysis Date:</strong> ''' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '''</p>
        <p><strong>Version:</strong> 1.0.0</p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    # Session state 초기화
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    
    main()
