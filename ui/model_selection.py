"""
Model selection and training components
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_model_training_section(config: Dict[str, Any]):
    """Create model training interface"""
    st.title("🎯 모델 훈련")
    
    # Check if data is available
    if 'data' not in st.session_state:
        st.warning("⚠️ 먼저 '데이터 업로드' 페이지에서 데이터를 업로드하세요.")
        return
    
    df = st.session_state.data
    
    # Training configuration
    st.markdown("### ⚙️ 훈련 설정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**선택된 모델**\n{', '.join(config['models'])}")
    
    with col2:
        st.info(f"**데이터 모드**\n{config['data_mode']}")
    
    with col3:
        st.info(f"**CV Folds**\n{config['cv_folds']}")
    
    # Advanced settings display
    with st.expander("🔧 세부 설정"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**테스트 데이터 비율**: {config['test_size']}")
            st.write(f"**Random State**: {config['random_state']}")
        with col2:
            st.write(f"**총 데이터 크기**: {len(df):,} 행")
            st.write(f"**훈련 데이터 예상**: {int(len(df) * (1 - config['test_size'])):,} 행")
    
    # Start training button
    if st.button("🚀 모델 훈련 시작", type="primary", use_container_width=True):
        train_models(df, config)

def train_models(df: pd.DataFrame, config: Dict[str, Any]):
    """Train models with progress tracking"""
    try:
        # Import required modules
        from pipeline import MLPipeline
        from config import TARGET_COLUMN
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize pipeline
        status_text.text("🔧 파이프라인 초기화 중...")
        progress_bar.progress(10)
        
        pipeline = MLPipeline(
            target_column=TARGET_COLUMN,
            test_size=config['test_size'],
            random_state=int(config['random_state']),
            cv_folds=config['cv_folds']
        )
        
        # Load and preprocess data
        status_text.text("📊 데이터 전처리 중...")
        progress_bar.progress(30)
        
        results = pipeline.run_pipeline(
            data=df,
            mode=config['data_mode'].lower(),
            models_to_train=config['models']
        )
        
        progress_bar.progress(100)
        status_text.text("✅ 모델 훈련 완료!")
        
        # Store results in session state
        st.session_state.training_results = results
        st.session_state.trained_pipeline = pipeline
        
        # Display results
        display_training_results(results)
        
    except Exception as e:
        st.error(f"❌ 훈련 중 오류가 발생했습니다: {str(e)}")

def display_training_results(results: Dict[str, Any]):
    """Display training results"""
    st.markdown("### 📊 훈련 결과")
    
    if 'model_results' in results:
        # Model performance comparison
        st.markdown("#### 🏆 모델 성능 비교")
        
        performance_data = []
        for model_name, model_result in results['model_results'].items():
            performance_data.append({
                '모델': model_name,
                '정확도': f"{model_result.get('accuracy', 0):.4f}",
                'F1-Score': f"{model_result.get('f1_score', 0):.4f}",
                'ROC-AUC': f"{model_result.get('roc_auc', 0):.4f}",
                'CV 평균': f"{model_result.get('cv_scores', [0])[0] if model_result.get('cv_scores') else 0:.4f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Best model highlight
        if performance_data:
            best_model = max(performance_data, key=lambda x: float(x['정확도']))
            st.success(f"🎯 **최고 성능 모델**: {best_model['모델']} (정확도: {best_model['정확도']})")
    
    # Feature importance (if available)
    if 'feature_importance' in results:
        st.markdown("#### 📈 특성 중요도")
        
        importance_df = pd.DataFrame(results['feature_importance'])
        if not importance_df.empty:
            # Display top 10 features
            top_features = importance_df.head(10)
            st.bar_chart(top_features.set_index('feature')['importance'])
            
            with st.expander("전체 특성 중요도 보기"):
                st.dataframe(importance_df, use_container_width=True)

def create_prediction_section(config: Dict[str, Any]):
    """Create prediction interface"""
    st.title("🔮 예측")
    
    # Check if model is trained
    if 'trained_pipeline' not in st.session_state:
        st.warning("⚠️ 먼저 '모델 훈련' 페이지에서 모델을 훈련시키세요.")
        return
    
    pipeline = st.session_state.trained_pipeline
    
    # Prediction options
    st.markdown("### 📝 예측 방법 선택")
    
    prediction_mode = st.radio(
        "예측 방법을 선택하세요:",
        ["개별 고객 예측", "배치 예측"],
        horizontal=True
    )
    
    if prediction_mode == "개별 고객 예측":
        create_individual_prediction(pipeline)
    else:
        create_batch_prediction(pipeline)

def create_individual_prediction(pipeline):
    """Create individual customer prediction interface"""
    st.markdown("#### 👤 개별 고객 정보 입력")
    
    # Get feature names from the pipeline
    try:
        from config import ONEHOT_FEATURES, LABEL_FEATURES, BINARY_FEATURES
        
        # Create input form
        with st.form("individual_prediction"):
            col1, col2 = st.columns(2)
            
            user_input = {}
            
            with col1:
                st.markdown("**개인 정보**")
                # Example inputs - you'll need to customize based on your actual features
                user_input['AGE'] = st.selectbox("연령대", ["20대", "30대", "40대", "50대", "60대+"])
                user_input['SEX'] = st.selectbox("성별", ["남성", "여성"])
                user_input['JB_TP'] = st.selectbox("직업", ["회사원", "자영업", "공무원", "기타"])
                
            with col2:
                st.markdown("**라이프스타일**")
                user_input['CAR_YN'] = st.selectbox("차량 보유", ["Y", "N"])
                user_input['VIP_CARD_YN'] = st.selectbox("VIP 카드", ["Y", "N"])
                user_input['TRAVEL_OS'] = st.selectbox("해외여행", ["Y", "N"])
            
            submitted = st.form_submit_button("🔮 예측하기", type="primary")
            
            if submitted:
                # Convert input to dataframe
                input_df = pd.DataFrame([user_input])
                
                # Make prediction
                try:
                    # This would need to be implemented in your pipeline
                    st.info("개별 예측 기능은 파이프라인에 추가 구현이 필요합니다.")
                    
                    # Placeholder for prediction result
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("예측 등급", "A")
                    with col2:
                        st.metric("신뢰도", "85.2%")
                    with col3:
                        st.metric("위험도", "낮음")
                
                except Exception as e:
                    st.error(f"예측 중 오류가 발생했습니다: {str(e)}")

def create_batch_prediction(pipeline):
    """Create batch prediction interface"""
    st.markdown("#### 📄 배치 예측")
    
    # File upload for batch prediction
    uploaded_file = st.file_uploader(
        "예측할 데이터 파일을 업로드하세요 (CSV)",
        type=['csv'],
        help="예측하고자 하는 고객 데이터가 포함된 CSV 파일을 업로드하세요"
    )
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ 파일이 업로드되었습니다! ({len(batch_df):,}개 행)")
            
            # Display data preview
            st.markdown("##### 📊 데이터 미리보기")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # Predict button
            if st.button("🔮 배치 예측 실행", type="primary"):
                with st.spinner("예측 중..."):
                    # Placeholder for batch prediction
                    st.info("배치 예측 기능은 파이프라인에 추가 구현이 필요합니다.")
                    
                    # Mock prediction results
                    batch_df['predicted_grade'] = np.random.choice(['A', 'B', 'C', 'D'], len(batch_df))
                    batch_df['confidence'] = np.random.uniform(0.7, 0.95, len(batch_df))
                    
                    st.markdown("##### 📈 예측 결과")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    # Download results
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 결과 다운로드 (CSV)",
                        data=csv,
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )
            
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
    else:
        st.info("👆 예측할 데이터 파일을 업로드하세요.")
