"""
Visualization components for Streamlit app
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_visualization_section(config: Dict[str, Any]):
    """Create comprehensive visualization section"""
    st.title("📊 데이터 시각화")
    
    # Check if data is available
    if 'data' not in st.session_state:
        st.warning("⚠️ 먼저 '데이터 업로드' 페이지에서 데이터를 업로드하세요.")
        return
    
    df = st.session_state.data
    
    # Visualization options
    st.markdown("### 🎨 시각화 옵션")
    
    viz_type = st.selectbox(
        "시각화 유형을 선택하세요:",
        [
            "데이터 분포 분석",
            "타겟 변수 분석",
            "상관관계 분석",
            "특성별 분포",
            "모델 성능 시각화"
        ]
    )
    
    if viz_type == "데이터 분포 분석":
        create_distribution_analysis(df)
    elif viz_type == "타겟 변수 분석":
        create_target_analysis(df)
    elif viz_type == "상관관계 분석":
        create_correlation_analysis(df)
    elif viz_type == "특성별 분포":
        create_feature_analysis(df)
    elif viz_type == "모델 성능 시각화":
        create_model_performance_visualization()

def create_distribution_analysis(df: pd.DataFrame):
    """Create data distribution analysis"""
    st.markdown("#### 📈 데이터 분포 분석")
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**수치형 변수 개수**")
        st.metric("", len(numeric_cols))
    
    with col2:
        st.markdown("**범주형 변수 개수**")
        st.metric("", len(categorical_cols))
    
    # Numeric variables distribution
    if numeric_cols:
        st.markdown("##### 📊 수치형 변수 분포")
        
        selected_numeric = st.selectbox(
            "분석할 수치형 변수를 선택하세요:",
            numeric_cols
        )
        
        if selected_numeric:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    df, 
                    x=selected_numeric, 
                    title=f"{selected_numeric} 분포",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    df, 
                    y=selected_numeric, 
                    title=f"{selected_numeric} 박스 플롯"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("**기술통계량**")
            stats = df[selected_numeric].describe()
            stats_df = pd.DataFrame(stats).T
            st.dataframe(stats_df, use_container_width=True)
    
    # Categorical variables distribution
    if categorical_cols:
        st.markdown("##### 🏷️ 범주형 변수 분포")
        
        selected_categorical = st.selectbox(
            "분석할 범주형 변수를 선택하세요:",
            categorical_cols
        )
        
        if selected_categorical:
            # Value counts
            value_counts = df[selected_categorical].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"{selected_categorical} 분포",
                    labels={'x': selected_categorical, 'y': '개수'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"{selected_categorical} 비율"
                )
                st.plotly_chart(fig, use_container_width=True)

def create_target_analysis(df: pd.DataFrame):
    """Create target variable analysis"""
    st.markdown("#### 🎯 타겟 변수 분석")
    
    # Import TARGET_COLUMN with proper path handling
    try:
        try:
            from src.config import TARGET_COLUMN
        except ImportError:
            # Fallback to relative path
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
            try:
                from src.config import TARGET_COLUMN
            except ImportError:
                # Use default target column name
                TARGET_COLUMN = "KCB_grade"
    except Exception:
        TARGET_COLUMN = "KCB_grade"
    
    if TARGET_COLUMN not in df.columns:
        st.error(f"타겟 변수 '{TARGET_COLUMN}'를 찾을 수 없습니다.")
        # Try to find similar column names
        possible_targets = [col for col in df.columns if 'grade' in col.lower() or 'target' in col.lower()]
        if possible_targets:
            st.info(f"가능한 타겟 변수: {', '.join(possible_targets)}")
        return
    
    # Target distribution
    target_dist = df[TARGET_COLUMN].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target distribution chart
        fig = px.bar(
            x=target_dist.index,
            y=target_dist.values,
            title="KCB 등급 분포",
            labels={'x': 'KCB 등급', 'y': '개수'},
            color=target_dist.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Target distribution pie
        fig = px.pie(
            values=target_dist.values,
            names=target_dist.index,
            title="KCB 등급 비율"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Target vs features analysis
    st.markdown("##### 📊 등급별 특성 분석")
    
    # Select feature for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)
    
    if numeric_cols:
        selected_feature = st.selectbox(
            "분석할 특성을 선택하세요:",
            numeric_cols
        )
        
        if selected_feature:
            # Box plot by target
            fig = px.box(
                df,
                x=TARGET_COLUMN,
                y=selected_feature,
                title=f"등급별 {selected_feature} 분포"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics by target
            grouped_stats = df.groupby(TARGET_COLUMN)[selected_feature].describe()
            st.dataframe(grouped_stats, use_container_width=True)

def create_correlation_analysis(df: pd.DataFrame):
    """Create correlation analysis"""
    st.markdown("#### 🔗 상관관계 분석")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        st.warning("상관관계 분석을 위한 수치형 변수가 없습니다.")
        return
    
    # Correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Correlation heatmap
    fig = px.imshow(
        corr_matrix,
        title="상관관계 히트맵",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # High correlation pairs
    st.markdown("##### 🔍 높은 상관관계 변수 쌍")
    
    # Find high correlations
    threshold = st.slider("상관계수 임계값", 0.5, 0.9, 0.7, 0.05)
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    '변수1': corr_matrix.columns[i],
                    '변수2': corr_matrix.columns[j],
                    '상관계수': corr_val
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('상관계수', key=abs, ascending=False)
        st.dataframe(high_corr_df, use_container_width=True)
    else:
        st.info(f"상관계수가 {threshold} 이상인 변수 쌍이 없습니다.")

def create_feature_analysis(df: pd.DataFrame):
    """Create feature-specific analysis"""
    st.markdown("#### 🔍 특성별 상세 분석")
    
    # Feature selection
    all_columns = df.columns.tolist()
    selected_features = st.multiselect(
        "분석할 특성들을 선택하세요:",
        all_columns,
        default=all_columns[:5] if len(all_columns) >= 5 else all_columns
    )
    
    if not selected_features:
        st.warning("분석할 특성을 선택해주세요.")
        return
    
    # Analysis type
    analysis_type = st.radio(
        "분석 유형:",
        ["개별 분포", "상호 비교", "타겟과의 관계"],
        horizontal=True
    )
    
    if analysis_type == "개별 분포":
        # Individual feature distributions
        for feature in selected_features:
            st.markdown(f"**{feature} 분포**")
            
            if df[feature].dtype in ['object', 'category']:
                # Categorical
                value_counts = df[feature].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"{feature} 분포"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Numeric
                fig = px.histogram(df, x=feature, title=f"{feature} 분포")
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "상호 비교" and len(selected_features) >= 2:
        # Pairwise comparison
        if len(selected_features) > 5:
            st.warning("성능을 위해 처음 5개 특성만 표시합니다.")
            selected_features = selected_features[:5]
        
        # Create pairplot
        sample_size = min(1000, len(df))
        sample_df = df[selected_features].sample(sample_size)
        
        fig = px.scatter_matrix(sample_df, title="특성 간 관계")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "타겟과의 관계":
        # Import TARGET_COLUMN with proper path handling
        try:
            try:
                from src.config import TARGET_COLUMN
            except ImportError:
                # Fallback to relative path
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
                try:
                    from src.config import TARGET_COLUMN
                except ImportError:
                    # Use default target column name
                    TARGET_COLUMN = "KCB_grade"
        except Exception:
            TARGET_COLUMN = "KCB_grade"
        
        if TARGET_COLUMN not in df.columns:
            st.error(f"타겟 변수 '{TARGET_COLUMN}'를 찾을 수 없습니다.")
            return
        
        for feature in selected_features:
            if feature == TARGET_COLUMN:
                continue
            
            st.markdown(f"**{feature} vs {TARGET_COLUMN}**")
            
            if df[feature].dtype in ['object', 'category']:
                # Categorical vs target
                crosstab = pd.crosstab(df[feature], df[TARGET_COLUMN], normalize='index')
                fig = px.bar(
                    crosstab.reset_index(),
                    x=feature,
                    y=crosstab.columns.tolist(),
                    title=f"{feature}별 {TARGET_COLUMN} 분포"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Numeric vs target
                fig = px.box(
                    df,
                    x=TARGET_COLUMN,
                    y=feature,
                    title=f"{TARGET_COLUMN}별 {feature} 분포"
                )
                st.plotly_chart(fig, use_container_width=True)

def create_model_performance_visualization():
    """Create model performance visualization"""
    st.markdown("#### 🤖 모델 성능 시각화")
    
    # Check if training results are available
    if 'training_results' not in st.session_state:
        st.warning("⚠️ 먼저 '모델 훈련' 페이지에서 모델을 훈련시키세요.")
        return
    
    results = st.session_state.training_results
    
    if 'model_results' not in results:
        st.warning("모델 결과를 찾을 수 없습니다.")
        return
    
    model_results = results['model_results']
    
    # Performance comparison
    st.markdown("##### 📊 모델 성능 비교")
    
    # Create performance dataframe
    performance_data = []
    for model_name, metrics in model_results.items():
        performance_data.append({
            '모델': model_name,
            '정확도': metrics.get('accuracy', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
            '정밀도': metrics.get('precision', 0),
            '재현율': metrics.get('recall', 0)
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Radar chart
    fig = go.Figure()
    
    metrics = ['정확도', 'F1-Score', 'ROC-AUC', '정밀도', '재현율']
    
    for _, row in performance_df.iterrows():
        values = [row[metric] for metric in metrics]
        values += [values[0]]  # Close the radar chart
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=row['모델']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="모델 성능 레이더 차트"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available)
    if 'feature_importance' in results:
        st.markdown("##### 📈 특성 중요도")
        
        importance_df = pd.DataFrame(results['feature_importance'])
        if not importance_df.empty:
            top_features = importance_df.head(15)
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title="상위 15개 특성 중요도",
                color='importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
