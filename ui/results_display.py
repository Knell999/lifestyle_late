"""
Results display and model comparison components
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go

def create_results_display(results: Dict[str, Any]):
    """Display model training results"""
    st.markdown("### 📊 모델 성능 결과")
    
    if not results or 'model_results' not in results:
        st.warning("표시할 결과가 없습니다.")
        return
    
    # Performance metrics table
    create_performance_table(results['model_results'])
    
    # Performance comparison charts
    create_performance_charts(results['model_results'])
    
    # Feature importance
    if 'feature_importance' in results:
        create_feature_importance_chart(results['feature_importance'])

def create_performance_table(model_results: Dict[str, Any]):
    """Create performance comparison table"""
    st.markdown("#### 🏆 성능 지표 비교")
    
    performance_data = []
    for model_name, metrics in model_results.items():
        performance_data.append({
            '모델': model_name,
            '정확도': metrics.get('accuracy', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
            '정밀도': metrics.get('precision', 0),
            '재현율': metrics.get('recall', 0),
            'CV 평균': np.mean(metrics.get('cv_scores', [0])),
            'CV 표준편차': np.std(metrics.get('cv_scores', [0]))
        })
    
    df = pd.DataFrame(performance_data)
    
    # Style the dataframe
    styled_df = df.style.format({
        '정확도': '{:.4f}',
        'F1-Score': '{:.4f}',
        'ROC-AUC': '{:.4f}',
        '정밀도': '{:.4f}',
        '재현율': '{:.4f}',
        'CV 평균': '{:.4f}',
        'CV 표준편차': '{:.4f}'
    }).background_gradient(subset=['정확도', 'F1-Score', 'ROC-AUC'], cmap='RdYlGn')
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Highlight best model
    best_model_idx = df['정확도'].idxmax()
    best_model = df.iloc[best_model_idx]['모델']
    best_accuracy = df.iloc[best_model_idx]['정확도']
    
    st.success(f"🎯 **최고 성능**: {best_model} (정확도: {best_accuracy:.4f})")

def create_performance_charts(model_results: Dict[str, Any]):
    """Create performance comparison charts"""
    st.markdown("#### 📈 성능 비교 차트")
    
    # Prepare data for plotting
    models = list(model_results.keys())
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision', 'recall']
    metric_names = ['정확도', 'F1-Score', 'ROC-AUC', '정밀도', '재현율']
    
    # Create tabs for different chart types
    tab1, tab2, tab3 = st.tabs(["막대 차트", "레이더 차트", "박스 플롯"])
    
    with tab1:
        # Bar chart comparison
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [model_results[model].get(metric, 0) for model in models]
            
            fig = px.bar(
                x=models,
                y=values,
                title=f"{metric_name} 비교",
                labels={'x': '모델', 'y': metric_name},
                color=values,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Radar chart
        fig = go.Figure()
        
        for model in models:
            values = [model_results[model].get(metric, 0) for metric in metrics]
            values += [values[0]]  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names + [metric_names[0]],
                fill='toself',
                name=model
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
    
    with tab3:
        # Box plot for CV scores
        cv_data = []
        for model in models:
            cv_scores = model_results[model].get('cv_scores', [])
            for score in cv_scores:
                cv_data.append({'모델': model, 'CV Score': score})
        
        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            fig = px.box(cv_df, x='모델', y='CV Score', title="교차검증 점수 분포")
            st.plotly_chart(fig, use_container_width=True)

def create_feature_importance_chart(feature_importance: pd.DataFrame):
    """Create feature importance visualization"""
    st.markdown("#### 📊 특성 중요도")
    
    if feature_importance.empty:
        st.info("특성 중요도 데이터가 없습니다.")
        return
    
    # Top 20 features
    top_features = feature_importance.head(20)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title="상위 20개 특성 중요도",
        labels={'importance': '중요도', 'feature': '특성'},
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Show full table in expander
    with st.expander("전체 특성 중요도 보기"):
        st.dataframe(feature_importance, use_container_width=True)

def create_model_comparison_section(config: Dict[str, Any]):
    """Create comprehensive model comparison section"""
    st.title("🔬 모델 비교 분석")
    
    # Check if results are available
    if 'training_results' not in st.session_state:
        st.warning("⚠️ 먼저 '모델 훈련' 페이지에서 모델을 훈련시키세요.")
        return
    
    results = st.session_state.training_results
    
    # Overview metrics
    st.markdown("### 📋 모델 성능 개요")
    
    if 'model_results' in results:
        model_results = results['model_results']
        
        # Create metric cards
        cols = st.columns(len(model_results))
        
        for i, (model_name, metrics) in enumerate(model_results.items()):
            with cols[i]:
                accuracy = metrics.get('accuracy', 0)
                f1_score = metrics.get('f1_score', 0)
                
                st.metric(
                    label=model_name,
                    value=f"{accuracy:.4f}",
                    delta=f"F1: {f1_score:.4f}"
                )
        
        # Detailed comparison
        create_results_display(results)
        
        # Model analysis
        st.markdown("### 🔍 상세 분석")
        
        selected_models = st.multiselect(
            "비교할 모델을 선택하세요:",
            list(model_results.keys()),
            default=list(model_results.keys())[:2]
        )
        
        if len(selected_models) >= 2:
            create_model_comparison_analysis(model_results, selected_models)

def create_model_comparison_analysis(model_results: Dict[str, Any], selected_models: list):
    """Create detailed model comparison analysis"""
    
    # Performance difference analysis
    st.markdown("#### 📊 성능 차이 분석")
    
    comparison_data = []
    for model in selected_models:
        metrics = model_results[model]
        comparison_data.append({
            '모델': model,
            '정확도': metrics.get('accuracy', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
            'CV 평균': np.mean(metrics.get('cv_scores', [0])),
            'CV 표준편차': np.std(metrics.get('cv_scores', [0]))
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate performance differences
    if len(comparison_df) == 2:
        diff_data = {}
        for metric in ['정확도', 'F1-Score', 'ROC-AUC']:
            diff = comparison_df.iloc[1][metric] - comparison_df.iloc[0][metric]
            diff_data[f'{metric} 차이'] = f"{diff:+.4f}"
        
        st.markdown("**성능 차이 (모델2 - 모델1):**")
        for metric, diff in diff_data.items():
            if float(diff) > 0:
                st.success(f"{metric}: {diff}")
            else:
                st.error(f"{metric}: {diff}")
    
    # Statistical significance test (placeholder)
    st.markdown("#### 📈 통계적 유의성")
    st.info("통계적 유의성 검정 결과는 추가 구현이 필요합니다.")
    
    # Recommendations
    st.markdown("#### 💡 모델 선택 권장사항")
    
    best_model = comparison_df.loc[comparison_df['정확도'].idxmax(), '모델']
    most_stable = comparison_df.loc[comparison_df['CV 표준편차'].idxmin(), '모델']
    
    recommendations = [
        f"🏆 **최고 성능**: {best_model}가 가장 높은 정확도를 보입니다.",
        f"📊 **안정성**: {most_stable}가 가장 안정적인 성능을 보입니다.",
        "⚖️ **권장**: 성능과 안정성을 모두 고려하여 모델을 선택하세요."
    ]
    
    for rec in recommendations:
        st.markdown(rec)
