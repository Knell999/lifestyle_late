"""
Data upload and preprocessing components
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_data_upload_section():
    """Create data upload and exploration section"""
    st.title("📂 데이터 업로드 및 탐색")
    
    # File upload
    st.markdown("### 📤 파일 업로드")
    uploaded_file = st.file_uploader(
        "CSV 파일을 선택하세요",
        type=['csv'],
        help="KCB 등급 예측에 필요한 데이터가 포함된 CSV 파일을 업로드하세요"
    )
    
    # Load default data if no file uploaded
    use_default = st.checkbox("기본 데이터 사용", help="프로젝트에 포함된 기본 데이터셋을 사용합니다")
    
    df = None
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ 파일이 성공적으로 업로드되었습니다! ({len(df):,}개 행, {len(df.columns)}개 열)")
        except Exception as e:
            st.error(f"❌ 파일 읽기 오류: {str(e)}")
            return None
    
    elif use_default:
        try:
            # Import config with proper path handling
            try:
                from src.config import DATA_PATH
            except ImportError:
                # Fallback to relative path
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
                from src.config import DATA_PATH
            
            # Try different possible paths for the data file
            possible_paths = [
                DATA_PATH,
                os.path.join('data', 'df_KCB_grade.csv'),
                os.path.join('..', 'data', 'df_KCB_grade.csv'),
                os.path.join(os.path.dirname(__file__), '..', 'data', 'df_KCB_grade.csv')
            ]
            
            df = None
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        break
                except:
                    continue
            
            if df is not None:
                st.success(f"✅ 기본 데이터가 로드되었습니다! ({len(df):,}개 행, {len(df.columns)}개 열)")
            else:
                st.error("❌ 기본 데이터 파일을 찾을 수 없습니다. 파일을 업로드해주세요.")
                return None
        except Exception as e:
            st.error(f"❌ 기본 데이터 로드 오류: {str(e)}")
            return None
    
    if df is not None:
        # Store in session state
        st.session_state.data = df
        
        # Data overview
        st.markdown("### 📊 데이터 개요")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 행 수", f"{len(df):,}")
        with col2:
            st.metric("총 열 수", f"{len(df.columns):,}")
        with col3:
            st.metric("메모리 사용량", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        with col4:
            missing_count = df.isnull().sum().sum()
            st.metric("결측값", f"{missing_count:,}")
        
        # Data preview
        st.markdown("### 👀 데이터 미리보기")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data info tabs
        tab1, tab2, tab3, tab4 = st.tabs(["기본 정보", "통계 요약", "결측값 분석", "데이터 타입"])
        
        with tab1:
            st.markdown("#### 📋 기본 정보")
            buffer = []
            buffer.append(f"데이터 형태: {df.shape}")
            buffer.append(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            buffer.append(f"중복된 행: {df.duplicated().sum():,}개")
            
            for info in buffer:
                st.text(info)
        
        with tab2:
            st.markdown("#### 📈 수치형 변수 통계 요약")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.info("수치형 변수가 없습니다.")
        
        with tab3:
            st.markdown("#### ❓ 결측값 분석")
            missing_info = df.isnull().sum()
            missing_percent = (missing_info / len(df)) * 100
            
            missing_df = pd.DataFrame({
                '열 이름': missing_info.index,
                '결측값 개수': missing_info.values,
                '결측값 비율(%)': missing_percent.values
            })
            missing_df = missing_df[missing_df['결측값 개수'] > 0].sort_values('결측값 개수', ascending=False)
            
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ 결측값이 없습니다!")
        
        with tab4:
            st.markdown("#### 🏷️ 데이터 타입 정보")
            dtype_info = pd.DataFrame({
                '열 이름': df.columns,
                '데이터 타입': df.dtypes.values,
                '고유값 개수': [df[col].nunique() for col in df.columns],
                '예시 값': [str(df[col].iloc[0]) if not pd.isna(df[col].iloc[0]) else 'NaN' for col in df.columns]
            })
            st.dataframe(dtype_info, use_container_width=True)
        
        # Target variable analysis if available
        try:
            # Import TARGET_COLUMN with proper path handling
            try:
                from src.config import TARGET_COLUMN
            except ImportError:
                # Fallback to relative path
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
                from src.config import TARGET_COLUMN
        except ImportError:
            # If config is not available, use default target column name
            TARGET_COLUMN = "KCB_grade"
        
        if TARGET_COLUMN in df.columns:
            st.markdown("### 🎯 타겟 변수 분석")
            target_dist = df[TARGET_COLUMN].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 등급 분포")
                st.dataframe(
                    pd.DataFrame({
                        '등급': target_dist.index,
                        '개수': target_dist.values,
                        '비율(%)': (target_dist.values / len(df)) * 100
                    }),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 📈 등급 분포 차트")
                st.bar_chart(target_dist)
        
        # Data quality assessment
        st.markdown("### 🔍 데이터 품질 평가")
        
        quality_checks = []
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_checks.append(f"⚠️ 중복된 행이 {duplicates}개 발견되었습니다.")
        else:
            quality_checks.append("✅ 중복된 행이 없습니다.")
        
        # Check for missing values
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            quality_checks.append(f"⚠️ 총 {total_missing}개의 결측값이 있습니다.")
        else:
            quality_checks.append("✅ 결측값이 없습니다.")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            quality_checks.append(f"⚠️ 상수 열이 발견되었습니다: {', '.join(constant_cols)}")
        else:
            quality_checks.append("✅ 모든 열이 적절한 변동성을 가지고 있습니다.")
        
        for check in quality_checks:
            if "⚠️" in check:
                st.warning(check)
            else:
                st.success(check)
        
        return df
    
    else:
        st.info("👆 위에서 CSV 파일을 업로드하거나 기본 데이터를 선택하세요.")
        return None
