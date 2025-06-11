"""
Data preprocessing module for KCB Grade prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

from src.config import (
    ONEHOT_FEATURES, LABEL_FEATURES, BINARY_FEATURES, 
    CONTINUOUS_FEATURES, FIN_FEATURES, ONEHOT_FEATURES_LIFE,
    ONEHOT_FEATURES_FULL, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE
)

logger = logging.getLogger(__name__)


class DataLoader:
    """데이터 로딩 클래스"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_data(self) -> pd.DataFrame:
        """데이터를 로드합니다."""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"데이터 로드 완료: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            raise


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        
    def preprocess_full_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """전체 데이터 전처리"""
        df = df.copy()
        
        # 라벨 인코딩
        df['LIF_STG'] = self.label_encoder.fit_transform(df['LIF_STG'])
        
        # 전처리 파이프라인 정의
        self.preprocessor = ColumnTransformer([
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), ONEHOT_FEATURES),
            ('scaler', StandardScaler(), BINARY_FEATURES + CONTINUOUS_FEATURES + FIN_FEATURES)
        ])
        
        # 전처리 적용
        processed_array = self.preprocessor.fit_transform(df)
        
        # 컬럼명 생성
        onehot_names = self.preprocessor.named_transformers_['onehot'].get_feature_names_out(ONEHOT_FEATURES)
        scaled_names = BINARY_FEATURES + CONTINUOUS_FEATURES + FIN_FEATURES
        all_columns = np.concatenate([onehot_names, scaled_names])
        
        # DataFrame으로 변환
        processed_df = pd.DataFrame(processed_array, columns=all_columns)
        processed_df['LIF_STG'] = df['LIF_STG']
        
        X = processed_df
        y = df[TARGET_COLUMN]
        
        return X, y
    
    def preprocess_by_mode(self, df: pd.DataFrame, mode: str = 'life') -> Tuple[np.ndarray, pd.Series]:
        """모드별 데이터 전처리"""
        df = df.copy()
        df['LIF_STG'] = LabelEncoder().fit_transform(df['LIF_STG'])
        
        if mode == 'life':
            features = ONEHOT_FEATURES_LIFE
            scaler_features = BINARY_FEATURES + CONTINUOUS_FEATURES + ['LIF_STG']
            
            preprocessor = ColumnTransformer([
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), features),
                ('scaler', StandardScaler(), scaler_features)
            ])
            X = preprocessor.fit_transform(df)
            y = df[TARGET_COLUMN]
            
        elif mode == 'fin':
            df_encoded = pd.get_dummies(df, columns=['FAM_OWN_LIV_YN', 'OWN_LIV_YN'], 
                                      prefix=['FAM_OWN_LIV_YN', 'OWN_LIV_YN'], dtype=int)
            fin_col_filtered = [col for col in FIN_FEATURES if col not in ['FAM_OWN_LIV_YN', 'OWN_LIV_YN']]
            encoded_columns = [col for col in df_encoded.columns if 'FAM_OWN_LIV_YN' in col or 'OWN_LIV_YN' in col]
            X = df_encoded[fin_col_filtered + encoded_columns]
            y = df_encoded[TARGET_COLUMN]
            
        elif mode == 'full':
            features = ONEHOT_FEATURES_FULL
            scaler_features = BINARY_FEATURES + CONTINUOUS_FEATURES + FIN_FEATURES + ['LIF_STG']
            
            preprocessor = ColumnTransformer([
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), features),
                ('scaler', StandardScaler(), scaler_features)
            ])
            X = preprocessor.fit_transform(df)
            y = df[TARGET_COLUMN]
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        return X, y


class DataSplitter:
    """데이터 분할 클래스"""
    
    @staticmethod
    def split_data(X: pd.DataFrame, y: pd.Series, 
                   test_size: float = TEST_SIZE, 
                   random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """데이터를 훈련/테스트로 분할"""
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
