"""
Model training module for KCB Grade prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any, List, Tuple
import logging

from src.config import MODEL_PARAMS, RANDOM_STATE, N_JOBS

logger = logging.getLogger(__name__)


class ModelTrainer:
    """모델 학습 클래스"""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        
    def grid_search_rf(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """Random Forest 그리드서치"""
        params = MODEL_PARAMS['random_forest']
        clf = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE), 
            params, 
            cv=3, 
            scoring='roc_auc', 
            n_jobs=N_JOBS
        )
        clf.fit(X, y)
        logger.info(f"RF best params: {clf.best_params_}")
        return clf.best_estimator_
    
    def grid_search_xgb(self, X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
        """XGBoost 그리드서치"""
        params = MODEL_PARAMS['xgboost']
        clf = GridSearchCV(
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
            params, 
            cv=3, 
            scoring='roc_auc', 
            n_jobs=N_JOBS
        )
        clf.fit(X, y)
        logger.info(f"XGB best params: {clf.best_params_}")
        return clf.best_estimator_
    
    def grid_search_lgbm(self, X: pd.DataFrame, y: pd.Series) -> LGBMClassifier:
        """LightGBM 그리드서치"""
        params = MODEL_PARAMS['lightgbm']
        clf = GridSearchCV(
            LGBMClassifier(random_state=RANDOM_STATE), 
            params, 
            cv=3, 
            scoring='roc_auc', 
            n_jobs=N_JOBS
        )
        clf.fit(X, y)
        logger.info(f"LGBM best params: {clf.best_params_}")
        return clf.best_estimator_
    
    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
        """Logistic Regression 학습"""
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X, y)
        return model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """모든 모델 학습"""
        logger.info("모델 학습 시작...")
        
        # 개별 모델 학습
        rf = self.grid_search_rf(X_train, y_train)
        xgb = self.grid_search_xgb(X_train, y_train)
        lgbm = self.grid_search_lgbm(X_train, y_train)
        logreg = self.train_logistic_regression(X_train, y_train)
        
        # 앙상블 모델
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf), 
                ('xgb', xgb), 
                ('lgbm', lgbm), 
                ('logreg', logreg)
            ], 
            voting='soft'
        )
        ensemble.fit(X_train, y_train)
        
        models = {
            'RandomForest': rf,
            'XGBoost': xgb,
            'LightGBM': lgbm,
            'LogisticRegression': logreg,
            'Ensemble': ensemble
        }
        
        self.best_models = models
        logger.info("모든 모델 학습 완료")
        
        return models


class SingleModelTrainer:
    """단일 모델 학습 클래스 (모드별)"""
    
    @staticmethod
    def grid_search_rf_simple(X: np.ndarray, y: pd.Series) -> RandomForestClassifier:
        """간단한 Random Forest 그리드서치"""
        params = {
            'n_estimators': [100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        clf = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE), 
            params, 
            cv=3, 
            scoring='roc_auc', 
            n_jobs=N_JOBS
        )
        clf.fit(X, y)
        logger.info(f"RF Best Params: {clf.best_params_}")
        return clf.best_estimator_
