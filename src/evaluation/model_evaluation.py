"""
Model evaluation module for KCB Grade prediction
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, 
    confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from typing import Dict, Any, Tuple, List
import logging

from src.config import CV_FOLDS, RANDOM_STATE

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_single_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                            model_name: str) -> Dict[str, float]:
        """단일 모델 평가"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        result = {
            'accuracy': acc,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        logger.info(f"{model_name} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
        return result
    
    def get_classification_report(self, y_true: pd.Series, y_pred: np.ndarray, 
                                model_name: str) -> pd.DataFrame:
        """분류 리포트 생성"""
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        logger.info(f"\n{model_name} - Classification Report:\n{report_df.round(4)}")
        return report_df
    
    def evaluate_overfitting(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """과적합 평가"""
        # 훈련/테스트 AUC 계산
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        # 분류 리포트
        train_report = classification_report(y_train, model.predict(X_train), output_dict=True)
        test_report = classification_report(y_test, model.predict(X_test), output_dict=True)
        
        # 차이 계산
        p_diff = train_report['weighted avg']['precision'] - test_report['weighted avg']['precision']
        r_diff = train_report['weighted avg']['recall'] - test_report['weighted avg']['recall']
        f_diff = train_report['weighted avg']['f1-score'] - test_report['weighted avg']['f1-score']
        auc_diff = train_auc - test_auc
        
        # 과적합 수준 판단
        if auc_diff < 0.02 and p_diff < 0.05 and r_diff < 0.05 and f_diff < 0.05:
            overfitting_level = "낮음"
        elif auc_diff < 0.05 or p_diff < 0.07 or r_diff < 0.07 or f_diff < 0.07:
            overfitting_level = "중간 (주의 필요)"
        else:
            overfitting_level = "높음 (심각한 과적합)"
        
        result = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'auc_diff': auc_diff,
            'precision_diff': p_diff,
            'recall_diff': r_diff,
            'f1_diff': f_diff,
            'overfitting_level': overfitting_level
        }
        
        logger.info(f"\n{model_name} 과적합 평가:")
        logger.info(f"AUC(train): {train_auc:.4f}, AUC(test): {test_auc:.4f}")
        logger.info(f"Precision Δ: {p_diff:.4f}, Recall Δ: {r_diff:.4f}, F1 Δ: {f_diff:.4f}")
        logger.info(f"과적합 수준: {overfitting_level}")
        
        return result
    
    def cross_validate_models(self, models: Dict[str, Any], X: pd.DataFrame, 
                            y: pd.Series) -> Dict[str, Dict[str, float]]:
        """교차검증 평가"""
        cv_results = {}
        
        for name, model in models.items():
            skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
            
            cv_results[name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
            
            logger.info(f"\n{name} 교차검증 결과 (ROC-AUC):")
            logger.info(f" - AUC Scores: {np.round(scores, 4)}")
            logger.info(f" - AUC 평균: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        
        return cv_results
    
    def get_roc_curve_data(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                          y_test: pd.Series) -> Dict[str, Dict[str, np.ndarray]]:
        """ROC Curve 데이터 생성"""
        roc_data = {}
        
        for name, model in models.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            
            roc_data[name] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc_score
            }
        
        return roc_data
    
    def evaluate_all_models(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                          X_test: pd.DataFrame, y_train: pd.Series, 
                          y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """모든 모델 종합 평가"""
        results = {}
        
        for name, model in models.items():
            # 기본 평가
            basic_eval = self.evaluate_single_model(model, X_test, y_test, name)
            
            # 과적합 평가
            overfitting_eval = self.evaluate_overfitting(model, X_train, X_test, y_train, y_test, name)
            
            # 분류 리포트
            classification_rep = self.get_classification_report(y_test, basic_eval['predictions'], name)
            
            results[name] = {
                'basic_metrics': basic_eval,
                'overfitting_metrics': overfitting_eval,
                'classification_report': classification_rep
            }
        
        self.evaluation_results = results
        return results
