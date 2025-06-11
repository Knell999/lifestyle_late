"""
ML Pipeline for KCB Grade prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

from src.config import DATA_PATH
from src.preprocessing.data_preprocessing import DataLoader, DataPreprocessor, DataSplitter
from src.models.model_training import ModelTrainer, SingleModelTrainer
from src.evaluation.model_evaluation import ModelEvaluator
from src.visualization.plotting import Visualizer
from src.utils.helpers import ExperimentTracker, setup_logging

logger = logging.getLogger(__name__)


class MLPipeline:
    """머신러닝 파이프라인 클래스"""
    
    def __init__(self, experiment_name: str = "kcb_grade_prediction"):
        self.experiment_name = experiment_name
        self.tracker = ExperimentTracker(experiment_name)
        self.data_loader = DataLoader(DATA_PATH)
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        
        # 결과 저장용
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = None
        
    def set_dataframe(self, df: pd.DataFrame):
        """외부에서 데이터프레임 설정"""
        self.df = df
        logger.info(f"외부 데이터프레임 설정됨 - 형태: {df.shape}")
    
    def load_and_preprocess_data(self, use_external_data: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """데이터 로드 및 전처리"""
        logger.info("데이터 로드 및 전처리 시작...")
        
        # 데이터 로드
        if use_external_data and self.df is not None:
            logger.info("외부 데이터프레임 사용")
            data = self.df
        else:
            logger.info("기본 데이터 소스에서 데이터 로드")
            data = self.data_loader.load_data()
        
        # 전처리
        X, y = self.preprocessor.preprocess_full_data(data)
        
        # 데이터 분할
        self.X_train, self.X_test, self.y_train, self.y_test = DataSplitter.split_data(X, y)
        
        logger.info(f"데이터 분할 완료 - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self) -> Dict[str, Any]:
        """모델 학습"""
        logger.info("모델 학습 시작...")
        
        self.models = self.trainer.train_all_models(self.X_train, self.y_train)
        
        # 모델 저장
        for name, model in self.models.items():
            self.tracker.save_model(model, name)
        
        return self.models
    
    def evaluate_models(self) -> Dict[str, Dict[str, Any]]:
        """모델 평가"""
        logger.info("모델 평가 시작...")
        
        # 기본 평가
        results = self.evaluator.evaluate_all_models(
            self.models, self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        # 교차검증
        cv_results = self.evaluator.cross_validate_models(
            self.models, pd.concat([self.X_train, self.X_test]), 
            pd.concat([self.y_train, self.y_test])
        )
        
        # ROC 데이터
        roc_data = self.evaluator.get_roc_curve_data(self.models, self.X_test, self.y_test)
        
        # 결과 로깅
        self.tracker.log_results({
            'evaluation_results': results,
            'cross_validation_results': cv_results,
            'roc_data': roc_data
        })
        
        return results, cv_results, roc_data
    
    def create_visualizations(self, results: Dict[str, Dict[str, Any]], 
                            cv_results: Dict[str, Dict[str, Any]], 
                            roc_data: Dict[str, Dict[str, np.ndarray]]) -> None:
        """시각화 생성"""
        logger.info("시각화 생성 시작...")
        
        # ROC Curve
        self.visualizer.plot_roc_curves(roc_data)
        
        # 교차검증 결과
        self.visualizer.plot_cross_validation_results(cv_results)
        
        # 피처 중요도 (RandomForest)
        if 'RandomForest' in self.models:
            self.visualizer.plot_feature_importance(
                self.models['RandomForest'], 
                self.X_train.columns.tolist(),
                title="랜덤포레스트 피처 중요도"
            )
        
        # 평가 결과 요약
        self.visualizer.create_evaluation_summary_plot(results)
        
        # Confusion Matrix (각 모델별)
        for name, model in self.models.items():
            eval_result = results[name]['basic_metrics']
            self.visualizer.plot_confusion_matrix(
                self.y_test, eval_result['predictions'], 
                name, eval_result['accuracy'], eval_result['auc']
            )
    
    def run_full_pipeline(self, use_external_data: bool = False) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        logger.info("전체 파이프라인 실행 시작...")
        
        # 1. 데이터 로드 및 전처리
        self.load_and_preprocess_data(use_external_data=use_external_data)
        
        # 2. 모델 학습
        self.train_models()
        
        # 3. 모델 평가
        results, cv_results, roc_data = self.evaluate_models()
        
        # 4. 시각화
        self.create_visualizations(results, cv_results, roc_data)
        
        # 5. 실험 요약
        summary = self.tracker.get_experiment_summary()
        logger.info("전체 파이프라인 실행 완료!")
        
        return summary


class ModeComparisonPipeline:
    """모드별 비교 파이프라인"""
    
    def __init__(self, experiment_name: str = "mode_comparison"):
        self.experiment_name = experiment_name
        self.tracker = ExperimentTracker(experiment_name)
        self.data_loader = DataLoader(DATA_PATH)
        self.preprocessor = DataPreprocessor()
        self.visualizer = Visualizer()
        
    def run_mode_comparison(self) -> Dict[str, Any]:
        """모드별 비교 실행"""
        logger.info("모드별 비교 파이프라인 실행 시작...")
        
        # 데이터 로드
        df = self.data_loader.load_data()
        
        results = {}
        fpr_data = {}
        tpr_data = {}
        auc_data = {}
        
        modes = ['life', 'fin', 'full']
        
        for mode in modes:
            logger.info(f"{mode} 모드 처리 중...")
            
            # 전처리
            X, y = self.preprocessor.preprocess_by_mode(df, mode)
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = DataSplitter.split_data(
                pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X, y
            )
            
            # 모델 학습
            model = SingleModelTrainer.grid_search_rf_simple(X_train, y_train)
            
            # 예측 및 평가
            y_prob = model.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_auc_score, roc_curve
            auc_score = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            # 결과 저장
            results[mode] = {
                'auc': auc_score,
                'model': model
            }
            fpr_data[mode] = fpr
            tpr_data[mode] = tpr
            auc_data[mode] = auc_score
            
            logger.info(f"{mode} 모드 AUC: {auc_score:.4f}")
        
        # 시각화
        self.visualizer.plot_mode_comparison(fpr_data, tpr_data, auc_data)
        
        # 결과 로깅
        self.tracker.log_results({
            'mode_comparison_results': {mode: {'auc': data['auc']} for mode, data in results.items()},
            'fpr_data': fpr_data,
            'tpr_data': tpr_data,
            'auc_data': auc_data
        })
        
        logger.info("모드별 비교 파이프라인 실행 완료!")
        
        return results
