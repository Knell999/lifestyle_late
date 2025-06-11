"""
Visualization module for KCB Grade prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, List, Optional
import logging

from src.config import FEATURE_NAME_KO

logger = logging.getLogger(__name__)


class Visualizer:
    """시각화 클래스"""
    
    def __init__(self):
        self.setup_korean_font()
    
    def setup_korean_font(self):
        """한글 폰트 설정"""
        if platform.system() == 'Darwin':  # macOS
            # 사용 가능한 한글 폰트 찾기
            font_candidates = [
                'Apple SD Gothic Neo',
                'AppleGothic',
                'NanumGothic',
                'Malgun Gothic'
            ]
            
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            for font in font_candidates:
                if font in available_fonts:
                    plt.rcParams['font.family'] = font
                    plt.rcParams['axes.unicode_minus'] = False
                    logger.info(f"한글 폰트 설정 완료: {font}")
                    return
            
            # 기본 폰트 설정
            plt.rcParams['font.family'] = 'AppleGothic'
            plt.rcParams['axes.unicode_minus'] = False
            logger.info("기본 한글 폰트 설정: AppleGothic")
    
    def plot_confusion_matrix(self, y_test: pd.Series, y_pred: np.ndarray, 
                            model_name: str, acc: float, auc: float) -> None:
        """Confusion Matrix 시각화"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["0", "1"], yticklabels=["0", "1"])
        plt.title(f"{model_name}\nConfusion Matrix\n(Accuracy: {acc:.2f}, AUC: {auc:.2f})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, roc_data: Dict[str, Dict[str, np.ndarray]], 
                       title: str = "ROC Curve - All Models") -> None:
        """ROC Curve 시각화"""
        plt.figure(figsize=(10, 7))
        
        for name, data in roc_data.items():
            plt.plot(data['fpr'], data['tpr'], 
                    label=f"{name} (AUC={data['auc']:.4f})")
        
        # 기준선 추가
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                              top_n: int = 20, title: str = "피처 중요도") -> None:
        """피처 중요도 시각화"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # 중요도 Series 생성
            importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            
            # 상위 N개 추출 및 한글 변환
            top_importance = importance_series.head(top_n).copy()
            top_importance.index = [FEATURE_NAME_KO.get(feat, feat) for feat in top_importance.index]
            
            # 시각화
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_importance.values, y=top_importance.index, palette="viridis")
            plt.title(f"{title} (Top {top_n})", fontsize=15, fontweight="bold")
            plt.yticks(fontsize=13, fontweight="bold")
            plt.xlabel("중요도", fontsize=12)
            plt.ylabel("변수명", fontsize=12)
            plt.grid(True, axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            logger.warning(f"모델 {type(model).__name__}은 feature_importances_ 속성이 없습니다.")
    
    def plot_cross_validation_results(self, cv_results: Dict[str, Dict[str, Any]], 
                                    title: str = "Model Stability Comparison (ROC-AUC)") -> None:
        """교차검증 결과 시각화"""
        # 데이터 준비
        data = {}
        for name, results in cv_results.items():
            data[name] = results['scores']
        
        df_cv = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_cv, palette='Pastel1')
        sns.swarmplot(data=df_cv, color=".25", size=8)
        
        plt.title(title, fontsize=15)
        plt.ylabel('ROC-AUC Scores', fontsize=13)
        plt.ylim(0.85, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    def plot_mode_comparison(self, fpr_data: Dict[str, np.ndarray], 
                           tpr_data: Dict[str, np.ndarray], 
                           auc_data: Dict[str, float],
                           title: str = "ROC Curve - Mode Comparison") -> None:
        """모드별 비교 시각화"""
        plt.figure(figsize=(10, 7))
        
        colors = {'life': 'green', 'fin': 'orange', 'full': 'blue'}
        labels = {'life': 'LIFE 데이터', 'fin': '금융 데이터', 'full': 'LIFE+금융 데이터'}
        
        for mode in ['life', 'fin', 'full']:
            if mode in fpr_data:
                plt.plot(fpr_data[mode], tpr_data[mode], 
                        label=f"{labels[mode]} (AUC={auc_data[mode]:.2f})", 
                        color=colors[mode])
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(title, fontsize=14)
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def create_evaluation_summary_plot(self, results: Dict[str, Dict[str, Any]]) -> None:
        """평가 결과 종합 시각화"""
        # 메트릭 추출
        models = list(results.keys())
        accuracies = [results[model]['basic_metrics']['accuracy'] for model in models]
        aucs = [results[model]['basic_metrics']['auc'] for model in models]
        
        # 서브플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy 시각화
        bars1 = ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy Comparison', fontsize=14)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        
        # 값 표시
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # AUC 시각화
        bars2 = ax2.bar(models, aucs, color='lightcoral', alpha=0.7)
        ax2.set_title('Model AUC Comparison', fontsize=14)
        ax2.set_ylabel('AUC', fontsize=12)
        ax2.set_ylim(0, 1)
        
        # 값 표시
        for bar, auc in zip(bars2, aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
