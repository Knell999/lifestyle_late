"""
Utility functions for KCB Grade prediction project
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict
import json
import pickle


def setup_logging(level: str = "INFO") -> None:
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def save_model(model: Any, filepath: str) -> None:
    """모델 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"모델 저장 완료: {filepath}")


def load_model(filepath: str) -> Any:
    """모델 로드"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logging.info(f"모델 로드 완료: {filepath}")
    return model


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """결과 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # numpy 배열을 리스트로 변환
    def convert_numpy(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    logging.info(f"결과 저장 완료: {filepath}")


def create_experiment_dir(base_dir: str = "experiments") -> str:
    """실험 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"실험 디렉토리 생성: {exp_dir}")
    return exp_dir


class ExperimentTracker:
    """실험 추적 클래스"""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.exp_dir = create_experiment_dir(base_dir)
        self.results = {}
        self.config = {}
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """설정 로깅"""
        self.config = config
        config_path = os.path.join(self.exp_dir, "config.json")
        save_results(config, config_path)
    
    def log_results(self, results: Dict[str, Any]) -> None:
        """결과 로깅"""
        self.results.update(results)
        results_path = os.path.join(self.exp_dir, "results.json")
        save_results(self.results, results_path)
    
    def save_model(self, model: Any, model_name: str) -> None:
        """모델 저장"""
        model_path = os.path.join(self.exp_dir, f"{model_name}.pkl")
        save_model(model, model_path)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """실험 요약 정보"""
        return {
            "experiment_name": self.experiment_name,
            "experiment_dir": self.exp_dir,
            "config": self.config,
            "results_summary": {
                key: value for key, value in self.results.items() 
                if not isinstance(value, (list, dict))
            }
        }
