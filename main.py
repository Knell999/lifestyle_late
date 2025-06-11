"""
KCB Grade Prediction - Main Execution Script
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import MLPipeline, ModeComparisonPipeline
from src.utils.helpers import setup_logging


def run_full_analysis():
    """전체 분석 실행"""
    print("=" * 60)
    print("KCB Grade Prediction - 전체 분석 시작")
    print("=" * 60)
    
    # 로깅 설정
    setup_logging()
    
    # 파이프라인 실행
    pipeline = MLPipeline("full_analysis")
    summary = pipeline.run_full_pipeline()
    
    print(f"\n실험 결과가 저장되었습니다: {summary['experiment_dir']}")
    print("=" * 60)


def run_mode_comparison():
    """모드별 비교 분석 실행"""
    print("=" * 60)
    print("KCB Grade Prediction - 모드별 비교 분석 시작")
    print("=" * 60)
    
    # 로깅 설정
    setup_logging()
    
    # 모드별 비교 파이프라인 실행
    comparison_pipeline = ModeComparisonPipeline("mode_comparison")
    results = comparison_pipeline.run_mode_comparison()
    
    print("\n모드별 AUC 결과:")
    for mode, data in results.items():
        print(f"  {mode}: {data['auc']:.4f}")
    
    print("=" * 60)


def main():
    """메인 함수"""
    print("KCB Grade Prediction System")
    print("1. 전체 분석 실행")
    print("2. 모드별 비교 분석")
    print("3. 둘 다 실행")
    
    choice = input("\n선택하세요 (1/2/3): ").strip()
    
    if choice == "1":
        run_full_analysis()
    elif choice == "2":
        run_mode_comparison()
    elif choice == "3":
        run_full_analysis()
        print("\n" + "=" * 60)
        run_mode_comparison()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
