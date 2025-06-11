"""
Quick test script for the modularized pipeline
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.data_preprocessing import DataLoader, DataPreprocessor
from src.config import DATA_PATH


def test_data_loading():
    """데이터 로딩 테스트"""
    print("=" * 50)
    print("데이터 로딩 테스트")
    print("=" * 50)
    
    try:
        loader = DataLoader(DATA_PATH)
        df = loader.load_data()
        print(f"✅ 데이터 로드 성공: {df.shape}")
        print(f"컬럼 수: {len(df.columns)}")
        print(f"첫 번째 컬럼들: {list(df.columns[:5])}")
        return df
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None


def test_preprocessing(df):
    """전처리 테스트"""
    print("\n" + "=" * 50)
    print("전처리 테스트")
    print("=" * 50)
    
    try:
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_full_data(df)
        print(f"✅ 전처리 성공")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"y 클래스 분포:")
        print(y.value_counts().sort_index())
        return X, y
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")
        return None, None


def test_mode_preprocessing(df):
    """모드별 전처리 테스트"""
    print("\n" + "=" * 50)
    print("모드별 전처리 테스트")
    print("=" * 50)
    
    preprocessor = DataPreprocessor()
    modes = ['life', 'fin', 'full']
    
    for mode in modes:
        try:
            X, y = preprocessor.preprocess_by_mode(df, mode)
            print(f"✅ {mode} 모드 전처리 성공: X shape {X.shape}")
        except Exception as e:
            print(f"❌ {mode} 모드 전처리 실패: {e}")


def main():
    """메인 테스트 함수"""
    print("🚀 KCB Grade Prediction Pipeline 모듈 테스트")
    
    # 데이터 로딩 테스트
    df = test_data_loading()
    if df is None:
        return
    
    # 전처리 테스트
    X, y = test_preprocessing(df)
    if X is None or y is None:
        return
    
    # 모드별 전처리 테스트
    test_mode_preprocessing(df)
    
    print("\n" + "=" * 50)
    print("✅ 모든 모듈 테스트 완료!")
    print("이제 main.py를 실행하여 전체 파이프라인을 실행할 수 있습니다.")
    print("=" * 50)


if __name__ == "__main__":
    main()
