# KCB Grade Prediction Project

금융 데이터와 라이프스타일 데이터를 활용한 KCB 등급 예측 머신러닝 프로젝트

## 📋 프로젝트 개요

이 프로젝트는 고객의 금융 정보와 라이프스타일 정보를 활용하여 KCB(Korea Credit Bureau) 등급을 예측하는 머신러닝 모델을 개발합니다.

## 🏗️ 프로젝트 구조

```
lifestyle_late/
├── main.py                    # 메인 실행 파일
├── run.sh                     # 실행 스크립트
├── requirements.txt           # 의존성 패키지
├── data/                      # 데이터 폴더
│   └── df_KCB_grade.csv      # 원본 데이터
├── notebook/                  # Jupyter 노트북
│   └── [금융 원핫 다시 - 최종] 금융 + 라이프 모델.ipynb
├── src/                       # 소스 코드
│   ├── config.py             # 설정 파일
│   ├── pipeline.py           # ML 파이프라인
│   ├── preprocessing/        # 데이터 전처리
│   │   └── data_preprocessing.py
│   ├── models/               # 모델 학습
│   │   └── model_training.py
│   ├── evaluation/           # 모델 평가
│   │   └── model_evaluation.py
│   ├── visualization/        # 시각화
│   │   └── plotting.py
│   └── utils/               # 유틸리티
│       └── helpers.py
└── experiments/             # 실험 결과 (자동 생성)
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 실행

#### 방법 1: 스크립트 실행
```bash
./run.sh
```

#### 방법 2: 직접 실행
```bash
python main.py
```

## 📊 주요 기능

### 1. 데이터 전처리
- OneHot Encoding for 범주형 변수
- Standard Scaling for 연속형 변수
- Label Encoding for 타겟 변수
- 모드별 데이터 분할 (Life, Financial, Full)

### 2. 모델 학습
- **Random Forest**: 앙상블 기반 분류
- **XGBoost**: 그래디언트 부스팅
- **LightGBM**: 경량화된 그래디언트 부스팅
- **Logistic Regression**: 로지스틱 회귀
- **Ensemble**: 모든 모델의 투표 앙상블

### 3. 모델 평가
- 정확도 (Accuracy)
- ROC-AUC Score
- 분류 리포트 (Precision, Recall, F1-Score)
- 과적합 평가
- 교차검증 (5-Fold Stratified)

### 4. 시각화
- ROC Curve 비교
- Feature Importance
- Confusion Matrix
- 교차검증 결과 박스플롯
- 모드별 성능 비교

## 🔧 설정

`src/config.py`에서 다음 설정을 변경할 수 있습니다:

- 데이터 경로
- 변수 그룹 정의
- 모델 하이퍼파라미터
- 교차검증 설정

## 📈 결과

실험 결과는 `experiments/` 폴더에 자동으로 저장됩니다:

- `config.json`: 실험 설정
- `results.json`: 평가 결과
- `*.pkl`: 학습된 모델 파일

## 🎯 모델 성능

### 전체 데이터 기준 (예시)
- **Random Forest**: AUC 0.965+
- **XGBoost**: AUC 0.920+
- **LightGBM**: AUC 0.915+
- **Ensemble**: AUC 0.940+

### 모드별 비교
- **LIFE 데이터**: 라이프스타일 변수만 사용
- **금융 데이터**: 금융 변수만 사용
- **LIFE+금융**: 모든 변수 사용 (최고 성능)

## 📝 주요 특징

### 변수 그룹
- **OneHot Features**: 성별, 직업, 주거형태 등
- **Binary Features**: VIP카드, 자동차 보유 등
- **Continuous Features**: 소비금액, 관심도 등
- **Financial Features**: 대출잔액, 연체정보 등

### 과적합 방지
- 교차검증을 통한 일반화 성능 확인
- 훈련/테스트 성능 차이 모니터링
- 앙상블 기법으로 안정성 향상

## 🛠️ 개발 환경

- Python 3.8+
- scikit-learn
- XGBoost
- LightGBM
- pandas, numpy
- matplotlib, seaborn

## 📚 사용법

### 전체 분석 실행
```python
from src.pipeline import MLPipeline

pipeline = MLPipeline("my_experiment")
summary = pipeline.run_full_pipeline()
```

### 모드별 비교
```python
from src.pipeline import ModeComparisonPipeline

comparison = ModeComparisonPipeline("mode_comparison")
results = comparison.run_mode_comparison()
```

## ✨ 주요 개선사항

기존 노트북 기반 작업을 다음과 같이 모듈화했습니다:

1. **재사용성**: 각 기능이 독립적인 클래스로 분리
2. **확장성**: 새로운 모델이나 전처리 방법 쉽게 추가 가능
3. **실험 관리**: 자동화된 실험 추적 및 결과 저장
4. **코드 품질**: 로깅, 에러 처리, 타입 힌트 적용
5. **시각화**: 자동화된 그래프 생성 및 한글 폰트 지원