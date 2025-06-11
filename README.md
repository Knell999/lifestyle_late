# KCB Grade Prediction Project

금융 데이터와 라이프스타일 데이터를 활용한 KCB 등급 예측 머신러닝 프로젝트

## 📋 프로젝트 개요

이 프로젝트는 고객의 금융 정보와 라이프스타일 정보를 활용하여 KCB(Korea Credit Bureau) 등급을 예측하는 머신러닝 모델을 개발합니다.

**주요 특징:**
- 🔧 **모듈화된 아키텍처**: 재사용 가능한 컴포넌트로 구성
- 🤖 **다중 ML 모델**: Random Forest, XGBoost, LightGBM, Ensemble
- 🎯 **자동화된 파이프라인**: 데이터 전처리부터 모델 평가까지 자동화
- 📊 **웹 인터페이스**: Streamlit 기반 대화형 웹 앱
- 📈 **시각화**: 한글 폰트 지원으로 완벽한 데이터 시각화

## 🏗️ 프로젝트 구조

```
lifestyle_late/
├── streamlit_app.py          # 🌐 Streamlit 웹 애플리케이션
├── main.py                   # 📋 메인 실행 파일 (CLI)
├── run_streamlit_uv.sh       # 🚀 Streamlit 앱 실행 스크립트 (UV 환경)
├── run.sh                    # 🔧 기본 실행 스크립트
├── requirements.txt          # 📦 의존성 패키지
├── data/                     # 📊 데이터 폴더
│   └── df_KCB_grade.csv     # 원본 데이터 (Git LFS)
├── notebook/                 # 📓 Jupyter 노트북
│   └── [금융 원핫 다시 - 최종] 금융 + 라이프 모델.ipynb
├── src/                      # 💻 소스 코드
│   ├── config.py            # ⚙️ 설정 파일
│   ├── pipeline.py          # 🔄 ML 파이프라인
│   ├── preprocessing/       # 🔧 데이터 전처리
│   │   └── data_preprocessing.py
│   ├── models/              # 🤖 모델 학습
│   │   └── model_training.py
│   ├── evaluation/          # 📊 모델 평가
│   │   └── model_evaluation.py
│   ├── visualization/       # 📈 시각화
│   │   └── plotting.py
│   └── utils/              # 🛠️ 유틸리티
│       └── helpers.py
├── ui/                      # 🎨 웹 UI 컴포넌트
│   ├── __init__.py         # UI 모듈 초기화
│   ├── sidebar.py          # 사이드바 컴포넌트
│   ├── main_content.py     # 메인 컨텐츠
│   ├── data_upload.py      # 데이터 업로드
│   ├── model_selection.py  # 모델 선택 및 훈련
│   ├── results_display.py  # 결과 표시
│   └── visualization.py    # 시각화 컴포넌트
├── .streamlit/             # 🎛️ Streamlit 설정
│   ├── config.toml        # 앱 설정
│   └── credentials.toml   # 인증 정보
└── experiments/            # 🧪 실험 결과 (자동 생성)
```

## 🚀 빠른 시작

### 방법 1: 웹 인터페이스 (권장) 🌐

**UV 환경 (권장):**
```bash
# UV를 사용한 빠른 실행
./run_streamlit_uv.sh
```

**일반 Python 환경:**
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run streamlit_app.py
```

**브라우저에서 접속:**
- 로컬 URL: http://localhost:8501
- 자동으로 브라우저가 열리며 웹 인터페이스를 사용할 수 있습니다

### 방법 2: 명령줄 인터페이스 (CLI) 💻

#### 방법 1: 스크립트 실행
```bash
./run.sh
```

#### 방법 2: 직접 실행
```bash
python main.py
```

## 📊 주요 기능

### 🌐 웹 인터페이스 (Streamlit)
1. **홈 페이지**: 프로젝트 개요 및 기능 소개
2. **데이터 업로드**: 
   - CSV 파일 업로드 및 미리보기
   - 데이터 품질 검증 및 통계 정보
   - 기본 데이터셋 사용 옵션
3. **모델 훈련**: 
   - 다중 모델 동시 훈련 (RF, XGBoost, LightGBM, Ensemble)
   - 실시간 진행상황 표시
   - 하이퍼파라미터 설정
4. **예측**: 
   - 개별 고객 예측
   - 배치 예측 (CSV 업로드)
   - 예측 결과 다운로드
5. **시각화**: 
   - 데이터 분포 분석
   - 상관관계 히트맵
   - 모델 성능 비교 차트
6. **모델 비교**: 
   - 성능 지표 비교
   - 특성 중요도 분석
   - 레이더 차트 및 박스 플롯

### 💻 CLI 인터페이스
1. **데이터 전처리**:
   - OneHot Encoding for 범주형 변수
   - Standard Scaling for 연속형 변수
   - Label Encoding for 타겟 변수
   - 모드별 데이터 분할 (Life, Financial, Full)

2. **모델 학습**:
   - **Random Forest**: 앙상블 기반 분류
   - **XGBoost**: 그래디언트 부스팅
   - **LightGBM**: 경량화된 그래디언트 부스팅
   - **Logistic Regression**: 로지스틱 회귀
   - **Ensemble**: 모든 모델의 투표 앙상블

3. **모델 평가**:
   - 정확도 (Accuracy)
   - ROC-AUC Score
   - 분류 리포트 (Precision, Recall, F1-Score)
   - 과적합 평가
   - 교차검증 (5-Fold Stratified)

4. **시각화**:
   - ROC Curve 비교
   - Feature Importance
   - Confusion Matrix
   - 교차검증 결과 박스플롯
   - 모드별 성능 비교

## 🛠️ 기술 스택

### 프론트엔드 (웹 인터페이스)
- **Streamlit**: 웹 애플리케이션 프레임워크
- **Plotly**: 인터랙티브 차트
- **Altair**: 선언적 시각화

### 백엔드 (머신러닝)
- **Pandas**: 데이터 조작 및 분석
- **NumPy**: 수치 계산
- **Scikit-learn**: 머신러닝 알고리즘
- **XGBoost**: 그래디언트 부스팅
- **LightGBM**: 경량 그래디언트 부스팅
- **Matplotlib/Seaborn**: 시각화

### 개발 환경
- **UV**: Python 패키지 관리자 (권장)
- **Git LFS**: 대용량 파일 관리
- **Python 3.8+**: 프로그래밍 언어

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

### 🌐 웹 인터페이스 사용법

1. **데이터 업로드**:
   ```bash
   # Streamlit 앱 실행
   ./run_streamlit_uv.sh
   ```
   - 브라우저에서 "데이터 업로드" 페이지로 이동
   - CSV 파일 업로드 또는 기본 데이터 사용

2. **모델 훈련**:
   - "모델 훈련" 페이지에서 원하는 모델 선택
   - 데이터 모드 및 하이퍼파라미터 설정
   - "모델 훈련 시작" 버튼 클릭

3. **예측 수행**:
   - "예측" 페이지에서 개별 또는 배치 예측 선택
   - 결과 확인 및 다운로드

4. **시각화 및 분석**:
   - "시각화" 페이지에서 다양한 차트 확인
   - "모델 비교" 페이지에서 성능 분석

### 💻 CLI 사용법

#### 전체 분석 실행
```python
from src.pipeline import MLPipeline

pipeline = MLPipeline("my_experiment")
summary = pipeline.run_full_pipeline()
```

#### 모드별 비교
```python
from src.pipeline import ModeComparisonPipeline

comparison = ModeComparisonPipeline("mode_comparison")
results = comparison.run_mode_comparison()
```

## ✨ 주요 개선사항

기존 노트북 기반 작업을 다음과 같이 개선했습니다:

### 🔄 모듈화 및 아키텍처
1. **재사용성**: 각 기능이 독립적인 클래스로 분리
2. **확장성**: 새로운 모델이나 전처리 방법 쉽게 추가 가능
3. **실험 관리**: 자동화된 실험 추적 및 결과 저장
4. **코드 품질**: 로깅, 에러 처리, 타입 힌트 적용

### 🌐 웹 인터페이스 추가
5. **Streamlit 앱**: 사용자 친화적 웹 인터페이스
6. **실시간 시각화**: 인터랙티브 차트 및 그래프
7. **직관적 조작**: 드래그 앤 드롭 파일 업로드
8. **한글 지원**: 완벽한 한글 폰트 및 인터페이스

### 📈 고급 기능
9. **다중 실행 환경**: CLI와 웹 인터페이스 동시 지원
10. **실시간 모니터링**: 훈련 진행상황 및 성능 추적
11. **배치 처리**: 대량 데이터 예측 및 결과 다운로드
12. **모델 비교**: 직관적인 성능 비교 도구

## 🔧 문제해결

### Streamlit 앱 실행 오류
```bash
# UV 환경에서 의존성 재설치
uv pip install -r requirements.txt

# 포트 충돌 시 다른 포트 사용
uv run streamlit run streamlit_app.py --server.port 8502
```

### 한글 폰트 문제 (macOS)
- 시스템에 AppleGothic 폰트가 설치되어 있는지 확인
- 필요시 나눔폰트 또는 맑은고딕 폰트 설치

### 대용량 데이터 처리
- Git LFS를 사용하여 대용량 파일 관리
- 메모리 부족 시 배치 크기 조정

## 📞 지원

- **이슈 신고**: [GitHub Issues](https://github.com/Knell999/lifestyle_late/issues)
- **기능 요청**: Pull Request 환영
- **문서**: 프로젝트 README 및 코드 주석 참조

---

**버전**: v2.0.0 (Streamlit 웹 인터페이스 추가)  
**개발자**: KHJ  
**라이선스**: MIT  
**업데이트**: 2025년 6월 11일