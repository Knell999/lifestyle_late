#!/bin/bash

# KCB Grade Prediction Streamlit App 실행 스크립트
# 표준 Python 가상환경에서 Streamlit 앱을 실행합니다.

echo "🏦 KCB Grade Prediction Streamlit App"
echo "===================================="

# 현재 디렉토리 확인
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ streamlit_app.py 파일을 찾을 수 없습니다."
    echo "올바른 프로젝트 디렉토리에서 실행하세요."
    exit 1
fi

# 가상환경 확인 및 활성화
if [ -d ".venv" ]; then
    echo "🔧 가상환경 활성화 중..."
    source .venv/bin/activate
else
    echo "❌ 가상환경을 찾을 수 없습니다."
    echo "가상환경을 먼저 생성하세요: python -m venv .venv"
    exit 1
fi

# 필요한 패키지 설치
echo "📦 의존성 확인 및 설치 중..."
pip install -r requirements.txt --quiet

# 데이터 파일 확인
if [ ! -f "data/df_KCB_grade.csv" ]; then
    echo "⚠️  데이터 파일을 찾을 수 없습니다: data/df_KCB_grade.csv"
    echo "앱에서 파일 업로드 기능을 사용하거나 데이터 파일을 확인하세요."
fi

# Streamlit 앱 실행
echo ""
echo "🚀 Streamlit 앱 실행 중..."
echo "📱 브라우저에서 자동으로 열립니다"
echo "🔗 로컬 URL: http://localhost:8501"
echo "⏹️  종료하려면 Ctrl+C를 누르세요"
echo ""

streamlit run streamlit_app.py --server.port 8501
