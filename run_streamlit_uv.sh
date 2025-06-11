#!/bin/bash

# KCB Grade Prediction Streamlit App 실행 스크립트
# UV 환경에서 Streamlit 앱을 실행합니다.

echo "🏦 KCB Grade Prediction Streamlit App"
echo "===================================="

# 현재 디렉토리 확인
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ streamlit_app.py 파일을 찾을 수 없습니다."
    echo "올바른 프로젝트 디렉토리에서 실행하세요."
    exit 1
fi

# UV 환경 확인
echo "🔧 UV 환경 확인 중..."
if ! command -v uv &> /dev/null; then
    echo "❌ UV가 설치되어 있지 않습니다."
    echo "UV를 먼저 설치하세요: https://docs.astral.sh/uv/"
    exit 1
fi

# 필요한 패키지 설치
echo "📦 의존성 확인 및 설치 중..."
uv pip install -r requirements.txt

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

uv run streamlit run streamlit_app.py --server.port 8501
