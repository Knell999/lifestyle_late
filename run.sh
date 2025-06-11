#!/bin/bash

# KCB Grade Prediction 실행 스크립트

echo "KCB Grade Prediction Pipeline"
echo "=============================="

# 가상환경 활성화
if [ -d ".venv" ]; then
    echo "가상환경 활성화 중..."
    source .venv/bin/activate
else
    echo "가상환경을 찾을 수 없습니다. .venv 폴더를 확인해주세요."
    exit 1
fi

# 의존성 설치 확인
echo "의존성 확인 중..."
pip install -r requirements.txt --quiet

# 메인 스크립트 실행
echo "파이프라인 실행 중..."
python main.py

echo "실행 완료!"
