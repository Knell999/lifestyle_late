#!/bin/bash

# KCB Grade Prediction Project 검증 스크립트
# 프로젝트의 모든 컴포넌트가 올바르게 작동하는지 확인합니다.

echo "🔍 KCB Grade Prediction Project 검증 시작"
echo "============================================"

# 1. 필수 파일 존재 확인
echo "📂 필수 파일 확인 중..."

required_files=(
    "streamlit_app.py"
    "main.py"
    "requirements.txt"
    "src/config.py"
    "src/pipeline.py"
    "data/df_KCB_grade.csv"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✅ 모든 필수 파일이 존재합니다"
else
    echo "❌ 누락된 파일:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
fi

# 2. UV 환경 확인
echo ""
echo "🔧 UV 환경 확인 중..."
if command -v uv &> /dev/null; then
    echo "✅ UV가 설치되어 있습니다"
    uv --version
else
    echo "❌ UV가 설치되어 있지 않습니다"
fi

# 3. Python 의존성 확인
echo ""
echo "📦 Python 의존성 확인 중..."
uv pip list | grep -E "(streamlit|pandas|scikit-learn|xgboost|lightgbm|matplotlib|seaborn|plotly|altair)" || echo "⚠️ 일부 패키지가 설치되지 않았을 수 있습니다"

# 4. 모듈 import 테스트
echo ""
echo "🐍 Python 모듈 import 테스트..."
uv run python -c "
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import sklearn
    import xgboost
    import lightgbm
    print('✅ 모든 핵심 라이브러리 import 성공')
except ImportError as e:
    print(f'❌ Import 오류: {e}')
" 2>/dev/null

# 5. 프로젝트 구조 확인
echo ""
echo "🏗️ 프로젝트 구조 확인 중..."
directories=("src" "ui" "data" "notebook" ".streamlit")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/ 디렉토리 존재"
    else
        echo "❌ $dir/ 디렉토리 누락"
    fi
done

# 6. 데이터 파일 크기 확인
echo ""
echo "📊 데이터 파일 확인 중..."
if [ -f "data/df_KCB_grade.csv" ]; then
    file_size=$(du -h "data/df_KCB_grade.csv" | cut -f1)
    echo "✅ 데이터 파일 크기: $file_size"
    
    # 데이터 행 수 확인
    if command -v uv &> /dev/null; then
        row_count=$(uv run python -c "import pandas as pd; print(len(pd.read_csv('data/df_KCB_grade.csv')))" 2>/dev/null)
        if [ ! -z "$row_count" ]; then
            echo "✅ 데이터 행 수: $row_count"
        fi
    fi
else
    echo "❌ 데이터 파일이 없습니다"
fi

# 7. Git 상태 확인
echo ""
echo "📋 Git 상태 확인 중..."
if git status &> /dev/null; then
    echo "✅ Git 저장소가 초기화되어 있습니다"
    
    # 커밋 수 확인
    commit_count=$(git rev-list --count HEAD 2>/dev/null)
    echo "📈 총 커밋 수: $commit_count"
    
    # 마지막 커밋 정보
    last_commit=$(git log -1 --oneline 2>/dev/null)
    echo "📝 마지막 커밋: $last_commit"
    
    # 태그 확인
    tags=$(git tag -l 2>/dev/null)
    if [ ! -z "$tags" ]; then
        echo "🏷️ 태그: $tags"
    fi
else
    echo "❌ Git 저장소가 초기화되지 않았습니다"
fi

# 8. 실행 스크립트 권한 확인
echo ""
echo "🚀 실행 스크립트 확인 중..."
scripts=("run.sh" "run_streamlit_uv.sh" "run_streamlit.sh")
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "✅ $script 실행 가능"
        else
            echo "⚠️ $script 실행 권한 없음 (chmod +x $script 실행 필요)"
        fi
    else
        echo "❌ $script 파일 없음"
    fi
done

echo ""
echo "🎉 검증 완료!"
echo ""
echo "🚀 실행 방법:"
echo "  웹 인터페이스: ./run_streamlit_uv.sh"
echo "  CLI 인터페이스: ./run.sh 또는 python main.py"
echo ""
echo "📚 더 많은 정보는 README.md를 참조하세요."
