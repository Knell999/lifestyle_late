# 데이터 설정 가이드

## 대용량 데이터 파일 처리

이 프로젝트는 395MB 크기의 `data/df_KCB_grade.csv` 파일을 사용합니다.

### Git LFS 설정

대용량 파일은 Git LFS(Large File Storage)로 관리됩니다:

```bash
# Git LFS 설치 (필요한 경우)
git lfs install

# 데이터 파일 다운로드 (클론 후)
git lfs pull
```

### 로컬 개발 시 주의사항

1. **첫 클론 시**: `git lfs pull` 명령어로 실제 데이터 파일을 다운로드해야 합니다
2. **데이터 파일 없이 테스트**: Streamlit 앱에서 파일 업로드 기능을 사용할 수 있습니다
3. **메모리 제한**: 대용량 데이터 처리 시 시스템 메모리를 고려하세요

### 대안 데이터 소스

원본 데이터가 없는 경우:
- Streamlit 웹 인터페이스의 "데이터 업로드" 기능 사용
- 비슷한 구조의 샘플 데이터로 테스트 가능
- `src/config.py`에서 데이터 경로 수정 가능

### 트러블슈팅

**Git LFS 오류 시:**
```bash
# LFS 재설정
git lfs install --force
git lfs pull --all
```

**메모리 부족 시:**
- 데이터 샘플링하여 작은 크기로 테스트
- 배치 처리 크기 조정 (`src/config.py`)