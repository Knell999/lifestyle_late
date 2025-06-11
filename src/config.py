"""
Configuration file for the KCB Grade Prediction project
"""

# 데이터 경로
DATA_PATH = "data/df_KCB_grade.csv"
TARGET_COLUMN = "KCB_grade"

# 변수 정의
ONEHOT_FEATURES = [
    'SEX', 'JB_TP', 'HOME_ADM', 'COM_ADM', 'HB_1ST', 'B1Y_EQP_MDL', 
    'B1Y_MOB_OS', 'FAM_OWN_LIV_YN', 'OWN_LIV_YN', 'AGE', 'ROP'
]

LABEL_FEATURES = ['LIF_STG']

BINARY_FEATURES = [
    'VIP_CARD_YN', 'CAR_YN', 'TRAVEL_OS', 'TRAVEL_JJ', 'GOLF_INDOOR',
    'PREFER_SPORTS', 'RNCAR_JJ_YN', 'AIRLINE_OS_YN'
]

CONTINUOUS_FEATURES = [
    'R3M_FOOD_AMT', 'R3M_ENT_AMT', 'R3M_DEP_AMT', 'R3M_TRAVEL_AMT',
    'R3M_BEAUTY_AMT', 'R3M_ITRT_FIN_PAY', 'R3M_ITRT_FIN_INSUR',
    'R3M_ITRT_FIN_BANK', 'R3M_ITRT_LIFE_HEALTH', 'R3M_ITRT_SHOP_MART',
    'R3M_ITRT_ENT_SVOD'
]

FIN_FEATURES = [
    'TOT_ASST', 'HOUS_LN_BAL', 'CRDT_LN_BAL', 'OWN_HOUS_CNT', 
    'FAM_OWN_HOUS_CNT', 'CRDT_LN_BAL_NEW',
    'PYE_MAX_DLQ_DAY', 'PYE_C1L120237', 'PYE_C1L120250',
    'PYE_L10210000', 'PYE_L10216000', 'PYE_L10231000',
    'CD_USE_AMT', 'YR_SALES', 'SHP_CD_USE', 'SHP_LN_BAL',
    'PYE_D10110001', 'PYE_D10110003', 'PYE_D10110006',
    'PYE_D1011000C', 'PYE_D10210D00', 'PYE_D10232000'
]

# 라이프스타일 전용 변수
ONEHOT_FEATURES_LIFE = [
    'SEX', 'JB_TP', 'HOME_ADM', 'COM_ADM', 'HB_1ST', 'B1Y_EQP_MDL', 'B1Y_MOB_OS', 'AGE'
]

ONEHOT_FEATURES_FULL = ONEHOT_FEATURES_LIFE + ['FAM_OWN_LIV_YN', 'OWN_LIV_YN']

# 모델 하이퍼파라미터
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': [100],
        'max_depth': [30, 50, 100],
        'min_samples_split': [2, 3, 4, 5]
    },
    'xgboost': {
        'n_estimators': [100],
        'max_depth': [30, 50, 100],
        'learning_rate': [0.1, 0.2, 0.3, 0.4]
    },
    'lightgbm': {
        'n_estimators': [100],
        'max_depth': [30, 50, 100],
        'learning_rate': [0.1, 0.2, 0.3, 0.4]
    }
}

# 한글 변수명 매핑
FEATURE_NAME_KO = {
    'PYE_L10231000': '총약정금액',
    'CD_USE_AMT': '카드소비금액',
    'PYE_C1L120237': '1개월전 카드잔액 소진율',
    'CRDT_LN_BAL': '신용대출잔액',
    'PYE_L10210000': '대출건수',
    'PYE_L10216000': '신용대출건수',
    'PYE_C1L120250': '단기카드대출 소진율',
    'TOT_ASST': '총자산평가금액',
    'LIF_STG': '라이프스테이지',
    'PYE_D1011000C': '연체건수(1년내)',
    'FAM_OWN_HOUS_CNT': '주택보유건수(가구)',
    'OWN_LIV_YN': '자가거주여부',
    'FAM_OWN_LIV_YN': '자가거주여부(가구)',
    'SEX_2': '성별_여자',
    'HOUS_LN_BAL': '주택담보대출잔액',
    'R3M_FOOD_AMT': '3개월 요식 이용금액',
    'OWN_HOUS_CNT': '주택보유건수',
    'YR_SALES': '연매출추정금액',
    'R3M_ITRT_ENT_SVOD': '3개월 엔터(sVOD) 관심도',
    'CRDT_LN_BAL_NEW': '신규신용대출여부',
    'R3M_ITRT_FIN_BANK': '3개월 은행/카드_관심도',
    'ROP_1': '주택소유유형 - 1주택자',
    'PYE_D10110006': '연체건수(6개월내발생)',
    'JB_TP_910': '직업군 - 기타/무직'
}

# 기타 설정
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = -1
