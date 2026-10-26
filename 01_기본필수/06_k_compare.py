"""실습 06 기본필수: 전날의 20건에서 같은 분할로 K만 비교합니다."""
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# CSV를 읽고 요청 데이터를 학습·검증·테스트 세트로 나눕니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
develop, test = train_test_split(raw, test_size=0.2, random_state=42, stratify=raw["failed"])
train, valid = train_test_split(develop, test_size=0.25, random_state=42, stratify=develop["failed"])
# test는 보관만 합니다. 학습·예측에 넣지 않습니다.

# 글자 수를 입력으로 사용해 이웃 수에 따른 분류 성능을 비교합니다.
X_train = train[["input_chars"]]
X_valid = valid[["input_chars"]]

# K별 학습과 성능 비교
for k in [1, 3, 5]:
    # 같은 데이터 분할에서 이웃 수만 바꿔 비교합니다.
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)
    print("K:", model[-1].n_neighbors, "valid F1:", round(f1_score(valid["failed"], prediction, zero_division=0), 3))

print("분할:", len(train), len(valid), len(test))
