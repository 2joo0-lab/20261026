"""실습 07~08 기본필수: 같은 파일을 완성하고 다시 실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score

# CSV를 읽고 요청 데이터를 학습·검증·테스트 세트로 나눕니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
develop, test = train_test_split(raw, test_size=0.2, random_state=42, stratify=raw["failed"])
train, valid = train_test_split(develop, test_size=0.25, random_state=42, stratify=develop["failed"])
# test는 보관만 합니다. 학습·예측에 넣지 않습니다.

# 실행시간과 실패 여부를 예측할 입력 특성을 선택합니다.
features = ["input_chars", "task_type"]
X_train = train[features]
X_valid = valid[features]
pre = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])

# 실행시간 회귀 예측
regression_rows = []
classification_rows = []
result = valid[["request_id", "elapsed_ms", "failed"]].copy()
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression())]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # 실행시간을 학습하고 검증 자료의 실행시간을 예측합니다.
    model.fit(X_train, train["elapsed_ms"])
    prediction = model.predict(X_valid)
    regression_rows.append({"model": name, "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction)})
    result[name + "_ms"] = prediction

# 실패 여부 분류
for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(max_iter=1000))]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # 실패 여부를 학습하고 검증 자료의 실패 여부를 예측합니다.
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)
    classification_rows.append({"model": name, "F1": f1_score(valid["failed"], prediction, zero_division=0)})
    result[name + "_failed"] = prediction

# 회귀와 분류 결과를 각각 CSV 파일로 저장합니다.
regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)
regression.to_csv(HERE / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
result.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
print("분할:", len(train), len(valid), len(test))
print("회귀 비교 / 낮은 MAE가 좋음")
print(regression.round(3).to_string(index=False))
print("분류 비교 / 높은 F1이 좋음")
print(classification.round(3).to_string(index=False))
print("저장: regression_comparison.csv, classification_comparison.csv, validation_predictions.csv")
print("test는 학습·예측하지 않았습니다.")
