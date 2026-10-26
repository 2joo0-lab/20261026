"""실습 05 기본필수: train에서 정한 기준을 새 입력에도 적용합니다."""
from sklearn.preprocessing import StandardScaler

# 학습 데이터와 검증 데이터
train = [[100], [200], [300]]
valid = [[400]]

# 학습 데이터의 기준을 구해 검증 데이터에 적용
scaler = StandardScaler()
scaler.fit(train)  # 평균과 크기 기준을 train에서 구합니다.
converted = scaler.transform(valid)

# 변환 결과 확인
print("train 기준 평균:", scaler.mean_)
print("train 변환:", scaler.transform(train).round(3))
print("valid 변환:", converted.round(3))
print("변환 후 기준 평균:", scaler.mean_)
# valid에 다시 fit하지 않습니다. StandardScaler는 자료를 정규분포로 만들지 않습니다.
