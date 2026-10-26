"""실습 01 기본필수: 학습과 예측 두 문장을 연결합니다."""
from sklearn.linear_model import LinearRegression

# 데이터 입력
X = [[1], [2], [3], [4], [5]]
y = [52, 61, 69, 80, 89]  

# 입력 데이터와 정답으로 모델을 학습합니다.
model = LinearRegression() 
model.fit(X, y)
prediction = model.predict([[6]])

# 결과값 출력
print("기울기:", round(model.coef_[0], 1))
print("절편:", round(model.intercept_, 1))
print("6시간 예측:", round(prediction[0], 1))

