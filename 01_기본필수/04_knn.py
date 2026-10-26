"""실습 04 기본필수: 이웃 3개의 투표와 평균을 비교합니다."""
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 데이터 준비
X = [[1], [2], [4], [7], [9]]
class_y = [0, 0, 1, 1, 1]
number_y = [10, 20, 40, 70, 90]

# 3.2에 가까운 입력은 4, 2, 1입니다. 거리: 0.8, 1.2, 2.2

# 가까운 이웃 3개의 다수결로 클래스를 예측합니다.
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, class_y)
print("분류:", classifier.predict([[3.2]]))

# 같은 이웃 3개의 숫자 정답 평균으로 값을 예측합니다.
regressor = KNeighborsRegressor(n_neighbors=3)
regressor.fit(X, number_y)
print("회귀:", regressor.predict([[3.2]]).round(2))
