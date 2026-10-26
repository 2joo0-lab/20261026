"""실습 03 기본필수: 작은 정상 자료로 확률과 클래스를 확인합니다."""
from sklearn.linear_model import LogisticRegression

# 원리 확인용 데이터입니다. X의 한 값은 작업 복잡도, y는 성공 0 / 실패 1입니다.
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

# 로지스틱 회귀 모델 준비
model = LogisticRegression()
model.fit(X,y)
new = [[2], [5]]

#확률 배열에서 클래스가 나오는 순서를 확인합니다.
print("클래스 순서:", model.classes_)
print("확률 [성공, 실패]:") 
print(model.predict_proba(new).round(3))
print("예측 클래스:", model.predict(new))

