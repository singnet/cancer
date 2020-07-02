from funs_balance import random_upsample_balance


X = [[1,1],[2,1], [3,1], [4,1], [1,0], [2,0]]
y = [1,1,1,1,0,0]

Xb, yb = random_upsample_balance(X,y)
print(Xb)
print(yb)
