import numpy as np
import statsmodels.api as sm

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])
x = sm.add_constant(x)

#print(x)
#print(y)

model = sm.Logit(y, x)
result = model.fit(method='newton',maxiter=100)
#print(result.params)
print(result.predict(x))
