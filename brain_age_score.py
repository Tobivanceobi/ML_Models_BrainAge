import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Brain Age Prediction Challenge results.csv')
print(df)
names = ['tsneurotech', 'MethodA', 'thatsvenyouknow', 'OUR SCORE', 'zeta', 'Nitin_Singh', 'tstrypst', 'JuanG', 'stupid', 'vivekraja', 'mdjain']
scores = [1.156811, 1.600948, 1.603094, 1.62356, 1.640561, 1.660653, 1.681221, 1.681343, 1.694003, 1.772961, 1.781127]

plt.barh(names, scores)
plt.xlim(1, 2)
plt.xlabel('MAE in years')
plt.title('Brain Age Prediction Challenge Ranking')
plt.show()

