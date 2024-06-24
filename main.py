import pandas as pd
import numpy as np
data=pd.read_csv(r"D:\coding journey\aiml\python\udemy\Machine Learning A-Z (Codes and Datasets)\Part 5 - Association Rule Learning\Section 28 - Apriori\Python\Market_Basket_Optimisation.csv",header=None)
print(data)
transactions = []
for i in range(0, 7501):
  transactions.append([str(data.values[i,j]) for j in range(0, 20)])

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

result=list(rules)
print(result)

def dataframe(result):
  lhs=[tuple((results[2][0][0]))[0] for results in result]
  rhs=[tuple((results[2][0][1]))[0] for results in result]
  support=[((results[1])) for results in result]
  confidence=[((results[2][0][2])) for results in result]
  lift=[((results[2][0][3])) for results in result]
  result=list(zip(lhs,rhs,support,confidence,lift))
  result=pd.DataFrame(result,columns=["lhs","rhs","support","confidence","lift"])
  return result

result=dataframe(result)

result=result.nlargest(n=10,columns='lift')
print(result)