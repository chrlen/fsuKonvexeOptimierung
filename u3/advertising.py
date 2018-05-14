import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("Advertising.csv")
plt.scatter(data[:,0],data[:,1])




#plt.figure()



#data['TV'].plot().scatter()




#print(data['TV'])
