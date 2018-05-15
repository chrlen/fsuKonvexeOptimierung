import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv("Advertising.csv")
eta = data['Sales']
etaMean = eta.mean()
meanFreeEta = eta -etaMean


for columnName in ['TV','Radio','Newspaper']:
    xi = data[columnName]
    xiMean = xi.mean()
    meanFreeXi = xi - xiMean
    
    #Parameterschätzung
    x_1 = sum(meanFreeXi * meanFreeEta) / sum(np.power(meanFreeXi,2))
    x_2 = etaMean -x_1*xiMean
    
    #Erzeuge Scatterplot
    plt.scatter(data[columnName],data['Sales'],color='C2')
    
    #Berechne Funktionswerte für Regressionsgrade 
    #auf dem x-Interval des Scatterplots
    axes = plt.gca()
    x_axis = np.array(axes.get_xlim())
    regVals = x_1 * x_axis + x_2
    plt.plot(x_axis,regVals,'--',color='C3')
    
    plt.title(columnName)
    plt.show()






#plt.figure()



#data['TV'].plot().scatter()




#print(data['TV'])
