# Sequential Learning with Perceptrons
Below, you'll find the barebone code for a perpectron, which is implemented making use of Python classes
Documentation: https://docs.python.org/3/tutorial/classes.html

Tasks:
1. Load the .csv data using pandas
2. Visualize the data using pandas.plot() (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html) or plt.plot() (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)
3. Divide the data into a training set and into a test set, then create np.arrays for the x and y columns of each data set
4. Initialize the weights of the percepton randomly in a range between -0.2 and 0.2
5. Implement the net_process function, calculating the weighted sum of it's inputs + bias. Use the predict function to examine the output of the untrained network. 
6. Implement the fit function of the network by making use of the delta rule in it's sequential form (online training).
7. Display meaningful metrics as an print output in every epoch and use the fit function to train on the training data (hint: since we use online training, a very low learning rate might be needed)
8. Collect the chosen metric data while training and return it as an output of the fit function. Visualize those metrics.
9. You might notice, that the networks metrics won't improve after several epochs. Why is that? Implement an "abort mechanism", which terminates training in that case.
10. Use the network to predict on test data of your choice


```python
import matplotlib.pyplot as plt #library for visualizing data
%matplotlib widget 
#setting for jupyter lab
plt.rcParams['figure.figsize'] = [12, 6] #setting figure size (plots)

import pandas as pd #(software library for data analysis and manipulation, https://pandas.pydata.org/docs/)
import numpy as np #(software library for matrix multiplications, https://numpy.org/doc/)
import statistics as stats #(python module for statistic calculations, https://docs.python.org/3/library/statistics.html)
```


```python
class Perceptron_sequential():
    
    def __init__(self):
        self.weights={}
        #self.weights['m'] = ?
        #self.weights['b'] = ?
        
    def activation(self, data):
        activation = data * 1 #(linear)
        return activation
    
    def net_process(self, x):
        #net = ?
        return net
    
    def predict(self,x):
        pred = self.activation(self.net_process(x))
        return pred
        
    
    #def fit(self, X_train, Y_train, X_val, Y_val, epochs, lrate):
        #?
            
                
                
```
