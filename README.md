# Machine Learning -  K-mean clustering - Entry level Project(1) : 

![image](https://user-images.githubusercontent.com/78548649/199871765-5f8108c7-ddc9-42ef-be0f-9c2399af5113.png)

K-mean clustering is an unsupervised learning algorithms, which helps classify the data you see into groups.
i.e find the underlying structure

<details><summary>❌ K Nearest Neighbo( or K - NN) </summary>
<p>
  By the way, K Nearest Neighbo( or K - NN) is different from K-mean clustering. 
  K-NN is a supervised learning algorithm that classifies the data you know into groups.
  
  #### ![image](https://user-images.githubusercontent.com/78548649/199906946-52650a4b-898e-4209-8b87-1348828ca17e.png)

</p>
</details>

Here is a get-hand-dirty project: 

data: iris dataset 鳶尾花卉資料集

![image](https://user-images.githubusercontent.com/78548649/199905175-5c84d3e2-faae-48f5-bddd-7cf9fa6bd7c5.png)

<details><summary>👉 dataset background</summary>
<p>
  
One of the datasets in sci-kit learn, does not require reading any CSV files from external. It is convenient for new bees like me to code ML projects 
sci-kit learn 中的數據集之一，不需要從外部讀取任何 CSV 文件。 方便像我這樣的新手編寫ML項目

 #### ![image](https://user-images.githubusercontent.com/78548649/199905706-b4f4311e-7946-4c0c-af3e-460a0ca5d411.png)

Fisher created it in 1936, containing measures of sepal length, sepal width, petal length, and petal width for three species of Iris (Iris setosa, Iris virginica, and Iris versicolor).
費舍爾在 1936 年創建了它，其中包含三種鳶尾花（Iris setosa、Iris virginica 和 Iris versicolor）的萼片長度、萼片寬度、花瓣長度和花瓣寬度的測量值。

</p>
</details>

Task: 



Steps: 

1. Import library 

<details><summary> code </summary>
<p>
  
````
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
````
</p>
</details>

2. Load iris dataset from sklearn

<details><summary> code </summary>
<p>
  
````
#### This is 150 rows × 4 columns for the whole dataset 
#### With 4 features 	: | sepal length (cm) |	sepal width (cm) | petal length (cm) |	petal width (cm) | 

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

data = scale(iris.data) # scale the iris data
data

````
</p>
</details>


3. K-means algorithm

<details><summary> code </summary>
<p>
classify the cluster already known

````
kmeans = KMeans(n_clusters=3, random_state=1)     # As known that there are 3 spieces of iris, it has 3 clusters
kmeans.fit(x)


labels = kmeans.labels_
centroids = kmeans.cluster_centers_

````
</p>
</details>

4. K-means clusting plot

<details><summary> code </summary>
<p>

````
x = pd.DataFrame(x, columns = features)

colormap = np.array(['b', 'r', 'y'])
plt.scatter(x['sepal length (cm)'], x['sepal width (cm)'], c=colormap[labels])
plt.scatter(centroids[:,0], centroids[:,1], s = 300, alpha=0.5, marker = 'x', c = 'k')

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)');
```
</p>
</details>


5. Evaluation results













 

