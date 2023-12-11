# Dataset

### How to build
1. Move to the 'dataset' branch<br>
Detailed explanation is written in README.md

2. Download the source code files
Upload the files(ex. 'load_pickle_code.ipynb') on Google Colaboratory and execute.<br>
Once you execute, these dataset files will be created: dataset.csv, unmon_dataset.csv, open_binary.csv, open_multi.csv.
- dataset.csv: dataset for monitored websites
- unmon_dataset.csv: dataset for unmonitored websites
- open_binary.csv: dataset for binary classification for open-world case
- open_multi.csv: dataset for multi-class classification for open-world case



# Models
## 1) k-NN Classifier
### How to build
#### 1. Clone Repository and move to the cloned directory
~~~
git clone https://github.com/2023-ML-Ewha/practice.git
~~~

~~~
cd practice
~~~

<br> 

#### 2. Switch to branch 'knn'
~~~
git checkout knn
~~~

<br> 

#### 3. Open the directory and download the 'knn.py' file
~~~
open .
~~~
And download manually

<br> 

#### 4. Run the code(knn.py) using Google Colaboratory
Upload the file 'knn.py' and **dataset files** on the Google Colaboratory, and run the code. <br> The recommended experimental setting is:
Python 3 Google Compute Engine Backend
System RAM:1.3 / 12.7 GB 
Disk: 26.9 / 107.7 GB

<br> 

#### 4-1. (Optional and NOT recommended) just in case you have all the modules and if you want to run the code in the terminal:

Depending on the version of Python that you use, execute source code with the command below
~~~
python knn.py
~~~

or

~~~
python3 knn.py
~~~
<br> 
<br> 

cf) 
'knn.py' is an implementation of k-nn model, incorporating general features which are considered as the strongest indicators in the paper(Wang et al. Effective Attacks and Provable Defenses for Website Fingerprinting (Usenix Security 14)).  
<br>
'custom_knn.py' is an implementation of k-nn model which incorporated the general features and applied a custom distance function that was recommended in the paper.
To run, you can go through the same steps as described above. Note that only the multi-class classification of Open-world case was implemented in 'custom_knn.py' for practice, unlike 'knn.py' which includes all cases.
<br> 
<br> 

## 2) Random-Forest
### How to build
#### 1. Clone Repository and move to the cloned directory
~~~
git clone https://github.com/2023-ML-Ewha/practice.git
~~~

~~~
cd practice
~~~

<br> 

#### 2. Switch to branch 'random-forest'
~~~
git checkout random-forest
~~~

<br> 

#### 3. Open the directory and download the 'random_forest_final.ipynb' file 
~~~
start .
~~~
(command for a Windows environment)<br>
And download the file manually

<br> 

#### 4. Run the code(random_forest_final.ipynb) using Google Colaboratory
Upload the file 'random_forest_final.ipynb' and **dataset files** on the Google Colaboratory, and run the code. <br> The recommended experimental setting is:
Python 3 Google Compute Engine Backend
System RAM:1.1 / 12.7 GB 
Disk: 26.9 / 107.7 GB

<br>
<br> 
cf)  'random_forest_final.ipynb' implements a random forest model, incorporating key features recommended in the paper(Hayes and Danezis. k-fingerprinting: A Robust Scalable Website Fingerprinting Technique (Usenix Security16))
<br>
<br>

## 3) CUMUL
