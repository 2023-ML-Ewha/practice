import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

multi_data = pd.read_csv('open_multi.csv')

# Multi-Class Classification for Open-World Case
X_multi = multi_data.drop('y', axis=1)
y_multi = multi_data['y']
  
# Parameters
FEAT_NUM = X_multi.shape[1]  # number of features
ROUND_NUM = 3000000
NEIGHBOUR_NUM = 5  # number of neighbors for kNN
RECOPOINTS_NUM = 5  # number of neighbors for distance learning

def alg_init_weight():
    return np.random.rand(FEAT_NUM) / 2.0 + 0.5

def dist(feat1, feat2, weight):
    return np.sum(weight * np.abs(feat1 - feat2) * (feat1 != -1) * (feat2 != -1))

def alg_recommend2(feat, featclasses, weight):
    featlen = len(feat)
    distlist = np.zeros(featlen)
    recogoodlist = np.zeros(RECOPOINTS_NUM, dtype=int)
    recobadlist = np.zeros(RECOPOINTS_NUM, dtype=int)

    for i in range(ROUND_NUM // featlen):
        id_ = i % featlen
        print("\rLearning weights... {} ({:d}-{:d})".format(i, 0, ROUND_NUM // featlen), end="", flush=True)

        try:
            trueclass = featclasses[id_]  # 수정된 부분

            # Learn distance to other feat elements, put in distlist
            for k in range(featlen):
                distlist[k] = dist(feat[id_], feat[k], weight)

            # Set my own distance to max
            distlist[id_] = np.max(distlist)

            pointbadness = 0
            maxgooddist = 0  # the greatest distance of all the good neighbors NEIGHBOUR_NUM

            # Find the good neighbors: NEIGHBOUR_NUM lowest distlist values of the same class
            for k in range(RECOPOINTS_NUM):
                minind = np.argmin(distlist)
                maxgooddist = max(maxgooddist, distlist[minind])
                distlist[minind] = np.max(distlist)
                recogoodlist[k] = minind

            # Update: Ensure trueclass value is within the valid range
            trueclass_values = set(featclasses)  # Convert to set for faster lookup
            if 0 <= trueclass < len(trueclass_values):
                for dind in range(featlen):
                    try:
                        if np.array_equal(featclasses[dind], trueclass):
                            distlist[dind] = np.max(distlist)
                    except (IndexError, ValueError) as e:
                        print(f"Error: {e}, dind: {dind}, featlen: {featlen}, trueclass: {trueclass}")

            for k in range(RECOPOINTS_NUM):
                ind = np.argmin(distlist)
                if distlist[ind] <= maxgooddist:
                    pointbadness += 1
                distlist[ind] = np.max(distlist)
                recobadlist[k] = ind

            pointbadness /= float(RECOPOINTS_NUM)
            pointbadness += 0.2

            featdist = np.zeros(FEAT_NUM)
            badlist = np.zeros(FEAT_NUM, dtype=int)
            minbadlist = 0
            countbadlist = 0

            for f in range(FEAT_NUM):
                if weight[f] == 0:
                    badlist[f] = 0
                else:
                    maxgood = 0
                    countbad = 0

                    for k in range(RECOPOINTS_NUM):
                        n = np.abs(feat[id_][f] - feat[recogoodlist[k]][f])
                        if feat[id_][f] == -1 or feat[recogoodlist[k]][f] == -1:
                            n = 0
                        maxgood = max(maxgood, n)

                    for k in range(RECOPOINTS_NUM):
                        n = np.abs(feat[id_][f] - feat[recobadlist[k]][f])
                        if feat[id_][f] == -1 or feat[recobadlist[k]][f] == -1:
                            n = 0
                        featdist[f] += n
                        if n <= maxgood:
                            countbad += 1

                    badlist[f] = countbad
                    if countbad < minbadlist:
                        minbadlist = countbad

            for f in range(FEAT_NUM):
                if badlist[f] != minbadlist:
                    countbadlist += 1

            w0id = np.zeros(countbadlist, dtype=int)
            change = np.zeros(countbadlist)

            temp = 0
            C1 = 0
            C2 = 0

            for f in range(FEAT_NUM):
                if badlist[f] != minbadlist:
                    w0id[temp] = f
                    change[temp] = weight[f] * 0.02 * badlist[f] / float(RECOPOINTS_NUM)
                    C1 += change[temp] * featdist[f]
                    C2 += change[temp]
                    weight[f] -= change[temp]
                    temp += 1

            totalfd = 0

            for f in range(FEAT_NUM):
                if badlist[f] == minbadlist and weight[f] > 0:
                    totalfd += featdist[f]

            for f in range(FEAT_NUM):
                if badlist[f] == minbadlist and weight[f] > 0:
                    weight[f] += C1 / totalfd

        except IndexError:
            print(f"Index Error at id_: {id_}")

    print("\n")


# Define a pipeline with k-NN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=NEIGHBOUR_NUM, weights='uniform', algorithm='auto'))
])

# Multi-Class Classification for Open-World Case
scaler = StandardScaler()
X_multi_scaled = scaler.fit_transform(X_multi)
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi_scaled, y_multi, test_size=0.2, random_state=42)

# Distance learning for multi-class classification
feat_multi = X_multi_train
featclasses_multi = y_multi_train.values
weight_multi = alg_init_weight()

# Learn weights for multi-class classification
print(f"Learning weights for {ROUND_NUM} rounds for multi-class classification...")
for _ in range(ROUND_NUM):
    alg_recommend2(feat_multi, featclasses_multi, weight_multi)
print("Finished learning weights for multi-class classification.")

# Set the k-NN classifier with the specified parameters for multi-class classification
knn_classifier_multi = KNeighborsClassifier(n_neighbors=NEIGHBOUR_NUM, weights='uniform', algorithm='auto')
# Set the learned weights to the pipeline
pipeline.named_steps['knn'].set_params(weights=weight_multi)

knn_classifier_multi.fit(X_multi_train, y_multi_train)

# Evaluate multi-class classification model for Open-World Case
y_multi_pred = knn_classifier_multi.predict(X_multi_test)

print("\nMulti-Class Classification Report - Open-World Case:")
print(classification_report(y_multi_test, y_multi_pred))
accuracy_multi = accuracy_score(y_multi_test, y_multi_pred)
print("Accuracy:", accuracy_multi)

