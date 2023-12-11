# dataset





### dataset loading
```
import pickle

USE_SUBLABEL = False
URL_PER_SITE = 10
TOTAL_URLS   = 950

# Load the pickle file
print("Loading datafile...")
# with open("/content/mon_standard.pkl", 'rb') as fi: # Path to mon_standard.pkl in Colab
#   data = pickle.load(fi)
with open("C:/Users/jain5/Desktop/기계학습/mon_standard.pkl", 'rb') as fi: # Path to mon_standard.pkl in Colab
    data = pickle.load(fi)
print('loading 완료')
    
X1 = [] # Array to store instances (timestamps) - 19,000 instances, e.g., [[0.0, 0.5, 3.4, ...], [0.0, 4.5, ...], [0.0, 1.5, ...], ... [... ,45.8]]
X2 = [] # Array to store instances (direction*size) - size information
y = [] # Array to store the site of each instance - 19,000 instances, e.g., [0, 0, 0, 0, 0, 0, ..., 94, 94, 94, 94, 94]

# Differentiate instances and sites, and store them in the respective x and y arrays
# x array (direction*timestamp), y array (site label)
for i in range(TOTAL_URLS):
    if USE_SUBLABEL:
        label = i
    else:
        label = i // URL_PER_SITE # Calculate which site's URL the current URL being processed belongs to and set that value as the label. Thus, URLs fetched from the same site are labeled identically.
    for sample in data[i]:
        size_seq = []
        time_seq = []
        for c in sample:
            dr = 1 if c > 0 else -1
            time_seq.append(abs(c))
            size_seq.append(dr * 512)
        X1.append(time_seq)
        X2.append(size_seq)
        y.append(label)
size = len(y)

print(f'Total samples: {size}') # Output: 19000
```


### incoming_packets_size/outgoing_packets_size
```
# 관찰별로 들어오는 송신/수신 패킷의 누적합
incoming_packets_size=[0]*19000# Server->Client
outgoing_packets_size=[0]*19000
i=0
j=0
for i in range(19000):
    for j in range(len(X2[i])):
        if X2[i][j]<0: # X2[0] -> 0번 인덱스에는 1번 웹페이지의 1번째 관찰 결과가 리스트로 저장. 1번 인덱스는 1번 웹페이지의 2번째 관찰 결과...이렇게 20회씩 관찰
            incoming_packets_size[i]+=X2[i][j] # 음수인거..즉 수신 packet size 누적합
        else:
            outgoing_packets_size[i]+=X2[i][j] # 양수...송신 packet size 누적합
len(outgoing_packets_size) #19000
```


### total_packets_size
```
# 전체 누적 합
total_packets_size=[0]*190000

i=0
for i in range(19000):
    total_packets_size[i] = incoming_packets_size[i] + outgoing_packets_size[i]
len(total_packets_size)
```


### N_packets
```
N_packets=[]
i=0
for i in range(len(X1)):
    N_packets.append(len(X1[i])) # N_packets는 모든 관찰 횟수
```


### incoming_packets_N/outgoing_packets_N
```
# 관찰별로 들어오는 송신/수신 패킷의 수
incoming_packets_N=[0]*19000# Server->Client
outgoing_packets_N=[0]*19000
i=0
j=0
for i in range(19000):
    for j in range(len(X2[i])):
        if X2[i][j]<0: # X2[0] -> 0번 인덱스에는 1번 웹페이지의 1번째 관찰 결과가 리스트로 저장. 1번 인덱스는 1번 웹페이지의 2번째 관찰 결과...이렇게 20회씩 관찰
            incoming_packets_N[i]+=1
        else:
            outgoing_packets_N[i]+=1
```



### incoming_fraction/outgoing_fraction
```
a=[]
outgoing_fraction=[]
for i in range(19000): 
    a.append(outgoing_packets_N[i]/len(X1[i])) # 일단 fraction 전체(19000) 관찰 결과별로 저장
b=[]
incoming_fraction=[]
for i in range(19000): 
    b.append(incoming_packets_N[i]/len(X1[i])) # 일단 fraction 전체(19000) 관찰 결과별로 저장
```


### incoming_packets_three/outgoing_packets_three
```
incoming_packets_three=[0]*19000# Server->Client
outgoing_packets_three=[0]*19000
i=0
j=0
for i in range(19000):
    for j in range(30): # 처음 관찰하는 30개의 패킷만 관찰
        if X2[i][j]<0:
            incoming_packets_three[i]+=1
        else:
            outgoing_packets_three[i]+=1
```


### time_fraction
```
# 추가 feature, 시간당 총 패킷 수 비율
c=[]
time_fraction=[]
for i in range(19000): 
    c.append(N_packets[i]/X1[i][-1]) # 일단 fraction 전체(19000) 관찰 결과별로 저장
```


### in_std/out_std
```
out_std=[]
in_std=[]
i=0
j=0
a1=[]
a2=[]
for i in range(19000):
    for j in range(len(X2[i])):
        if X2[i][j]<0:
            a1.append(j+1) # a1에는 수신(incoming) 패킷 목록 저장
        else:
            a2.append(j+1)
    out_std.append(np.std(a1))
    a1=[]
    in_std.append(np.std(a2))   
    a2=[]
```


### in_mean/out_mean
```
out_mean=[]
in_mean=[]
i=0
j=0
a1=[]
a2=[]
for i in range(19000):
    for j in range(len(X2[i])):
        if X2[i][j]<0:
            a1.append(j+1) # a1에는 수신(incoming) 패킷 목록 저장
        else:
            a2.append(j+1)
    out_mean.append(np.mean(a1))
    a1=[]
    in_mean.append(np.mean(a2))   
    a2=[]
```


### sum_N
```
sum_N=[]
i=0
for i in range(19000):
    sum_N.append(incoming_packets_N[i]+outgoing_packets_N[i]+N_packets[i])
```

### dataframe 만들기
```
import pandas as pd

# index 0~19(20개씩)은 같은 페이지 관찰 결과이다.
my_df=pd.DataFrame(zip(incoming_packets_N, outgoing_packets_N, incoming_packets_three, outgoing_packets_three, a, b,
                      out_std, in_std, out_mean, in_mean, N_packets,c,sum_N, incoming_packets_size, outgoing_packets_size, total_packets_size, y), 
                   columns=['N_in_packets','N_out_packets','in_first_thirty','out_first_thirty', 'out_fraction','in_fraction',
                           'out_std','in_std', 'out_mean', 'in_mean','N_packets', 'time_fraction','sum_N','incoming_packets_size',
                            'outgoing_packets_size','total_packets_size','y'])

```

### dataset 저장
```
# dataset 바탕화면에 저장
import pandas as pd
import os

os.chdir("저장 경로")
my_df.to_csv("dataset.csv")
```
