import pandas as pd
from tqdm import trange
import pickle

class Person:
    def __init__(self, **data):
        self.ID = data['ID']
        self.loc = [data['X'], data['Y']]
        self.time_step = int(data['Time_step'])

# 数据准备和超参数定义
data = pd.read_csv('./students003.csv')

# 收集每个时刻所有的人以及对应的坐标
people = [[] for _ in range(n_frame)]
for index in trange(len(data)):
    person = Person(**data.loc[index].to_dict())
    people[person.time_step].append(person)

with open('./people.pkl', 'wb') as f:
    pickle.dump(people, f)