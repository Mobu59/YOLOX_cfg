import json

all_data_path = '/world/data-gpu-94/goods_detection_data/all_goods.v2.json'
dirty_data_path = './all_dirty_data.txt'
save_json_path = open('./goods_dirty_data.json', 'w')
lst = []
f_d = open(dirty_data_path, 'r')
lines = f_d.readlines()
for line in lines:
    line = line.split('\n')[0]
    lst.append(line)
with open(all_data_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        k, items = line[0], line[1]
        if k in lst:
           save_json_path.write(k + '\t' + items + '\n') 


