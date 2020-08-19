layer_units = [16,8,4,2,1]
tmp = []
for i in range(1, 5):
    tmp.append([[0 for _ in range(layer_units[i])] for _ in range(layer_units[i])])

print(tmp)