import pickle


with open('./data/gcc.pkl', 'rb') as f:
    raw_data = pickle.load(f)  # len1 * 61

print(len(raw_data))
print(len(raw_data[0]))
print(raw_data[0])
print(raw_data[1])
print(raw_data[2])
print(raw_data[3])