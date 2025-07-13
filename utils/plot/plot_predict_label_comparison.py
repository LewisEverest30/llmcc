from matplotlib import pyplot as plt

RESULT_FILE = r'output\test\predict_result_2025-06-30-20-46-28.txt'
BEGIN = 12500
END = 15000
label_list = []
output_list = []
with open(RESULT_FILE, 'r') as f:
    for i, line in enumerate(f):
        line = line.strip()
        label, output = line.split()
        label_list.append(float(label))
        output_list.append(float(output))

label_list = label_list[BEGIN:END]
output_list = output_list[BEGIN:END]
plt.plot(label_list, label='Label')
plt.plot(output_list, label='Prediction')
plt.legend(
    fontsize=18,
)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Throughput', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
