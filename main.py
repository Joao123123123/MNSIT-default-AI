from PIL import Image
import numpy as np
import os
import random
import math

weights_file = "weights.npy"
weights_file2 = "weights_2.npy"
weights_file3 = "weights_3.npy"
bias_file1 = "bias1.npy"
bias_file2 = "bias2.npy"
bias_file3 = "bias3.npy"
dataset = "C:/Users/joaom/PycharmProjects/PythonProject/train"
learning = 0.0001

if os.path.exists(weights_file3):
    weightstres = np.load(weights_file3)
    print("Loaded existing weights.")
else:
    weightstres = np.random.uniform(-0.05, 0.05, (10, 64))
    np.save(weights_file3, weightstres)
    print("Generated new weights and saved.")

if os.path.exists(weights_file2):
    weightsdois = np.load(weights_file2)
    print("Loaded existing weights.")
else:
    weightsdois = np.random.uniform(-0.05, 0.05, (64, 128))
    np.save(weights_file2, weightsdois)
    print("Generated new weights and saved.")

if os.path.exists(weights_file):
    weightsum = np.load(weights_file)
    print("Loaded existing weights.")
else:
    weightsum = np.random.uniform(-0.05, 0.05, (128, 784))
    np.save(weights_file, weightsum)
    print("Generated new weights and saved.")

if os.path.exists(bias_file1):
    bias01 = np.load(bias_file1)
else:
    bias01 = [0]*128
    np.save(bias_file1, bias01)

if os.path.exists(bias_file2):
    bias02 = np.load(bias_file2)
else:
    bias02 = [0]*64
    np.save(bias_file2, bias02)

if os.path.exists(bias_file3):
    bias03 = np.load(bias_file3)
else:
    bias03 = [0]*10
    np.save(bias_file3, bias03)

def sigmoid(x):
    x=np.clip(x, -400, 400)
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0.01
tcerto=0
terrado=0
counter = [0] * 10
for step in range(300000):
    true = random.randint(0, 9)
    label_folder = os.path.join(dataset, str(true))
    img_name = random.choice(os.listdir(label_folder))
    img_path = os.path.join(label_folder, img_name)

    img = Image.open(img_path).convert("L").resize((28, 28))
    pixels = np.array(img) / 255.0
    inputvector = pixels.flatten()

    target = [0] * 10
    target[true] = 1
    counter[true] += 1

    Values = [[0 for layer in range(4)] for neuron in range(784)]
    for i in range(784):
        Values[i][0] = inputvector[i]

    Total1 = [0 for _ in range(128)]
    for k in range(128):
        Total1[k] =0
        for j in range(784):
            Total1[k] += Values[j][0] * weightsum[k][j]
        Values[k][1] = relu(Total1[k] + bias01[k])




    Total2 = [0 for _ in range(64)]
    for k2 in range(64):
        Total2[k2]=0
        for j2 in range(128):
            Total2[k2] += Values[j2][1] * weightsdois[k2][j2]
        Values[k2][2] = relu(Total2[k2] + bias02[k2])


    High = 0
    Highlabel = ""
    Total3 = [0 for _ in range(10)]
    sums=0
    for k3 in range(10):
        for j3 in range(64):
            Total3[k3] += Values[j3][2] * weightstres[k3][j3]
        Total3[k3] +=bias03[k3]
    for k4 in range(10):
        sums += np.exp(np.clip(Total3[k4] - max(Total3), -20, 20))
    for k3 in range(10):
        Values[k3][3] = np.exp(np.clip(Total3[k3] - max(Total3), -20, 20)) / sums
        if Values[k3][3] > High:
            High = Values[k3][3]
            Highlabel = k3



    dcdt4 = [0]*10
    dcdt3 = [0]*64
    dcdt2 = [0]*128

    for m2 in range(10):
        dcdt4[m2] = (Values[m2][3] - target[m2])

    for t2 in range(64):
        Variable=0
        for m2 in range(10):
            Variable += dcdt4[m2]*weightstres[m2][t2]
        dcdt3[t2] = Variable*relu_derivative(Values[t2][2])

    for a2 in range(128):
        Jorge=0
        for t2 in range(64):
            Jorge += dcdt3[t2]*weightsdois[t2][a2]
        dcdt2[a2]=Jorge*relu_derivative(Values[a2][1])

    dcdt4 = np.clip(dcdt4, -1, 1)
    dcdt3 = np.clip(dcdt3, -1, 1)
    dcdt2 = np.clip(dcdt2, -1, 1)


    for m3 in range(10):
        bias03[m3] -= dcdt4[m3] * learning
        for t3 in range(64):
            weightstres[m3][t3] -= dcdt4[m3] * Values[t3][2] * learning

    for t3 in range(64):
        bias02[t3] -= dcdt3[t3] * learning
        for a3 in range(128):
            weightsdois[t3][a3] -= dcdt3[t3] * Values[a3][1] * learning

    for a3 in range(128):
        bias01[a3] -= dcdt2[a3] * learning
        for b3 in range(784):
            weightsum[a3][b3] -= dcdt2[a3] * Values[b3][0] * learning

    if step % 50 == 0:
        for i in range(10):
            print(f"Output {i}: {round(Values[i][3], 4)}")
    if Highlabel == true:
        print(f"✅ Step {step}: Correct — Predicted {Highlabel}, True {true}")
        tcerto+=1

    else:
        print(f"❌ Step {step}: Wrong — Predicted {Highlabel}, True {true}")
        terrado+=1

    if step % 100 == 0:
        print("Label distribution:", counter)
        print(f"Errados:{terrado}, Certos:{tcerto}")
        terrado=0
        tcerto=0
        np.save(weights_file, weightsum)
        np.save(weights_file2, weightsdois)
        np.save(weights_file3, weightstres)
        np.save(bias_file1, bias01)
        np.save(bias_file2, bias02)
        np.save(bias_file3, bias03)
        print("saved")

    print(f"Max dcdt2: {max(dcdt2)}")
print("this was the biggest", High)
print("this was the number", Highlabel)

np.save(weights_file, weightsum)
np.save(weights_file2, weightsdois)
np.save(weights_file3, weightstres)
np.save(bias_file1, bias01)
np.save(bias_file2, bias02)
np.save(bias_file3, bias03)