import pickle
import matplotlib.pyplot as plt

temp = [1, 20, 300]

with open("models/manual_logs.pkl", "rb") as file:
    stats = pickle.load(file)

losses = []
wers = []
cers = []
epochs = []

for item in stats:
    losses.append(item["avg_train_loss"])
    wers.append(item["avg_wer"])
    cers.append(item["avg_cer"])
    epochs.append(item["epoch:"])

fig = plt.figure(figsize=(9, 3))

ax1 = fig.add_subplot(131)
ax1.set_title("train loss")
ax1.plot(epochs, losses)

ax2 = fig.add_subplot(132)
ax2.set_title("WER")
ax2.plot(epochs, wers)

ax3 = fig.add_subplot(133)
ax3.set_title("CER")
ax3.plot(epochs, cers)


plt.show()

