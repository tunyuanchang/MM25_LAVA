import matplotlib.pyplot as plt
import numpy as np

FILES = ["../result/lstm_loss.txt"]

train_loss = []
valid_loss = []

for FILE_NAME in FILES:
    with open(FILE_NAME, 'r') as f:
        train_loss.append(f[0].strip().split(','))
        valid_loss.append(f[1].strip().split(','))

num_epochs = len(train_loss)
epochs = range(1, num_epochs+1)

# # Training losses
# train_losses = [200.9629, 190.3367, 190.1605]

# # Validation losses
# val_losses = [210.6575, 216.0242, 214.4271]

# # Training losses
# train_losses1 = [209.1844, 206.1392, 206.1209]

# # Validation losses
# val_losses1 = [226.6485, 225.8860, 226.2799]

# plt.figure(figsize=(8,5))
# plt.plot(epochs, train_losses, c='orange', marker='o', label='TCN (Training)')
# plt.plot(epochs, val_losses, c='blue', marker='o', label='TCN (Testing)')

# plt.plot(epochs, train_losses1, c='darkgoldenrod', marker='x', label='Concatenation (Training)')
# plt.plot(epochs, val_losses1, c='navy',marker='x', label='Concatenation (Testing)')

# plt.xlabel('Epoch')
# plt.ylabel('MPJPE (mm)')
# plt.title('Loss')

# plt.xticks(epochs)

# # Create y-axis ticks from 180 to 230 with step of 10
# yticks = np.arange(180, 231, 10)
# plt.yticks(yticks)

# plt.legend(loc='lower left')
# plt.grid(True)


# plt.savefig('loss_plot.png')
# # plt.show()