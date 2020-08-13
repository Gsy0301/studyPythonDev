'''
线性回归
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

x = np.linspace(0, 5, 256)
noise = np.random.randn(256) * 2
y = x * 5 + 7 + noise
df = pd.DataFrame()
df['x'] = x
df['y'] = y
sns.lmplot(x='x', y='y', data=df, height=4)

train_x = x.reshape(-1, 1).astype('float32')
train_y = y.reshape(-1, 1).astype('float32')
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

model = nn.Linear(1, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 3000

for i in range(1, epochs + 1):
    optimizer.zero_grad()
    out = model(train_x)
    loss = loss_fn(out, train_y)
    loss.backward()
    optimizer.step()
    print('epoch{}\tloss{}'.format(i, loss.item()))

w, b = model.parameters()
print(w.item(), b.item())

pred = model.forward(train_x).data.numpy().squeeze()
plt.plot(x, y, 'go', label='Truth', alpha=0.3)
plt.plot(x, pred, label='Predicted')
plt.legend()
plt.show()
