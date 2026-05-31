import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

from dataset.fashion_mnist import load_fashion_mnist

(x_train, t_train), (x_test, t_test) = load_fashion_mnist(flatten=False)

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=150, output_size=10, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=25, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)

trainer.train()

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(trainer.train_acc_list))
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("Fashion MNIST CNN Accuracy")
plt.show()