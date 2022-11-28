from __future__ import print_function
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np
import logging

mnist = mx.test_utils.get_mnist()

batch_size = 100
train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

net = nn.HybridSequential()
with net.name_scope():
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(64, activation='relu'))
    net.add(nn.Dense(10))

net.hybridize()

gpus = mx.test_utils.list_gpus()
ctx =  [mx.gpu()] if gpus else [mx.cpu(0), mx.cpu(1)]
net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.02})

epoch = 10
metric = mx.metric.Accuracy()
softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
for i in range(epoch):
    train_data.reset()
    for batch in train_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        with ag.record():
            for x, y in zip(data, label):
                z = net(x)
                loss = softmax_cross_entropy_loss(z, y)
                loss.backward()
                outputs.append(z)
        metric.update(label, outputs)
        trainer.step(batch.data[0].shape[0])
    name, acc = metric.get()
    metric.reset()
    print('training acc at epoch %d: %s=%f'%(i, name, acc))

metric = mx.metric.Accuracy()
val_data.reset()
for batch in val_data:
    data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
    outputs = []
    for x in data:
        outputs.append(net(x))
    metric.update(label, outputs)
print('validation acc: %s=%f'%metric.get())
assert metric.get()[1] > 0.94

net.export("net", epoch=1)

converted_model_path = onnx_mxnet.export_model('./net-symbol.json', './net-0001.params', [(128,784)], np.double, 'converted.onnx')

