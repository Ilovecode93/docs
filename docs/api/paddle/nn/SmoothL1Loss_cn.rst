.. _cn_api_paddle_nn_SmoothL1Loss:

SmoothL1Loss
-------------------------------

.. py:class:: paddle.nn.SmoothL1Loss(reduction='mean', delta=1.0, name=None)

该 OP 计算输入 input 和标签 label 间的 SmoothL1 损失，如果逐个元素的绝对误差低于 1，则创建使用平方项的条件
，否则为 L1 损失。在某些情况下，它可以防止爆炸梯度，也称为 Huber 损失，该损失函数的数学计算公式如下：

    .. math::
         loss(x,y) = \frac{1}{n}\sum_{i}z_i

`z_i`的计算公式如下：

    .. math::

        \mathop{z_i} = \left\{\begin{array}{rcl}
        0.5(x_i - y_i)^2 & & {if |x_i - y_i| < delta} \\
        delta * |x_i - y_i| - 0.5 * delta^2 & & {otherwise}
        \end{array} \right.

参数
::::::::::
    - **reduction** (string，可选): - 指定应用于输出结果的计算方式，数据类型为 string，可选值有：`none`, `mean`, `sum`。默认为 `mean`，计算 `mini-batch` loss 均值。设置为 `sum` 时，计算 `mini-batch` loss 的总和。设置为 `none` 时，则返回 loss Tensor。
    - **delta** (string，可选): SmoothL1Loss 损失的阈值参数，用于控制 Huber 损失对线性误差或平方误差的侧重。数据类型为 float32。默认值= 1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

输入
::::::::::
    - **input** (Tensor)：输入 `Tensor`，数据类型为 float32。其形状为 :math:`[N, C]`，其中 `C` 为类别数。对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_k]`，k >= 1。
    - **label** (Tensor)：输入 input 对应的标签值，数据类型为 float32。数据类型和 input 相同。



返回
:::::::::
Tensor，计算 `SmoothL1Loss` 后的损失值。


代码示例
:::::::::

COPY-FROM: paddle.nn.SmoothL1Loss
