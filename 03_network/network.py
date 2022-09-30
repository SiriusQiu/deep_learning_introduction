import numpy as np

def step_function(x):
    return np.array(x>0, dtype=int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# 在计算的过程中容易出现移除问题，因此减去最大值
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum = np.sum(exp_x)
    return exp_x/sum




# 稠密层
class Dense():
    def __init__(self, input_num:int, output_num:int, activate_func=None):
        # 权重初始化，权重矩阵的形状是(input_num, output_num)，输入矩阵的形状是(1, input_num)，输出形状是(1, output_num)
        self.__w=np.random.normal(0, 1, (input_num, output_num))
        self.__b = np.random.normal(0, 1, (1, output_num))
        self.__activate_func=activate_func

    def forward(self, x):
        y = np.dot(x, self.__w) + self.__b
        if self.__activate_func == None:
            return y
        else:
            return self.__activate_func(y)

class Output():
    def __init__(self, output_func):
        self.__output_func = output_func

    def forward(self, x):
        return self.__output_func(x)


class Network():
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        output = x
        for lay in self.layers:
            output = lay.forward(output)

        return output

x = np.random.normal(0,1, (1, 10))
print(x)

network = Network()
network.add_layer(Dense(10,2))
network.add_layer(Dense(2,5))
network.add_layer(Output(softmax))

print(network.forward(x))
print(np.sum(network.forward(x)))