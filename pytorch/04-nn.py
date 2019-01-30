import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

#one parameters
params = list(net.parameters())
print(len(params))
#conv1's weight
print(params[0].size()) 


intput = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

#zero the gradient buffers of all parameters and backprop with random gradient
net.zero_grad()
out.backward(torch.randn(1, 10))




#two loss function
output = net(input)
target = torch.randn(10)
target = target.view(1, -1) #make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


#follow loss in the backward direction, get this graph of computations that looks like this:
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear -> MSELoss -> loss

#when we call loss.backward() the whole graph is differentiated w.r.t. the loss, all Tensors in the graph that has requires_graad=True will have their .grad Tennsor accumulated with the gradient.
#MSELoss
loss.grad_fn
#Linear
loss.grad_fn.next_function[0][0]
#RELu
loss.grad_fn.next_function[0][0].next_function[0][0]





#three backprop
net.zero_grad() #zeros the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)



#four update the weight: weight = weight -learning_rate * gradient

simply code:
learning_rate = 0.01
for f in net.parameters():
	f.data.sub_(f.grad.data * learning_rate)


import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

#Observe how gradient buffers had to be manually set to zero using optimizer.zero_grad(). This is because gradients are accumulated as explained in Backprop section.

