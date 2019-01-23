import torch


# step one
x = torch.empty(5,3)
print(x)

x = torch.rand(5,3)

x = torch.zeros(5, 3, dtype=torch.long)

x = torch.tensor([5.5, 3])

x = x.new_ones(5,3, dtype = torch.double)

x = torch.randn_like(x, dtype=torch.float)

x.size()


#step two
y = torch.rand(5,3)
pritn(x+y)

torch.add(x, y)

result = torch.empty(5,3)
torch.add(x, y, out = result)
print(result)

y.add_(x)
x.copy_(y)
x.t_()



view/ item
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)
t = x.view(-1, 4)
x.size()  y.size() z.size() t.size()   -1 inferred from other parameter

x = torch.randn(1)
x
x.item()





