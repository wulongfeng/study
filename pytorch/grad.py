import torch


# you can stop autograd from tracking history on Tensors with .requires_grad=True 
# by wrapping the code block in with torch.no_grad():

x = torch.randn(3, requires_grad=True)

print(x.requires_grad)
#True

print((x**2).requires_grad)
#True



with torch.no_grad():
	print(x.requires_grad)
	#True

	print((x**2).requires_grad)
	#False

#example one:grad
x = torch.ones(2,2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward()
print(x.grad)
#gradients d(out)/dx


#example two:Jacobian-vector product
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
	y = y * 2
print(y)

#y is no longer a scalar, torch.autograd could not compute the full jacobian directly,
#if we want the jacobian-vector product, simply pass the vector to backward as argument

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)



