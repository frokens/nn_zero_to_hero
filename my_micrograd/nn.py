import random
from engine import Value

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
    def __call__(self, x):
        return x
    
    def train_module(self, xs:list, ys:list,epcoche = 300):
        for i in range(epcoche):
            yspred = [self(x) for x in xs]
            loss = sum([(ypred - y)**2 for ypred, y in zip(yspred,ys)])
            
            loss.backward()

            for p in self.parameters():
                p.data -= 0.1 * p.grad
            
            print(f"Epoch {i+1}/{epcoche}, Loss: {loss.data}")
            self.zero_grad()

class Neuro(Module):
    def __init__(self, nin, activation = Value.tanh):
        self.w = [Value(random.uniform(-1, 1),label = f"w_{i}") for i in range(nin)]
        self.b = Value(0, label = "b")
        self.activation = activation

    def __call__(self, x):
        out = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
        return self.activation(out) 
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):
    def __init__(self, nin, nout, activation = Value.tanh):
        self.neurons = [Neuro(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    

class MLP(Module):
    def __init__(self, nin, nouts, activation = Value.tanh):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1],activation) for i in range(len(nouts))]
    
    def __call__ (self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

if __name__ == "__main__":
    x = [Value(1), Value(2), Value(3)]
    n = MLP(3,[4,4,1])
    xs=[[1.0, 2.0, 3.0],[2.0, 5.0, 1.0]]
    ys = [1.0,-1.0]
    n.train_module(xs, ys, epcoche = 100)
