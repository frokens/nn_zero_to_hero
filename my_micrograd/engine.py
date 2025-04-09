from graphviz import Digraph
from typing import Union

import math


class Value:
    def __init__(self,data,_children = (), _operater= "", label =""):
        for child in _children:
            if not isinstance(child,Value):
                print(_operater)
                raise f"children type: {child}"
        self.data = data
        self.grad = 0
        self.label = label
        self._backward = lambda: None 
        self._prev = set(_children)
        self._op = _operater

    def __add__(self, other) -> "Value":
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self,other), "+")
        
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward

        return out
    
    def __mul__(self, other)->"Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = backward

        return out
    
    def __pow__(self,other: Union[int, float]) -> "Value":
        assert isinstance(other,(int,float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), "**")

        def backward():
            self.grad += (other*(self.data ** (other - 1))) * out.grad
        out._backward = backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def backward():
            self.grad = out.grad * out.data
        out._backward = backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
            
        return out
    
    def backward(self):
        topo =[]
        visited = set()

        def buildtopo(v: "Value"):
          if v not in visited:
            visited.add(v)
            for child in v._prev:
                buildtopo(child)
            topo.append(v)
        
        buildtopo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()


    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), "relu")
        
        def _backward():
          self.grad += out.data * out.grad
        out._backward = _backward

        return out
       

    def __radd__(self,other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self): 
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    



def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot


if __name__ == "__main__":
    a = Value(5); a.label = "a"
    b = a.tanh()
    b.backward()
    print(a.grad)
    print(b.grad)