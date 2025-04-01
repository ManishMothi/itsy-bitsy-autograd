
import math


class Value:
    def __init__(self, data, _children=(), _operation=''):
        self.data = data
        self._children = set(_children)
        self._operation = _operation
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, operation={self._operation})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        output = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward
        return output 

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output 

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports integer/float powers"
        output = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * output.grad

        output._backward = _backward

        return output
    
    def __truediv__(self, other):
        if other == 0:
            raise Exception("Cannot divide by zero")
        return self * other ** -1

    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
    
    def __rtruediv__(self, other):
        return other * self ** -1

    def relu(self):
        output = Value(0 if self.data < 0 else self.data, (self,), 'relu')

        def _backward():
            self.grad = (0 if output < 0 else output) * output.grad

        output._backward = _backward
        return output 
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        output = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * output.grad
            
        output._backward = _backward
        return output

    def backward(self):
        '''
        Uses Topological sort for back prop.
        Called on final value of computation 
        '''
        children = []
        visited = set()
        def build_top_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_top_sort(child) # ensures top sort invariant 
                children.append(v)
                
        build_top_sort(self)

        self.grad = 1
        for v in reversed(children):
            v._backward()






    

    
