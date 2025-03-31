
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
        return output 

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')
        return output 

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data ** other.data, (self, other), f'**{other}')
        return output
    
    def __truediv__(self, other):
        if other == 0:
            raise Exception("Cannot divide by zero")
        return self * other ** -1
    

    
