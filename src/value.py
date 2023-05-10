class Value:
    
    def __init__(self, data, children=(), op='', label=''):
        # Only supports int or float data
        assert(isinstance(data, int) or isinstance(data, float))

        self.data = data
        self.grad = 0.0
        self.label = label
        
        # Internals
        self._prev = set(children)
        self._op = op
        self._backward = lambda : None

    def details(self):
        s  = f"Label={self.label},Op={self._op},data={self.data:.5f},grad={self.grad:.5f}"
        return s

    def __repr__(self):
        if self.label:
            return f"Value({self.label} => {self.data:.5f})"
        else:
            return f"Value({self.data:.5f})"
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += 1 * out.grad
            other.grad += 1 * out.grad
            # print(f"{self.details()}  | {other.details()}")
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        other_inverse = other**-1
        return self * other_inverse

    '''
    def __eq__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data == other.data
    '''

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other) 

    def __iadd__(self, other): # in-place addition
        if isinstance(other, Value):
            self.data += other.data
        else:
            self.data += other
        return self

    def __isub__(self, other): # in-place subtraction
        if isinstance(other, Value):
            self.data -= other.data
        else:
            self.data -= other
        return self
        
    def tanh(self):
        e2x = math.exp(2*self.data)
        t = (e2x - 1) / (e2x + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - (out.data ** 2)) * out.grad # this "out.grad" is computed not when tanh is called but when _backward() is called
            
        out._backward = _backward  # The function pointer, not the function's return value
        return out

    def exp(self): # called as self.exp(); which will give us e^d
        out = Value(math.exp(self.data),(self,),'exp')

        def _backward():
            self.grad += math.exp(self.data) * out.grad # this "out.grad" is computed not when exp is called but when _backward() is called

        out._backward = _backward
        return out

    def __pow__(self, n): # called as self ^ n, division is a special case with n = -1
        out = Value(self.data ** n, (self,),'**')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad # this "out.grad" is computed not when pow is called but when _backward() is called
        out._backward = _backward
        return out


    def __radd__(self, other):
        return Value(other) + self

    def __rsub__(self, other):
        return Value(other) - self

    def __rmul__(self, other):
        return Value(other) * self

    def __rtruediv__(self, other):
        return Value(other) / self

    def backward(self): 
        # Topological sort
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        rev_topo = reversed(topo)
        
        for node in rev_topo:
            # print(f"Clearing grad of node : {node.details()}")
            # node.grad = 0
            # I could zero out all gradients here but it will promote overlooking zeroing out 
            # gradients in other frameworkd(s.g. PyTorch). So sptting a warning
            if node.grad != 0:
                print("WARNING: Detected non-zero gradients. Did you remember to zero_grad()?")
                break
            
        # Initialize the gradient of initial node to 1 and then propagate backward
        for node in reversed(topo):
            node._backward()

