import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        # Clear old gradients before another optimization step accumulates new ones.
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        # Leaf modules without trainable values expose an empty parameter list.
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        # Start weights randomly; the bias begins at zero.
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        # A neuron computes a weighted sum, then optionally applies ReLU.
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        # Evaluate every neuron on the same input vector.
        out = [n(x) for n in self.neurons]
        # Keep single-output layers scalar-valued for convenient composition.
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # Flatten the parameters from all neurons into one list.
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        # nouts gives each layer's width; nin is the input width of layer one.
        sz = [nin] + nouts
        # Use ReLU in hidden layers and a linear activation in the output layer.
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        # Feed each layer's output into the next layer.
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
