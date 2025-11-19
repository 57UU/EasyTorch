from .tensor import Tensor
import typing
class Module:
    def __init__(self):
        self._parameters: typing.Dict[str, Tensor] = {}
        self._modules: typing.Dict[str, Module] = {}
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    def register_parameter(self, name, param):
        self._parameters[name] = param
    def register_module(self, name, module):
        self._modules[name] = module
    def parameters_iter(self):
        for param in self._parameters.values():
            for grad_num in param.data:
                yield grad_num
        for module in self._modules.values():
            yield from module.parameters()
    def parameters(self):
        return list(self.parameters_iter())
    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            # 如果是Tensor，注册到_parameters字典
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            # 如果是Module，注册到_modules字典  
            self.register_module(name, value)
        else:
            object.__setattr__(self, name, value)
    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")