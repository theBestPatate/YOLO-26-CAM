from typing import Callable, List
from torch.nn import Module
from torch import Tensor


class Activations_and_gradients_recorder:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(
        self,
        model: Module,
        target_layers: List[Module],
        reshape_transform: Callable,
    ):
        self.model = model
        self.gradients: List[Tensor] = []
        self.activations: List[Tensor] = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module: Module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module: Module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor that requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(gradient):
            if self.reshape_transform is not None:
                gradient = self.reshape_transform(gradient)
            self.gradients: List[Tensor] = [gradient.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients: List[Tensor] = []
        self.activations: List[Tensor] = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
