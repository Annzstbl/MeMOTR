import torch.nn as nn
# @HOOK_REGISTRY.register('gradient_collector')
# class GradientCollector(Hook):
# 原类继承Hook, 这在脱离mmdet框架后没有实现
class GradientCollector(nn.Module):
    def __init__(self,
                 runner,
                 hook_module=["transformer.decoder.class_embed"],
                 collect_func_module="criterion.loss_cls",
                 grad_type='output'):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
        """
        super(GradientCollector, self).__init__(runner)
        self.grad_type = grad_type

        grad_collect_func_module = self.get_tar_module(runner.model, collect_func_module)
        collect_func = getattr(grad_collect_func_module, 'collect_grad')
        hook_target_modules = [self.get_tar_module(runner.model, t) for t in hook_module]

        def _backward_fn_hook(module, grad_in, grad_out):
            if self.grad_type == 'input':
                collect_func(grad_in[0])
            elif self.grad_type == 'output':
                collect_func(grad_out[0])
            else:
                raise NotImplementedError
        for m in hook_target_modules:
            m.register_full_backward_hook(_backward_fn_hook)

    def get_tar_module(self, model, targets):
        targets = targets.split('.')
        targets_module = model
        for t in targets:
            targets_module = getattr(targets_module, t)
        return targets_module


# 可直接版
class GradientCollector:
    def __init__(self, model, hook_module, collect_func_module, grad_type='output'):
        self.grad_type = grad_type
        self.handles = []

        grad_collect_func_module = self.get_tar_module(model, collect_func_module)
        collect_func = getattr(grad_collect_func_module, 'collect_grad')
        hook_target_modules = [self.get_tar_module(model, t) for t in hook_module]

        def _backward_fn_hook(module, grad_in, grad_out):
            if self.grad_type == 'input':
                collect_func(grad_in[0])
            elif self.grad_type == 'output':
                collect_func(grad_out[0])
            else:
                raise NotImplementedError

        for m in hook_target_modules:
            self.handles.append(m.register_full_backward_hook(_backward_fn_hook))

    def get_tar_module(self, model, targets):
        targets = targets.split('.')
        targets_module = model
        for t in targets:
            targets_module = getattr(targets_module, t)
        return targets_module

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

# 不使用字符串而直接使用模型
class SimpleGradientCollector:
    def __init__(self, target_modules, collect_func, grad_type='output'):
        self.handles = []
        self.grad_type = grad_type

        def _hook(module, grad_in, grad_out):
            if self.grad_type == 'input':
                collect_func(grad_in[0])
            elif self.grad_type == 'output':
                collect_func(grad_out[0])
            else:
                raise NotImplementedError

        for m in target_modules:
            self.handles.append(m.register_full_backward_hook(_hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# 直接函数hook
def build_gradient_collector(target_module, collect_func, grad_type='output'):
    def _backward_fn_hook(module, grad_in, grad_out):
        if grad_type == 'input':
            collect_func(grad_in[0])
        elif grad_type == 'output':
            collect_func(grad_out[0])
        else:
            raise NotImplementedError(f"Unknown grad_type: {grad_type}")

    handle = target_module.register_full_backward_hook(_backward_fn_hook)
    return handle