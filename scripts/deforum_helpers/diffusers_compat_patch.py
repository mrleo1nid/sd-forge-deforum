#!/usr/bin/env python3
"""
Diffusers Compatibility Patch for Forge + Wan 2.2
Fixes compatibility issues between diffusers git main (required for WanPipeline) and Forge's Flux backend
"""

def patch_flow_match_scheduler():
    """
    Patch FlowMatchEulerDiscreteScheduler.time_shift to handle None self parameter

    Forge's backend calls: FlowMatchEulerDiscreteScheduler.time_shift(None, mu, 1.0, sigmas)
    But diffusers git main expects time_shift to be an instance method with self.config

    This patch makes it work both ways.
    """
    try:
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        import torch

        # Save original method
        original_time_shift = FlowMatchEulerDiscreteScheduler.time_shift

        @classmethod
        def patched_time_shift(cls, self_or_mu, mu_or_sigma=None, sigma_or_t=None, t=None):
            """
            Patched version that works when called as class method with None self OR as instance method

            Forge calls: FlowMatchEulerDiscreteScheduler.time_shift(None, mu, 1.0, sigmas)
            -> receives (cls, None, mu, 1.0, sigmas)

            New diffusers calls: instance.time_shift(mu, 1.0, sigmas)
            -> receives (cls, self, mu, 1.0, sigmas)

            Args:
                cls: The class
                self_or_mu: Either self (instance) or mu (if called with None as first arg)
                mu_or_sigma: Either mu or sigma depending on call pattern
                sigma_or_t: Either sigma or t depending on call pattern
                t: Time values (only present in Forge's call pattern)
            """
            # Determine call pattern
            if t is not None:
                # Forge pattern: time_shift(None, mu, sigma, t)
                # self_or_mu = None, mu_or_sigma = mu, sigma_or_t = sigma, t = t
                mu = mu_or_sigma
                sigma = sigma_or_t
                # Use static implementation
            elif hasattr(self_or_mu, 'config'):
                # Instance method pattern: self.time_shift(mu, sigma, t)
                # self_or_mu = self, mu_or_sigma = mu, sigma_or_t = sigma, t = None -> but wait, t is the 4th arg
                # Actually this won't happen because we're replacing the whole method
                # Let me just use original for this case
                return original_time_shift(self_or_mu, mu_or_sigma, sigma_or_t)
            else:
                # Assume Forge pattern with positional args
                mu = mu_or_sigma
                sigma = sigma_or_t

            # Static version for Forge compatibility (default "exponential" behavior)
            time_val = t if t is not None else sigma_or_t

            # Handle mixed scalar/tensor inputs - convert to match time_val's type
            if isinstance(time_val, torch.Tensor):
                # Convert mu and sigma to tensors if they aren't already
                if not isinstance(mu, torch.Tensor):
                    mu = torch.tensor(mu, dtype=time_val.dtype, device=time_val.device)
                if not isinstance(sigma, torch.Tensor):
                    sigma = torch.tensor(sigma, dtype=time_val.dtype, device=time_val.device)
                return torch.exp(mu) / (torch.exp(mu) + (1 / time_val - 1) ** sigma)
            else:
                # Use numpy for scalar inputs
                import numpy as np
                # Convert any tensors to numpy
                if isinstance(mu, torch.Tensor):
                    mu = mu.cpu().numpy()
                if isinstance(sigma, torch.Tensor):
                    sigma = sigma.cpu().numpy()
                if isinstance(time_val, torch.Tensor):
                    time_val = time_val.cpu().numpy()
                return np.exp(mu) / (np.exp(mu) + (1 / time_val - 1) ** sigma)

        # Replace the method
        FlowMatchEulerDiscreteScheduler.time_shift = patched_time_shift

        print("✅ Diffusers compatibility patch applied: FlowMatchEulerDiscreteScheduler.time_shift")
        return True

    except Exception as e:
        print(f"⚠️ Failed to apply diffusers compatibility patch: {e}")
        return False


def patch_torch_rmsnorm():
    """
    Add RMSNorm to torch.nn if not present (PyTorch < 2.4.0)

    Wan 2.2 models require RMSNorm which was added in PyTorch 2.4.0
    WebUI Forge uses PyTorch 2.3.1, so we need to provide a compatible implementation
    """
    import torch
    import torch.nn as nn

    if hasattr(nn, 'RMSNorm'):
        print("✅ torch.nn.RMSNorm already available (PyTorch 2.4.0+)")
        return True

    try:
        print("🔧 Adding RMSNorm compatibility for PyTorch < 2.4.0...")

        class RMSNorm(nn.Module):
            """
            Root Mean Square Layer Normalization
            Compatible implementation for PyTorch < 2.4.0
            """
            def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=False, device=None, dtype=None):
                super().__init__()
                factory_kwargs = {'device': device, 'dtype': dtype}

                if isinstance(normalized_shape, int):
                    normalized_shape = (normalized_shape,)
                self.normalized_shape = tuple(normalized_shape)
                self.eps = eps
                self.elementwise_affine = elementwise_affine

                if self.elementwise_affine:
                    self.weight = nn.Parameter(torch.ones(normalized_shape, **factory_kwargs))
                    if bias:
                        self.bias = nn.Parameter(torch.zeros(normalized_shape, **factory_kwargs))
                    else:
                        self.register_parameter('bias', None)
                else:
                    self.register_parameter('weight', None)
                    self.register_parameter('bias', None)

            def forward(self, input):
                # Compute RMS
                variance = input.pow(2).mean(-1, keepdim=True)
                input = input * torch.rsqrt(variance + self.eps)

                # Apply weight and bias if enabled
                if self.elementwise_affine:
                    input = input * self.weight
                    if self.bias is not None:
                        input = input + self.bias

                return input

            def extra_repr(self):
                return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

        # Add to torch.nn module
        nn.RMSNorm = RMSNorm
        torch.nn.RMSNorm = RMSNorm

        print("✅ RMSNorm compatibility patch applied successfully")
        return True

    except Exception as e:
        print(f"⚠️ Failed to apply RMSNorm patch: {e}")
        return False


def patch_diffusers_attention():
    """
    Patch PyTorch's scaled_dot_product_attention to filter out enable_gqa parameter

    Diffusers git main uses enable_gqa parameter in scaled_dot_product_attention,
    but this parameter was added in PyTorch 2.4.0. Forge uses PyTorch 2.3.1.

    Nuclear option: Patch torch.nn.functional directly to filter unsupported parameters.
    """
    try:
        import torch
        import inspect

        # Check PyTorch version
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if torch_version >= (2, 4):
            print("✅ PyTorch 2.4.0+ detected - enable_gqa parameter supported")
            return True

        print(f"🔧 PyTorch {torch.__version__} detected - patching scaled_dot_product_attention...")

        # Save original PyTorch function
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        # Get the original signature to know what parameters are actually supported
        original_params = set(inspect.signature(original_sdpa).parameters.keys())
        print(f"   Original SDPA parameters: {original_params}")

        def patched_scaled_dot_product_attention(*args, **kwargs):
            """Wrapper that filters out unsupported parameters like enable_gqa"""
            # Remove unsupported parameters
            if 'enable_gqa' in kwargs:
                if kwargs.pop('enable_gqa'):
                    print("   ⚠️ enable_gqa=True requested but not supported in PyTorch 2.3.1, ignoring")

            # Remove any other parameters not in original signature
            unsupported = [k for k in kwargs.keys() if k not in original_params]
            for param in unsupported:
                print(f"   ⚠️ Removing unsupported parameter: {param}={kwargs[param]}")
                kwargs.pop(param)

            # Call original function with filtered parameters
            return original_sdpa(*args, **kwargs)

        # Replace PyTorch's function globally
        torch.nn.functional.scaled_dot_product_attention = patched_scaled_dot_product_attention

        print("✅ PyTorch scaled_dot_product_attention patched to filter enable_gqa")
        return True

    except Exception as e:
        print(f"⚠️ Failed to apply attention patch: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_all_patches():
    """Apply all compatibility patches"""
    print("🔧 Applying diffusers compatibility patches for Forge + Wan 2.2...")
    patch_torch_rmsnorm()
    patch_flow_match_scheduler()
    patch_diffusers_attention()
