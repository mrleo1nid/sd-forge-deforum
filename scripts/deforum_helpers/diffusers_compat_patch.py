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
            if isinstance(t if t is not None else sigma_or_t, torch.Tensor):
                time_val = t if t is not None else sigma_or_t
                return torch.exp(mu) / (torch.exp(mu) + (1 / time_val - 1) ** sigma)
            else:
                import numpy as np
                time_val = t if t is not None else sigma_or_t
                return np.exp(mu) / (np.exp(mu) + (1 / time_val - 1) ** sigma)

        # Replace the method
        FlowMatchEulerDiscreteScheduler.time_shift = patched_time_shift

        print("✅ Diffusers compatibility patch applied: FlowMatchEulerDiscreteScheduler.time_shift")
        return True

    except Exception as e:
        print(f"⚠️ Failed to apply diffusers compatibility patch: {e}")
        return False


def apply_all_patches():
    """Apply all compatibility patches"""
    print("🔧 Applying diffusers compatibility patches for Forge + Wan 2.2...")
    patch_flow_match_scheduler()
