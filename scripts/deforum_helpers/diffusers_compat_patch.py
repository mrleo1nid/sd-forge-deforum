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
        def patched_time_shift(cls, mu, sigma, t):
            """
            Patched version that works when called as class method with None self

            Args:
                cls: Can be the class or None (from Forge's static call)
                mu: Shift parameter
                sigma: Scale parameter
                t: Time values (sigmas)
            """
            # If called with proper self (instance method), use original
            if cls is not None and hasattr(cls, 'config'):
                return original_time_shift(mu, sigma, t)

            # Otherwise, implement static version for Forge compatibility
            # This is the default "exponential" behavior
            if isinstance(t, torch.Tensor):
                return torch.exp(mu) / (torch.exp(mu) + (1 / t - 1) ** sigma)
            else:
                import numpy as np
                return np.exp(mu) / (np.exp(mu) + (1 / t - 1) ** sigma)

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
