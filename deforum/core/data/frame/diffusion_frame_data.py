from dataclasses import dataclass
from typing import Any, Optional

from ..render_data import RenderData


@dataclass
class DiffusionFrameData:
    """Immutable collection of less essential frame data."""
    contrast: float
    amount: float
    kernel: int
    sigma: float
    threshold: float
    cadence_flow_factor: float
    redo_flow_factor: float
    cfg_scale: float
    distilled_cfg_scale: float
    strength: float
    noise: float
    # Note: hybrid_comp_schedules removed

    def flow_factor(self):
        # Note: hybrid flow factor functionality removed
        return 1.0

    @staticmethod
    def create(data: RenderData, i):
        keys = data.animation_keys.deform_keys
        return DiffusionFrameData(
            contrast=keys.contrast_schedule_series[i],
            amount=keys.amount_schedule_series[i],
            kernel=int(keys.kernel_schedule_series[i]),
            sigma=keys.sigma_schedule_series[i],
            threshold=keys.threshold_schedule_series[i],
            cadence_flow_factor=keys.cadence_flow_factor_schedule_series[i],
            redo_flow_factor=keys.redo_flow_factor_schedule_series[i],
            cfg_scale=keys.cfg_scale_schedule_series[i],
            distilled_cfg_scale=keys.distilled_cfg_scale_schedule_series[i],
            strength=keys.strength_schedule_series[i],
            noise=keys.noise_schedule_series[i] if hasattr(keys, 'noise_schedule_series') and i < len(keys.noise_schedule_series) else 0.0
            # Note: hybrid schedules removed
        )
