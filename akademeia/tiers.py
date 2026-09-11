"""Registered rung targets (PROTOCOL "Tiers"). The SHAPES are the
registration; token budgets are DRAFTS until each rung's timing probe freezes
them by dated amendment. Never searched, never tuned mid-campaign.

ctx and batch_size are held constant across rungs so stream materialization
and the paired design stay comparable; scale enters only through depth/width
and budget."""
from akademeia.config import SubstrateConfig

TIER1 = SubstrateConfig()                     # 4L/256d, 30M tokens (v2 parity)
TIER2 = SubstrateConfig(n_layers=6, d_model=384, n_heads=8, ctx=256,
                        batch_size=32, total_tokens=45_000_000,
                        eval_every_tokens=2_250_000)      # the v2 tier-2 rung
TIER3 = SubstrateConfig(n_layers=8, d_model=640, n_heads=10, ctx=256,
                        batch_size=32, total_tokens=135_000_000,
                        eval_every_tokens=6_750_000)
TIER4 = SubstrateConfig(n_layers=12, d_model=1024, n_heads=16, ctx=256,
                        batch_size=32, total_tokens=450_000_000,
                        eval_every_tokens=22_500_000)

RUNGS = {1: TIER1, 2: TIER2, 3: TIER3, 4: TIER4}
