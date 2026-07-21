"""Retired in protocol v3.

Denoising is eval-only now (its identity operator makes the CGLS data-projection
return the observation exactly, so its training gradient is identically zero), and the
rotation is 4 configs over exp1-exp4. See FOUNDATION_MODEL.md, "Protocol v3".
"""

raise SystemExit(__doc__)
