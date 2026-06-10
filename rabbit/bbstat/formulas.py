"""Closed-form helpers for the bin-by-bin (BBB) statistical treatment.

Pure functions only — no TF Variables, no Fitter state. Used by
:mod:`rabbit.bbstat.bbstat`.
"""

import tensorflow as tf


def solve_quad_eq(a, b, c):
    """Positive root of ``a x² + b x + c = 0`` (TF-broadcast)."""
    return 0.5 * (-b + tf.sqrt(b**2 - 4.0 * a * c)) / a
