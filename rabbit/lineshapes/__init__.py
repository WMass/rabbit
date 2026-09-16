"""Physics lineshape providers for :mod:`rabbit.unbinned`.

A *provider* supplies the characteristic function of a resonance lineshape that
has no closed form, as a differentiable function of the fit parameters. It is
plugged into :class:`~rabbit.unbinned.TabulatedLineshapeKernel`, which is a
drop-in for any other :class:`~rabbit.unbinned.PhysicsKernel`.

The contract (see :class:`~rabbit.lineshapes.zgamma.ZGammaLineshape` for the
reference implementation):

``param_names``
    tuple of fit-parameter names the provider consumes.
``provider(values, t_abs) -> (re, im)``
    the additive complex exponent ``(log|phi|, arg phi)`` of the lineshape CF
    at the absolute conjugate variable ``t_abs = t / sigma_i``, broadcast to the
    shape of ``t_abs``.
``config() -> dict``
    JSON-serialisable description with a ``"type"`` key, so the provider
    survives the round trip through the datacard via :func:`make_provider`.
"""

from rabbit.lineshapes.zgamma import ZGammaLineshape

_PROVIDERS = {
    ZGammaLineshape.kind: ZGammaLineshape,
}


def make_provider(cfg, dtype=None):
    """Rebuild a provider from its :meth:`config` dict."""
    cfg = dict(cfg)
    typ = cfg.get("type")
    if typ not in _PROVIDERS:
        raise ValueError(
            f"unknown lineshape provider type '{typ}'; " f"known: {sorted(_PROVIDERS)}"
        )
    kw = {} if dtype is None else {"dtype": dtype}
    return _PROVIDERS[typ].from_config(cfg, **kw)


__all__ = ["ZGammaLineshape", "make_provider"]
