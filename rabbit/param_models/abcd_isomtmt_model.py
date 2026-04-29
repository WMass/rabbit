"""
IsoMT convenience wrappers for the four ABCD param models.

Hardcode the axis conventions used in CMS WMass nonprompt background
estimation:
  - y-axis: "relIso"  (bin 0 = pass/signal, bin 1 = fail/sideband)
  - x-axis: "mt"      (last bin = signal, earlier bins = sidebands)
  - smoothing axis: "pt"  (for the smooth variants only)

All four wrappers accept a single channel name and derive the per-region
channel dicts automatically. The extended variants require at least 3 mt bins;
the plain variants require at least 2.

CLI syntax:
    --paramModel ABCDIsoMT               <process> <channel>
    --paramModel ExtendedABCDIsoMT       <process> <channel>
    --paramModel SmoothABCDIsoMT         [order:N] <process> <channel>
    --paramModel SmoothExtendedABCDIsoMT [order:N] <process> <channel>

Region layout (shared by all four):

             relIso=signal(0)   relIso=sideband(1)
mt=extra-sb:       Bx                  Ax
mt=sideband:       B                   A
mt=signal:         D  ← predicted      C
"""

import h5py

from rabbit.param_models.abcd_model import ABCD
from rabbit.param_models.extended_abcd_model import ExtendedABCD
from rabbit.param_models.smooth_abcd_model import SmoothABCD
from rabbit.param_models.smooth_extended_abcd_model import SmoothExtendedABCD


def _isomtmt_regions(indata, channel_name):
    """Return (mt_signal_idx, mt_sideband_idx, mt_extra_sideband_idx) from the channel's mt axis size."""
    axes = indata.channel_info[channel_name]["axes"]
    mt_ax = next((a for a in axes if a.name == "mt"), None)
    if mt_ax is None:
        raise ValueError(
            f"Channel '{channel_name}' has no axis named 'mt'. "
            f"Available axes: {[a.name for a in axes]}"
        )
    if next((a for a in axes if a.name == "relIso"), None) is None:
        raise ValueError(
            f"Channel '{channel_name}' has no axis named 'relIso'. "
            f"Available axes: {[a.name for a in axes]}"
        )
    return mt_ax.size - 1, mt_ax.size - 2, mt_ax.size - 3


def _parse_isomtmt_args(tokens):
    """Parse [order:N] <process> <channel> from a token list."""
    order = 1
    if tokens and tokens[0].startswith("order:"):
        order = int(tokens.pop(0).split(":", 1)[1])
    if len(tokens) < 2:
        raise ValueError("Expected <process> <channel>")
    process = tokens.pop(0)
    channel = tokens.pop(0)
    if tokens:
        raise ValueError(f"Unexpected extra arguments: {tokens}")
    return order, process, channel


def _parse_isomtmt_args_with_params(tokens):
    """Parse [params:file.hdf5 | order:N] <process> <channel> from a token list.

    ``params:`` and ``order:`` are mutually exclusive; the params file encodes
    the polynomial order via its stored ``order`` field.
    """
    initial_params = None
    order = 1
    if tokens and tokens[0].startswith("params:"):
        params_file = tokens.pop(0).split(":", 1)[1]

        with h5py.File(params_file, mode="r") as f:
            initial_params = f["params"][...]
            order = f["order"][...]

    elif tokens and tokens[0].startswith("order:"):
        order = int(tokens.pop(0).split(":", 1)[1])
    if len(tokens) < 2:
        raise ValueError("Expected <process> <channel>")
    process = tokens.pop(0)
    channel = tokens.pop(0)
    if tokens:
        raise ValueError(f"Unexpected extra arguments: {tokens}")
    return order, process, channel, initial_params


class ABCDIsoMT(ABCD):
    """
    ABCD in the (mt × relIso) plane for a single channel.

    Uses the last two mt bins as sideband / signal and relIso bins 0 / 1 as
    signal / sideband.  All other axes in the channel become free per-bin
    parameters (outer axes of the ABCD model).
    """

    def __init__(self, indata, abcd_process, channel_name, **kwargs):
        mt_s, mt_sb, _ = _isomtmt_regions(indata, channel_name)
        if mt_sb < 0:
            raise ValueError(f"Channel '{channel_name}' mt axis has fewer than 2 bins")
        channel_A = {channel_name: {"relIso": 1, "mt": mt_sb}}
        channel_B = {channel_name: {"relIso": 0, "mt": mt_sb}}
        channel_C = {channel_name: {"relIso": 1, "mt": mt_s}}
        channel_D = {channel_name: {"relIso": 0, "mt": mt_s}}
        super().__init__(
            indata, abcd_process, channel_A, channel_B, channel_C, channel_D, **kwargs
        )

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        tokens = list(args)
        if len(tokens) < 2:
            raise ValueError("ABCDIsoMT expects: <process> <channel>")
        process, channel = tokens[0], tokens[1]
        if len(tokens) > 2:
            raise ValueError(f"Unexpected extra arguments: {tokens[2:]}")
        return cls(indata, process, channel, **kwargs)


class ExtendedABCDIsoMT(ExtendedABCD):
    """
    ExtendedABCD in the (mt × relIso) plane for a single channel.

    Uses the last three mt bins as extra-sideband / sideband / signal and
    relIso bins 0 / 1 as signal / sideband.
    """

    def __init__(self, indata, abcd_process, channel_name, **kwargs):
        mt_s, mt_sb, mt_xsb = _isomtmt_regions(indata, channel_name)
        if mt_xsb < 0:
            raise ValueError(f"Channel '{channel_name}' mt axis has fewer than 3 bins")
        channel_Ax = {channel_name: {"relIso": 1, "mt": mt_xsb}}
        channel_Bx = {channel_name: {"relIso": 0, "mt": mt_xsb}}
        channel_A = {channel_name: {"relIso": 1, "mt": mt_sb}}
        channel_B = {channel_name: {"relIso": 0, "mt": mt_sb}}
        channel_C = {channel_name: {"relIso": 1, "mt": mt_s}}
        channel_D = {channel_name: {"relIso": 0, "mt": mt_s}}
        super().__init__(
            indata,
            abcd_process,
            channel_A,
            channel_B,
            channel_C,
            channel_D,
            channel_Ax,
            channel_Bx,
            **kwargs,
        )

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        tokens = list(args)
        if len(tokens) < 2:
            raise ValueError("ExtendedABCDIsoMT expects: <process> <channel>")
        process, channel = tokens[0], tokens[1]
        if len(tokens) > 2:
            raise ValueError(f"Unexpected extra arguments: {tokens[2:]}")
        return cls(indata, process, channel, **kwargs)


class SmoothABCDIsoMT(SmoothABCD):
    """
    SmoothABCD in the (mt × relIso) plane, smoothed along the pt axis.

    Uses the last two mt bins and relIso bins 0 / 1.  The pt axis is the
    smoothing axis; all remaining axes become outer axes.
    """

    def __init__(self, indata, abcd_process, channel_name, order=1, **kwargs):
        mt_s, mt_sb, _ = _isomtmt_regions(indata, channel_name)
        if mt_sb < 0:
            raise ValueError(f"Channel '{channel_name}' mt axis has fewer than 2 bins")
        channel_A = {channel_name: {"relIso": 1, "mt": mt_sb}}
        channel_B = {channel_name: {"relIso": 0, "mt": mt_sb}}
        channel_C = {channel_name: {"relIso": 1, "mt": mt_s}}
        channel_D = {channel_name: {"relIso": 0, "mt": mt_s}}
        super().__init__(
            indata,
            "pt",
            abcd_process,
            channel_A,
            channel_B,
            channel_C,
            channel_D,
            order=order,
            **kwargs,
        )

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        tokens = list(args)
        order, process, channel = _parse_isomtmt_args(tokens)
        return cls(indata, process, channel, order=order, **kwargs)


class SmoothExtendedABCDIsoMT(SmoothExtendedABCD):
    """
    SmoothExtendedABCD in the (mt × relIso) plane, smoothed along the pt axis.

    Uses the last three mt bins and relIso bins 0 / 1.  The pt axis is the
    smoothing axis; all remaining axes become outer axes.

    CLI syntax:
        --paramModel SmoothExtendedABCDIsoMT [params:file.hdf5 | order:N] <process> <channel>

    ``params:file.hdf5`` loads initial parameter values and polynomial order from a
    file produced by setupRabbit.py --dumpSmoothingParams.  It is mutually
    exclusive with ``order:N``.
    """

    def __init__(self, indata, abcd_process, channel_name, order=1, **kwargs):
        mt_s, mt_sb, mt_xsb = _isomtmt_regions(indata, channel_name)
        if mt_xsb < 0:
            raise ValueError(f"Channel '{channel_name}' mt axis has fewer than 3 bins")
        channel_Ax = {channel_name: {"relIso": 1, "mt": mt_xsb}}
        channel_Bx = {channel_name: {"relIso": 0, "mt": mt_xsb}}
        channel_A = {channel_name: {"relIso": 1, "mt": mt_sb}}
        channel_B = {channel_name: {"relIso": 0, "mt": mt_sb}}
        channel_C = {channel_name: {"relIso": 1, "mt": mt_s}}
        channel_D = {channel_name: {"relIso": 0, "mt": mt_s}}
        super().__init__(
            indata,
            "pt",
            abcd_process,
            channel_A,
            channel_B,
            channel_C,
            channel_D,
            channel_Ax,
            channel_Bx,
            order=order,
            **kwargs,
        )

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        tokens = list(args)
        order, process, channel, initial_params = _parse_isomtmt_args_with_params(
            tokens
        )
        return cls(
            indata,
            process,
            channel,
            order=order,
            initial_params=initial_params,
            **kwargs,
        )
