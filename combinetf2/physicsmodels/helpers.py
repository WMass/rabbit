import importlib
import re

import hist
import numpy as np
import tensorflow as tf
from wums import boostHistHelpers as hh

from combinetf2 import tfhelpers

# dictionary with class name and the corresponding filename where it is defined
baseline_models = {
    "Basemodel": "physicsmodel",
    "Select": "physicsmodel",
    "Project": "project",
    "Normalize": "project",
    "Ratio": "ratio",
    "Normratio": "ratio",
    "AngularCoefficients": "angular_coefficients",
    "LamTung": "angular_coefficients",
}


def instance_from_class(class_name, *args, **kwargs):
    if "." in class_name:
        # import from full relative or abslute path
        parts = class_name.split(".")
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]
    else:
        # import one of the baseline models
        if class_name not in baseline_models:
            raise ValueError(
                f"Model {class_name} not found, available baseline models are {baseline_models.keys()}"
            )
        module_name = f"combinetf2.physicsmodels.{baseline_models[class_name]}"

    # Try to import the module
    module = importlib.import_module(module_name)

    model = getattr(module, class_name, None)
    if model is None:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'."
        )

    return model.parse_args(*args, **kwargs)


def parse_axis_selection(selection_str):
    """
    Parse a string specifying the axis selections in a dict where keys are axes and values the selections
    The input string is expected to have the format <axis_name_0>:<selection_0>,<axis_name_1>:<selection_1>,...
       i.e. a comma separated list of axis names and selections separated by ":", the selections can be indices, slice objects e.g. 'slice(0,2,2)', or special objects:
       - 'sum' to sum all bins of an axis
       - 'rebin()' to rebin an axis with new edges
       - 'None:None' for whch 'None' is returned, indicating no selection
    """
    sel = {}
    sum_axes = []
    rebin_axes = {}
    if selection_str != "None:None":
        selections = re.split(r",(?![^()]*\))", selection_str)
        for s in selections:
            sl = None
            k, v = s.split(":")
            if "slice" in v:
                slice_args = []
                for x in v[6:-1].split(","):
                    if x == "None":
                        slice_args.append(None)
                    elif "j" in x:
                        slice_args.append(complex(x))
                    else:
                        slice_args.append(int(x))
                sl = slice(*slice_args)
            elif v == "sum":
                sum_axes.append(k)
            elif v.startswith("rebin"):
                arr = np.fromstring(v[6:-1], sep=",", dtype=np.float32)
                rebin_axes[k] = arr
            elif v == "None":
                sl = slice(None)
            else:
                if "j" in v:
                    x = complex(v)
                else:
                    x = int(v)
                sl = slice(x, x + 1)
                # always sum/reduce this axis if only one bin is selected
                sum_axes.append(k)
            if sl is not None:
                sel[k] = sl

        for s in selections:
            if k not in sel.keys():
                sel[k] = slice(None)

    return sel, rebin_axes, sum_axes


class Term:
    def __init__(
        self,
        indata,
        channel,
        processes=[],
        selections={},
        rebin_axes={},
        sum_axes=[],
    ):
        info = indata.channel_info[channel]

        self.start = info["start"]
        self.stop = info["stop"]

        channel_axes = info["axes"]

        channel_axes_names = [a.name for a in channel_axes]

        self.has_data = not info["masked"] and len(processes) == 0

        self.exp_shape = tuple([len(a) for a in channel_axes])

        if processes is not None:
            if any(p not in indata.procs.astype(str) for p in processes):
                raise RuntimeError(
                    f"Not all selection processes found in channel. Selection processes: {processes}, Channel axes: {indata.procs}"
                )
            self.proc_idxs = [
                i for i, p in enumerate(indata.procs.astype(str)) if p in processes
            ]
        else:
            self.proc_idxs = None

        if selections is not None:
            if any(k not in channel_axes_names for k in selections.keys()):
                raise RuntimeError(
                    f"Not all selection axes found in channel. Selection axes: {selections.keys()}, Channel axes: {channel_axes_names}"
                )
            self.selections = tuple(
                [
                    selections[n] if n in selections.keys() else slice(None)
                    for i, n in enumerate(channel_axes_names)
                ]
            )
            self.selection_idxs = [
                i for i, n in enumerate(channel_axes_names) if n in selections.keys()
            ]
        else:
            self.selections = None
            self.selection_idxs = None

        # make dummy histogram to perform rebinning
        h = hist.Hist(*channel_axes)
        if self.selections:
            for k, s in selections.items():
                if isinstance(s, slice) and s.step is not None:
                    h = h[{k: slice(s.start, s.stop, hist.rebin(s.step))}]
                else:
                    h = h[{k: s}]

        self.segment_ids = {}
        self.num_segments = {}
        for n, new_edges in rebin_axes.items():
            old_edges = h.axes[n].edges
            h = hh.rebinHist(h, n, new_edges)
            segment_ids = np.zeros_like(old_edges[:-1])
            segment = -1
            for idx, v in enumerate(old_edges[:-1]):
                if any(np.isclose(v, x) for x in new_edges):
                    segment += 1

                segment_ids[idx] = segment

            if not np.isclose(new_edges[-1], old_edges[-1]):
                segment_ids[segment_ids == segment] = -1

            key = h.axes.name.index(n)
            self.segment_ids[key] = tf.constant(segment_ids, dtype=tf.int64)
            self.num_segments[key] = int(max(segment_ids) + 1)

        channel_axes = [a for a in h.axes if a.name not in sum_axes]

        self.sum_idxs = [i for i, n in enumerate(h.axes.name) if n in sum_axes]

        self.channel_axes = channel_axes

    def select(self, values, normalize=False, inclusive=True):
        values = values[self.start : self.stop]

        if len(values.shape) == 1:
            values = tf.reshape(values, self.exp_shape)
        else:
            if self.proc_idxs:
                values = tf.gather(values, indices=self.proc_idxs, axis=-1)
            values = tf.reshape(values, (*self.exp_shape, values.shape[1]))
            if inclusive:
                values = tf.reduce_sum(values, axis=-1)

        if self.selections:
            values = values[self.selections]

        if len(self.segment_ids):
            for i, s in self.segment_ids.items():
                values = tfhelpers.segment_sum_along_axis(
                    values, s, i, num_segments=self.num_segments[i]
                )

        if len(self.sum_idxs):
            values = tf.reduce_sum(values, axis=self.sum_idxs)

        if normalize:
            norm = tf.reduce_sum(values)
            values /= norm

        return values
