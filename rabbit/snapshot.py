"""Parameter snapshots, so an interrupted fit is not a lost fit.

A long fit is fragile in a way its length makes expensive. Everything the
minimiser has learned lives in one in-memory vector until the very end: the
output file is written only after ``minimize()`` returns, so a fit that is
killed, hits a wall clock limit, or dies writing its output leaves nothing at
all behind, however close to the minimum it had got.

A snapshot is that vector on disk. It is deliberately the smallest thing that
:meth:`rabbit.fitter.Fitter.load_fitresult` will accept -- the parameter values
and their names -- so a fit can be resumed from one with

    rabbit_fit.py input.hdf5 -o out/ --externalPostfit snapshot.hdf5

either to carry on minimising or, with ``--noFit``, to run just the postfit
step at the snapshot point. No covariance is stored: it does not exist until
the Hessian is computed, and ``load_fitresult`` treats it as optional.

Two things here are not incidental.

*The write is atomic.* Snapshots exist for the case where the process dies at a
moment it did not choose, and that includes while a snapshot is being written.
Writing in place would then leave a truncated file where a good one used to be,
turning the safety net into the thing that destroys the result. Each snapshot
is written to a temporary file in the same directory and moved into place with
``os.replace``, which is atomic on POSIX, so a reader sees either the previous
snapshot or the new one.

*The values are physical.* Under preconditioning the minimiser works in
internal coordinates and the callback's iterate is in those coordinates; the
transform has to be undone before writing. A snapshot of internal coordinates
would load without complaint and be silently wrong.

Nothing here imports the fitter, and so nothing here imports TensorFlow --
worth keeping that way. It makes the module usable from a signal handler and
from a plain subprocess, where pulling in TensorFlow costs minutes.
"""

import contextlib
import os
import signal
import threading

import h5py
import numpy as np
from wums import logging

logger = logging.child_logger(__name__)


def write_snapshot(filename, parms, x, meta=None):
    """Write parameter values and names to ``filename``, atomically.

    ``parms`` and ``x`` must be aligned. ``meta`` is an optional dict of
    scalars stored as HDF5 attributes; it is provenance only and nothing reads
    it back, so a snapshot stays loadable if its contents ever change.
    """
    x = np.asarray(x)
    parms = np.asarray(parms).astype(str)
    if x.shape != parms.shape:
        raise ValueError(
            f"snapshot: {x.size} parameter values against {parms.size} names"
        )

    directory = os.path.dirname(os.path.abspath(filename))
    # NB same directory as the target: os.replace is only atomic within a
    # filesystem, and /tmp is routinely a different one
    tmp = os.path.join(directory, f".{os.path.basename(filename)}.tmp")

    with h5py.File(tmp, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset(
            "parms", data=parms.astype(object), dtype=h5py.special_dtype(vlen=str)
        )
        for key, value in (meta or {}).items():
            f.attrs[key] = value
    os.replace(tmp, filename)


class Snapshotter:
    """Decides when to snapshot and where to put it.

    Holds the mapping back to physical coordinates, so callers never have to
    remember to undo the preconditioner themselves.
    """

    def __init__(self, filename, parms, to_physical=None, interval_hours=0.0):
        self.filename = filename
        self.parms = parms
        self.to_physical = to_physical if to_physical is not None else (lambda v: v)
        # negative or zero disables the periodic snapshots; an explicit save
        # (a signal, a failing minimiser) is always honoured
        self.interval = float(interval_hours) * 3600.0
        self.last_write = None
        self.count = 0
        # most recent accepted iterate, already in physical coordinates
        self.latest = None

    def update(self, xval):
        """Record ``xval`` (internal coordinates) as the latest good point.

        Converting on the way in rather than on the way out is what makes a
        snapshot safe during a preconditioner rebuild: for the minutes that
        takes, the stored transform no longer matches the stored iterate, and
        anything converting lazily would write a vector mapped by the wrong
        one.
        """
        self.latest = np.asarray(self.to_physical(np.asarray(xval)))

    def _write(self, reason, **meta):
        """Never raises: the snapshot is insurance, not the product, so a
        failure here must not take down a fit that is otherwise fine."""
        if self.filename is None or self.latest is None:
            return False
        try:
            write_snapshot(
                self.filename, self.parms, self.latest, meta={"reason": reason, **meta}
            )
        except Exception as ex:  # pragma: no cover - defensive
            logger.warning(f"Could not write snapshot to {self.filename}: {ex}")
            return False
        self.count += 1
        logger.info(
            f"Wrote parameter snapshot ({reason}) to {self.filename}; resume with "
            f"--externalPostfit {self.filename}"
        )
        return True

    def save(self, xval, reason, **meta):
        """Snapshot ``xval`` unconditionally, whatever the interval says."""
        self.update(xval)
        return self._write(reason, **meta)

    def save_latest(self, reason, **meta):
        """Snapshot the last point handed to :meth:`update`.

        This is the signal path: it must not need an iterate passed in, since
        a signal can arrive at any point in an iteration -- and with iterations
        running to hours on a large fit, waiting for the next one is not a
        usable answer.
        """
        return self._write(reason, **meta)

    def maybe_save(self, xval, elapsed, reason="periodic", **meta):
        """Record the point, and snapshot it if the interval has passed.

        The first call always writes. That is deliberate: it proves the path is
        writable at iteration 1 rather than at hour 100, which is when it
        matters and far too late to learn otherwise.
        """
        self.update(xval)
        if self.filename is None or self.interval <= 0:
            return False
        if self.last_write is not None and elapsed - self.last_write < self.interval:
            return False
        self.last_write = elapsed
        return self._write(reason, elapsed=elapsed, **meta)


@contextlib.contextmanager
def snapshot_on_signal(snapshotter):
    """Write a snapshot when the job is asked to stop, then get out of the way.

    SIGTERM is how a batch scheduler announces a wall clock limit and how
    ``kill`` reaches a fit someone has decided to stop; SIGINT is Ctrl-C.
    Both otherwise destroy every parameter value the fit has found.

    Doing the write in the Python signal handler does not work here, which was
    learned the expensive way: a preempted 22-hour fit wrote 33 periodic
    snapshots and zero signal ones. Python runs signal handlers only in the
    main thread, between bytecodes, and this fit spends its time inside single
    TensorFlow calls that run for minutes -- one iteration took 910 s. SIGTERM
    arrived, the interpreter never regained control to dispatch it, and SIGKILL
    followed. A cooperative "stop at the next iteration" fails for the same
    reason, only worse.

    So the C-level handler writes the signal number to a pipe (that is all
    ``set_wakeup_fd`` does, and it happens immediately and without the GIL),
    and a daemon thread blocked on the read end does the snapshot. TensorFlow
    releases the GIL for the duration of a long op, so that thread runs while
    the main thread is still down in C++. The Python-level handler installed
    alongside is deliberately a no-op: its only job is to make the signal
    "handled" so that set_wakeup_fd reports it.

    The thread then exits the process itself with 128+signum, the encoding a
    shell reports for a signal death. It cannot re-raise the signal properly:
    that needs signal.signal() to restore the default disposition, and Python
    only allows that from the main thread -- the very thread that is stuck.
    """
    if snapshotter.filename is None:
        yield
        return

    read_fd, write_fd = os.pipe()
    # set_wakeup_fd requires a non-blocking write end -- it must never stall
    # inside the C handler; the reader blocks, which is what we want.
    os.set_blocking(write_fd, False)
    os.set_blocking(read_fd, True)

    previous = {}
    stopping = threading.Event()

    def _worker():
        while True:
            try:
                data = os.read(read_fd, 1)
            except OSError:
                return
            if not data or stopping.is_set():
                return  # the sentinel written on clean exit
            signum = data[0]
            snapshotter.save_latest(f"signal-{signal.Signals(signum).name}")
            # Terminate from here rather than re-raising. Restoring SIG_DFL
            # first would be the tidy way, but signal.signal() only works on
            # the main thread -- and the main thread is the one stuck in C,
            # which is why this thread exists at all. os._exit is immediate and
            # skips interpreter shutdown, which matters: the snapshot is
            # already on disk and a scheduler that sent SIGTERM is counting
            # down to SIGKILL. 128+signum is the conventional encoding a shell
            # reports for a signal death.
            os._exit(128 + signum)

    thread = None
    prev_wakeup = None

    def _noop(signum, frame):
        """Never does the work; see the class docstring. Present only so the
        signal counts as handled and set_wakeup_fd fires for it."""

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, _noop)
        prev_wakeup = signal.set_wakeup_fd(write_fd)
    except ValueError:
        # only the main thread may install handlers or a wakeup fd; off it the
        # periodic and failure snapshots still work, so carry on rather than
        # refusing to fit
        logger.debug("Could not arm signal snapshots (not the main thread)")
        for sig, prev in previous.items():
            signal.signal(sig, prev)
        previous.clear()
    else:
        thread = threading.Thread(
            target=_worker, name="snapshot-on-signal", daemon=True
        )
        thread.start()

    try:
        yield
    finally:
        stopping.set()
        if prev_wakeup is not None:
            signal.set_wakeup_fd(prev_wakeup)
        for sig, prev in previous.items():
            signal.signal(sig, prev)
        if thread is not None:
            try:
                os.write(write_fd, b"\x00")  # wake the reader so it can exit
            except OSError:
                pass
            thread.join(timeout=5.0)
        for fd in (write_fd, read_fd):
            try:
                os.close(fd)
            except OSError:
                pass
