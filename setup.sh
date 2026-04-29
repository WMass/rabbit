export RABBIT_BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export PYTHONPATH="${RABBIT_BASE}:$PYTHONPATH"
export PATH="$PATH:${RABBIT_BASE}/bin"

# Enable XLA's multi-threaded Eigen path on CPU. ~1.3x speedup on dense
# large-model HVP/loss+grad on many-core systems, no downside on smaller
# problems. Append to any existing XLA_FLAGS so user-set flags survive.
if [[ ":${XLA_FLAGS:-}:" != *":--xla_cpu_multi_thread_eigen=true:"* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=true"
fi

echo "Created environment variable RABBIT_BASE=${RABBIT_BASE}"
