export TF_ENABLE_ONEDNN_OPTS=0 
export RABBIT_BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export PYTHONPATH="${RABBIT_BASE}:$PYTHONPATH"
export PATH="$PATH:${RABBIT_BASE}/bin"

echo "Created environment variable RABBIT_BASE=${RABBIT_BASE}"
