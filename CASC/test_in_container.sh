#!/bin/bash
# =============================================================================
# test_in_container.sh — run the SATResetCoP bundle inside the SAME runtime the
# StarExec 'containerised' branch uses for jobs: Alpine 3.21 (musl libc) with
# the gcompat glibc shim + libstdc++/libgcc, exactly as docker/job-runner.Dockerfile.
#
# This is the decisive local test: our bundle ships GLIBC binaries (conda-forge
# Python + SWI-Prolog), and they must run under Alpine's musl+gcompat. If Python
# or swipl fail to load here, the glibc bundle is NOT compatible with the job
# runner and must be rebuilt (Alpine-native / musl).
#
# Prereq: a running Docker (or Podman) daemon.  If Docker is inactive:
#     sudo systemctl start docker
#
# Usage:
#     bash CASC/test_in_container.sh [problem.p] [limit-seconds]
# =============================================================================
set -u
CASC="$(cd "$(dirname "$0")" && pwd)"
TARBALL="$CASC/SATResetCoP.tar.gz"
PROB="${1:-$HOME/TP/TPTP/Problems/PUZ/PUZ001+1.p}"
LIMIT="${2:-60}"
ENGINE="$(command -v podman || command -v docker)"

[ -n "$ENGINE" ]  || { echo "no docker/podman found"; exit 1; }
[ -f "$TARBALL" ] || { echo "missing $TARBALL — run build_package.sh first"; exit 1; }
[ -f "$PROB" ]    || { echo "benchmark not found: $PROB"; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
tar xzf "$TARBALL" -C "$WORK"

echo ">> engine   : $ENGINE"
echo ">> benchmark: $PROB"
echo ">> limit    : ${LIMIT}s"
echo ">> base     : alpine:3.21 + gcompat (mirrors job-runner.Dockerfile)"
echo

"$ENGINE" run --rm \
  -v "$WORK:/starexec/solver:ro" \
  -v "$PROB:/starexec/input/benchmark:ro" \
  -e STAREXEC_WALLCLOCK_LIMIT="$LIMIT" \
  -e STAREXEC_CPU_LIMIT="$LIMIT" \
  docker.io/library/alpine:3.21 sh -c '
    set -e
    # Mirror docker/job-runner.Dockerfile packages.
    apk add --no-cache bash coreutils gcompat libstdc++ libgcc >/dev/null 2>&1
    echo "--- [1] bundled Python via glibc-loader shim ---"
    /starexec/solver/python/shim/python3 --version || echo "!! PYTHON SHIM FAILED"
    echo "--- [2] bundled SWI-Prolog via glibc-loader shim ---"
    /starexec/solver/python/shim/swipl --version   || echo "!! SWIPL SHIM FAILED"
    echo "--- [3] full run as job-entrypoint.sh calls it (benchmark = STAREXEC default name) ---"
    mkdir -p /starexec/output_files
    bash /starexec/solver/bin/starexec_run_default \
      /starexec/input/benchmark /starexec/output_files
  '
rc=$?
echo
echo ">> container exit=$rc"
echo ">> PASS if step [3] printed a '% SZS status ...' line."
