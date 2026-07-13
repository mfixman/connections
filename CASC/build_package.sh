#!/bin/bash
# =============================================================================
# build_package.sh — assemble the SATResetCoP StarExec/CASC solver package.
#
# Produces a self-contained, relocatable bundle that runs on StarExec's
# (historically old-glibc CentOS) nodes without any system dependencies:
#
#   * CPython + python-sat  (portable SAT backend, replaces the native pydical)
#   * SWI-Prolog            (leanCoP FOF -> CNF clausification)
#   * the SATResetCoP prover source (already copied into SATResetCoP/src)
#
# Everything is fetched from conda-forge with a static micromamba, so the build
# needs no root and no pre-installed conda.  conda-forge packages target
# glibc 2.17 (manylinux2014), which is why the result is portable to old nodes.
#
# Usage:
#   bash CASC/build_package.sh                # build + tar
#   PY_VERSION=3.12 bash CASC/build_package.sh # pin a different Python
#
# Output:
#   CASC/SATResetCoP/python/          the bundled runtime
#   CASC/SATResetCoP.tar.gz           upload this to StarExec
# =============================================================================
set -euo pipefail

PY_VERSION="${PY_VERSION:-3.13}"
PLATFORM="${PLATFORM:-linux-64}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$HERE/SATResetCoP"
ENV_PREFIX="$PKG/python"
BUILD="$HERE/.build"
MM="$BUILD/micromamba/bin/micromamba"
TARBALL="$HERE/SATResetCoP.tar.gz"

mkdir -p "$BUILD"

# ---------------------------------------------------------------------------
# 1. Fetch a static micromamba (single binary, no install needed).
# ---------------------------------------------------------------------------
if [ ! -x "$MM" ]; then
  echo ">> downloading micromamba ..."
  mkdir -p "$BUILD/micromamba"
  # The releases tarball unpacks to bin/micromamba
  curl -Ls "https://micro.mamba.pm/api/micromamba/${PLATFORM}/latest" \
    | tar -xj -C "$BUILD/micromamba" bin/micromamba
fi

# ---------------------------------------------------------------------------
# 2. Create the runtime env: Python + SWI-Prolog + pip, from conda-forge.
# ---------------------------------------------------------------------------
echo ">> creating runtime environment (python=$PY_VERSION, swi-prolog) ..."
rm -rf "$ENV_PREFIX"
"$MM" create -y -p "$ENV_PREFIX" \
  -c conda-forge \
  "python=${PY_VERSION}" swi-prolog pip

# ---------------------------------------------------------------------------
# 3. Install the portable SAT backend into the bundled Python.
# ---------------------------------------------------------------------------
echo ">> installing python-sat into the bundle ..."
"$ENV_PREFIX/bin/python" -m pip install --no-input python-sat

# ---------------------------------------------------------------------------
# 3b. Bundle a glibc runtime + loader shims.
#
# The conda binaries are glibc, but the containerised StarExec job runner is
# Alpine/musl (docker/job-runner.Dockerfile) where glibc binaries do NOT load
# under gcompat. We ship a small glibc runtime (loader + core libs) and invoke
# Python/swipl through it via python/shim/*, with an explicit --library-path.
# This makes the bundle self-contained on ANY host libc (glibc SGE nodes AND
# the musl job runner). Requires docker/podman to extract the glibc runtime.
# ---------------------------------------------------------------------------
echo ">> bundling glibc runtime + loader shims ..."
ENGINE="$(command -v podman || command -v docker || true)"
[ -n "$ENGINE" ] || { echo "!! need docker/podman to extract a glibc runtime"; exit 1; }
rm -rf "$ENV_PREFIX/glibc"; mkdir -p "$ENV_PREFIX/glibc"
"$ENGINE" run --rm -v "$ENV_PREFIX/glibc:/out" docker.io/library/debian:12-slim bash -c '
  set -e
  for f in libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 librt.so.1 libutil.so.1 \
           libanl.so.1 libresolv.so.2 libnss_files.so.2 libnss_dns.so.2 \
           libcrypt.so.1 libz.so.1; do
    cp -L "/lib/x86_64-linux-gnu/$f" /out/ 2>/dev/null || true
  done
  cp -L /lib64/ld-linux-x86-64.so.2 /out/'
chmod 755 "$ENV_PREFIX/glibc"/*

# The shims restore a lost exec bit on the loader, and fall back to the host
# loader (fine on any glibc host) if the bundled one can't be executed.
mkdir -p "$ENV_PREFIX/shim"
cat > "$ENV_PREFIX/shim/python3" <<'SHIM'
#!/bin/sh
HERE=$(cd "$(dirname "$0")/.." && pwd)
LOADER="$HERE/glibc/ld-linux-x86-64.so.2"
[ -x "$LOADER" ] || chmod +x "$LOADER" 2>/dev/null
if [ -x "$LOADER" ]; then
  exec "$LOADER" \
    --library-path "$HERE/glibc:$HERE/lib" \
    "$HERE/bin/python3" -s -E "$@"
fi
[ -x "$HERE/bin/python3" ] || chmod +x "$HERE/bin/python3" 2>/dev/null
export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/bin/python3" -s -E "$@"
SHIM
cat > "$ENV_PREFIX/shim/swipl" <<'SHIM'
#!/bin/sh
HERE=$(cd "$(dirname "$0")/.." && pwd)
export SWI_HOME_DIR="$HERE/lib/swipl"
LOADER="$HERE/glibc/ld-linux-x86-64.so.2"
[ -x "$LOADER" ] || chmod +x "$LOADER" 2>/dev/null
if [ -x "$LOADER" ]; then
  exec "$LOADER" \
    --library-path "$HERE/glibc:$HERE/lib:$HERE/lib/swipl/lib/x86_64-linux" \
    "$HERE/lib/swipl/bin/x86_64-linux/swipl" "$@"
fi
[ -x "$HERE/lib/swipl/bin/x86_64-linux/swipl" ] || chmod +x "$HERE/lib/swipl/bin/x86_64-linux/swipl" 2>/dev/null
export LD_LIBRARY_PATH="$HERE/lib:$HERE/lib/swipl/lib/x86_64-linux${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/lib/swipl/bin/x86_64-linux/swipl" "$@"
SHIM
chmod +x "$ENV_PREFIX/shim/python3" "$ENV_PREFIX/shim/swipl"

# ---------------------------------------------------------------------------
# 4. Housekeeping: drop caches, ensure the placeholder TPTP root exists.
# ---------------------------------------------------------------------------
find "$PKG/src" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$PKG/tptp-root"
chmod +x "$PKG/bin/"starexec_run_* "$PKG/starexec_prover.py" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Self-test the bundle and its TPTP certificate on a trivial FOF theorem.
# ---------------------------------------------------------------------------
echo ">> self-test ..."
printf 'fof(a,axiom,![X]:(p(X)=>q(X))).\nfof(b,axiom,p(a)).\nfof(c,conjecture,q(a)).\n' \
  > "$BUILD/selftest+1.p"
"$ENV_PREFIX/shim/python3" "$PKG/starexec_prover.py" --proof --cpu-limit 30 "$BUILD/selftest+1.p" \
  | tee "$BUILD/selftest.out"
grep -q "SZS status Theorem" "$BUILD/selftest.out" \
  && grep -q "cnf(satresetcop_false,plain" "$BUILD/selftest.out" \
  && echo ">> self-test OK" \
  || { echo "!! self-test FAILED"; exit 1; }

# The self-test imports the staged source and recreates Python bytecode caches.
# They are unnecessary in the submission archive and make it look like a
# development checkout, so remove every cache after testing and before
# packaging.
find "$PKG" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$PKG" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.qlf' \) -delete 2>/dev/null || true
rm -rf "$ENV_PREFIX/var/cache"
rm -rf "$ENV_PREFIX/lib/python3.13/site-packages/pip" \
       "$ENV_PREFIX/lib/python3.13/site-packages"/pip-*.dist-info \
       "$ENV_PREFIX/bin/pip" "$ENV_PREFIX/bin/pip3" "$ENV_PREFIX/bin/pip3.13"

# ---------------------------------------------------------------------------
# 6. Package. The tar root must contain bin/ (StarExec requirement).
# ---------------------------------------------------------------------------
echo ">> creating $TARBALL ..."
tar -C "$PKG" -czf "$TARBALL" \
  bin src python starexec_prover.py tptp-root starexec_description.txt

echo
echo ">> done."
echo "   bundle : $ENV_PREFIX"
echo "   upload : $TARBALL"
du -sh "$TARBALL" 2>/dev/null || true
