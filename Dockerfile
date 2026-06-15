# =============================================================================
# Dockerfile – Foxy Anomaly Detection (Streamlit)
# Base: python:3.11-slim (glibc, pre-built wheels untuk scipy/numpy/sklearn)
# ALASAN tidak pakai Alpine: scipy, numpy, scikit-learn butuh glibc.
#   Alpine pakai musl → tidak ada pre-built manylinux wheel → harus compile
#   dari source → image lebih besar & build lebih lama.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: builder
# Tujuan: install semua Python dependency ke dalam isolated venv.
# Build tools (gcc, g++) HANYA ada di stage ini — tidak masuk ke runtime image.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies:
# - gcc, g++      : compile C-extension packages yang tidak punya pre-built wheel
# - libgomp-dev   : OpenMP header, dibutuhkan saat compile scikit-learn/numpy
# Semua ini akan dibuang setelah stage ini selesai (tidak ada di final image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Buat isolated virtual environment
# Manfaat: mudah di-copy antar stage, tidak mencemari system Python
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip agar resolusi wheel paling optimal
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── LAYER CACHE TRICK ──────────────────────────────────────────────────────
# Copy requirements.txt DULU, baru install.
# Jika app.py berubah tapi requirements.txt tidak, layer ini tidak di-rebuild.
# Ini menghemat waktu build secara signifikan.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: runtime (final image)
# Tujuan: image sekecil mungkin, hanya berisi apa yang dibutuhkan saat runtime.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Foxy Anomaly Detection"
LABEL org.opencontainers.image.description="Streamlit anomaly detection dashboard – production image"
LABEL org.opencontainers.image.base.name="python:3.11-slim"

WORKDIR /app

# Copy compiled virtual environment dari builder stage
# Tidak ada gcc/g++ di sini — attack surface minimal
COPY --from=builder /venv /venv

# Install HANYA runtime system library:
# - libgomp1 : OpenMP runtime library untuk scikit-learn parallel computation
#              (dynamic linking, dibutuhkan saat runtime bukan compile time)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ── Python environment flags ──────────────────────────────────────────────────
# PYTHONDONTWRITEBYTECODE : tidak buat .pyc files → image lebih bersih
# PYTHONUNBUFFERED        : output langsung ke stdout/stderr (penting untuk log)
# PYTHONFAULTHANDLER      : print Python traceback saat crash
ENV PATH="/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# ── Streamlit server configuration ────────────────────────────────────────────
# Menggunakan env vars (mirror dari .streamlit/config.toml) agar bisa
# di-override melalui docker-compose tanpa ubah kode.
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200 \
    STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200

# ── Matplotlib non-interactive backend ────────────────────────────────────────
# MPLBACKEND=Agg  : sudah diset di app.py, ini belt-and-suspenders
# MPLCONFIGDIR    : direktori cache matplotlib → arahkan ke /tmp agar
#                   non-root user bisa tulis tanpa perlu home directory writable
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# ── Copy source code (TERAKHIR untuk cache optimization) ─────────────────────
# Hanya app.py yang di-copy — tidak ada file lain yang disentuh
COPY app.py .

# ── Security: non-root user ───────────────────────────────────────────────────
# Explicit UID/GID 1001 untuk konsistensi di seluruh environment
# Pre-create /tmp subdirs agar tidak ada permission error saat startup
RUN groupadd -g 1001 appgroup \
    && useradd -u 1001 -g appgroup -M -s /sbin/nologin appuser \
    && mkdir -p /tmp/matplotlib \
    && chown -R appuser:appgroup /app /tmp/matplotlib

# Drop root — container berjalan sebagai non-privileged user
USER appuser

# Port internal Streamlit (TIDAK expose ke host langsung)
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
# /_stcore/health adalah endpoint built-in Streamlit (sejak v1.8+)
# start_period=60s karena import ML packages (numpy, scipy, sklearn) lambat
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=60s \
            --retries=3 \
    CMD python -c \
        "import urllib.request; \
         urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
        || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Exec form (bukan shell form) → PID 1 adalah streamlit → sinyal SIGTERM/SIGINT
# diterima langsung oleh proses, tidak hilang karena shell wrapper
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
