# Google Colab ↔ Local Runtime — Setup Guide (macOS/Windows/Linux)

This guide shows how to run a Colab notebook against your local machine so that code executes on your own CPU/GPU and uses your local files and Python environment. For this project, many RL training steps (policy updates, environment stepping, logging) are CPU‑bound. Running them locally typically reduces latency and speeds up training compared to Colab’s remote backend.

This document works with both installation methods used in this repo:

- Using Anaconda/conda (see docs/installation_anaconda.md)
- Using Python venv + pip (see docs/installation.md)

Under the hood, Colab connects via a small bridge package to a local Jupyter Notebook server. That bridge currently targets the classic Jupyter stack, so we pin to Notebook 6.x and Jupyter Server 1.x below.

---

## TL;DR (Local Runtime)

Pick one path that matches how you installed Python. These commands need to be run from the repository root.

### Option A — With Anaconda (conda)

```bash
# 0) Create/activate env (if not already)
# conda create -n gym python=3.10 -y
conda activate gym

# 1) Install classic Jupyter compatible with Colab bridge
conda install -y -c conda-forge "notebook<7" "jupyter_server<2" ipykernel
python -m pip install -U jupyter_http_over_ws

# 2) Enable the bridge
python -m jupyter server extension enable --py jupyter_http_over_ws

# 3) Launch classic Notebook (keep this terminal open)
python -m jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --NotebookApp.allow_credentials=True \
  --port=8888 --NotebookApp.port_retries=0
```

### Option B — Without Anaconda (python venv + pip)

```bash
# 0) Create/activate venv (adjust path/name as you like)
# python -m venv venv_for_ai4rgym
source venv_for_ai4rgym/bin/activate   # Windows PowerShell: .\venv_for_ai4rgym\Scripts\Activate.ps1

# 1) Install classic Jupyter compatible with Colab bridge
python -m pip install --upgrade pip setuptools
python -m pip install "notebook<7" "jupyter_server<2" ipykernel jupyter_http_over_ws

# 2) Enable the bridge
python -m jupyter server extension enable --py jupyter_http_over_ws

# 3) Launch classic Notebook (keep this terminal open)
python -m jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --NotebookApp.allow_credentials=True \
  --port=8888 --NotebookApp.port_retries=0
```

In Colab, click the Connect dropdown (top‑right) → “Connect to local runtime…” → enter `http://localhost:8888/` (or paste the full tokenized URL printed in your terminal).

---

## Step‑by‑Step

You can follow the detailed installation first, then return here to connect from Colab.

### 1) Install this repo and dependencies

- With Anaconda (**Recommended**): follow docs/installation_anaconda.md
- Without Anaconda: follow docs/installation.md

The following commands are expected to be run from the ai4r-gym directory root.

First ensure you can run `python -c "import ai4rgym; print('ok')"` in your environment.

### 2) Install the Colab bridge and classic Jupyter

Conda users:
```bash
conda install -y -c conda-forge "notebook<7" "jupyter_server<2" ipykernel
python -m pip install -U jupyter_http_over_ws
```

Venv + pip users:
```bash
python -m pip install --upgrade pip setuptools
python -m pip install "notebook<7" "jupyter_server<2" ipykernel jupyter_http_over_ws
```

Why classic? The `jupyter_http_over_ws` bridge expects classic Notebook APIs (e.g. `notebook.base`) that moved in Notebook 7/Jupyter Server 2.

### 3) Enable the server extension

```bash
python -m jupyter server extension enable --py jupyter_http_over_ws
python -m jupyter server extension list
```

You should see `jupyter_http_over_ws` listed as enabled.

### 4) Launch Jupyter with Colab‑friendly flags

```bash
python -m jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --NotebookApp.allow_credentials=True \
  --port=8888 --NotebookApp.port_retries=0
```

Keep this terminal open. If a `?token=...` appears, paste that entire URL into Colab’s dialog.

### 5) Connect from Colab

- Open a Colab notebook.
- Connect dropdown → “Connect to local runtime…”.
- Enter `http://localhost:8888/` (or paste the full `http://127.0.0.1:8888/?token=...`).

Optional: expose your env as a named kernel so it’s selectable inside Notebook/Lab:
```bash
python -m ipykernel install --user --name gym --display-name "Python (gym)"
```

---

## Notes on Performance

- Much of the RL training loop (policy updates, environment stepping) is CPU‑bound; local runtimes avoid network round‑trips and improve iteration speed.

---

## Platform Notes

This approach has been tested on macOS. The same approach applies on Linux and Windows with minor command/quoting differences:

- Windows PowerShell: activate venv with `.<path>\Scripts\Activate.ps1` and prefer double quotes for arguments, e.g.
  ```powershell
  python -m jupyter notebook \
    --NotebookApp.allow_origin="https://colab.research.google.com" \
    --NotebookApp.allow_credentials=True \
    --port=8888 --NotebookApp.port_retries=0
  ```
- Windows CMD: activate venv with `<path>\Scripts\activate.bat` and run the same `python -m jupyter notebook ...` line.
- Linux: commands match macOS; ensure firewall allows local ports when prompted.
- WSL: prefer launching the browser from Windows; still use `http://127.0.0.1:8888/` in Colab.

---

## Troubleshooting

### A) ModuleNotFoundError: No module named 'notebook.base'
You are on Notebook 7 / Server 2. Pin to classic:
```bash
conda install -y -c conda-forge "notebook<7" "jupyter_server<2"  # conda
# or
python -m pip install "notebook<7" "jupyter_server<2"            # venv/pip
```

### B) “Jupyter Server Terminals requires Jupyter Server 2.0+”
An extension for Server 2 is present. Remove it and re‑enable the bridge:
```bash
python -m pip uninstall -y jupyter_server_terminals
python -m jupyter server extension enable --py jupyter_http_over_ws
```

### C) Colab can’t connect / CORS
Ensure both flags are present when launching Notebook:
```bash
--NotebookApp.allow_origin='https://colab.research.google.com' \
--NotebookApp.allow_credentials=True
```
Try a different port (e.g. `--port=8890`) and use that in Colab.

### D) “Validation failed: The module … could not be found”
Make sure you installed the bridge in the same env that runs `jupyter` and call commands via `python -m ...` from that env.

### E) Missing “Connect to local runtime” UI
Refresh the page; try Chrome. In some UIs it appears under Runtime → Change runtime type.

---

## Useful Diagnostics

```bash
# Show interpreter and pip
which python; which pip; python -V; pip -V

# Check versions of classic stack
python -c "import notebook, jupyter_server; print('notebook', notebook.__version__); print('server', jupyter_server.__version__)"

# List server extensions
python -m jupyter server extension list
```

---

## Security Notes

- The `--allow_origin` flag scopes cross‑origin access to Colab only.
- Keep the default token; only set `--NotebookApp.token=''` on trusted local machines.
- Bind to localhost; do not expose publicly.

---

### Done

Once connected, Colab runs against your local environment (packages, CPU/GPU, files) while keeping the familiar Colab UI.
