# Google Colab ↔ Local Runtime (Conda + Classic Jupyter) — Setup Guide (macOS/Linux)

This is a **battle‑tested path** to run Google Colab notebooks on **your local Conda environment**. It reflects what actually worked: using the **classic Jupyter Notebook 6.x** + **Jupyter Server 1.x** with the \`\` bridge, and avoiding newer Server‑2‑only extensions that break validation.

---

## TL;DR (Copy‑paste, no prompts)

```bash
# 0) Activate your conda env
conda activate gym

# 1) Install classic stack compatible with the Colab bridge
conda install -y -c conda-forge "notebook<7" "jupyter_server<2" ipykernel
python -m pip install --no-cache-dir -U jupyter_http_over_ws

# 2) Enable the bridge
python -m jupyter server extension enable --py jupyter_http_over_ws

# 3) Launch classic Notebook (keep this running)
python -m jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --NotebookApp.allow_credentials=True \
  --port=8888 --NotebookApp.port_retries=0
```

In Colab, click the **Connect** dropdown (top‑right) → **Connect to local runtime…** → enter `http://localhost:8888/` (or include the `?token=...` URL shown in your terminal if prompted).

---

## Why this specific setup?

- The Colab bridge package \`\` currently expects **classic Notebook APIs** (e.g., `notebook.base`). Those were removed/relocated in **Notebook 7 / Jupyter Server 2**, so you’ll see errors like `ModuleNotFoundError: No module named 'notebook.base'` if you use the modern stack.
- Therefore, we pin to `and`. This combination works reliably with the bridge.
- Some extensions (e.g., \`\`) **require Server 2.0+** and will crash validation on Server 1.x. Uninstall or disable them for this workflow.

---

## Full Setup (step‑by‑step)

### 1) Create/activate a Conda environment

```bash
conda create -n gym python=3.10 -y
conda activate gym
```

(Use your existing env if you already have one.)

### 2) Install compatible Jupyter stack and the Colab bridge

```bash
conda install -y -c conda-forge "notebook<7" "jupyter_server<2" ipykernel
python -m pip install --no-cache-dir -U jupyter_http_over_ws
```

### 3) Verify the bridge is importable (optional but reassuring)

```bash
python - <<'PY'
import sys, pkgutil, jupyter_http_over_ws as m
print("py:", sys.executable)
print("module found?", pkgutil.find_loader('jupyter_http_over_ws') is not None)
print("module path:", m.__file__)
PY
```

You should see the module path under your conda env.

### 4) Enable the server extension (use the env’s Python)

```bash
python -m jupyter server extension enable --py jupyter_http_over_ws
python -m jupyter server extension list
```

`jupyter_http_over_ws` should show **enabled**.

> If the list command crashes on an unrelated extension, see **Troubleshooting** below.

### 5) Launch **classic Notebook** with Colab‑friendly flags

```bash
python -m jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --NotebookApp.allow_credentials=True \
  --port=8888 --NotebookApp.port_retries=0
```

- Keep this terminal open.
- If a token is printed, you can paste the full URL (including `?token=...`) into Colab’s dialog.

### 6) Connect from Google Colab

- Open any Colab notebook.
- Click the drop-down button beside the **Connect** button (top‑right) → **Connect to local runtime…**.
- Enter `http://localhost:8888/` (or paste the full tokenized URL from your terminal).

> **Tip:** If you don’t see the option, reload the page or try Chrome. The option may be under **Runtime → Change runtime type** in some UIs.

---

## Optional: expose the env as a named kernel

This lets you switch kernels cleanly inside Notebook/Lab.

```bash
python -m ipykernel install --user --name gym --display-name "Python (gym)"
```

---

## Troubleshooting (real errors we hit & fixes)

### A) `ModuleNotFoundError: No module named 'notebook.base'`

**Cause:** You’re on **Notebook 7 / Server 2**, but the bridge expects classic Notebook 6.x. **Fix:**

```bash
conda install -y -c conda-forge "notebook<7" "jupyter_server<2"
```

### B) `Jupyter Server Terminals requires Jupyter Server 2.0+`

**Cause:** `jupyter_server_terminals` is installed but you’re on Server 1.x. The *extension manager* dies while validating it. **Fix (remove the landmine):**

```bash
python -m pip uninstall -y jupyter_server_terminals
rm -f $(python - <<'PY'
from jupyter_core.paths import jupyter_config_path
import os
for p in jupyter_config_path():
    d=os.path.join(p,'jupyter_server_config.d')
    if os.path.isdir(d):
        f=os.path.join(d,'jupyter_server_terminals.json')
        print(f)
PY)
```

Re‑run the enable/list steps for the bridge.

### C) `jupyter command 'jupyter-serverextension' not found`

**Cause:** Old one‑word subcommand. **Fix:** Use the new form: `jupyter server extension ...` (we always call it via `python -m jupyter ...`).

### D) “Validation failed: The module … could not be found”

**Cause:** The package isn’t installed in the **same env** that `jupyter` is using. **Fix:** Always use the env’s Python:

```bash
which python; which pip
python -m pip install -U jupyter_http_over_ws
python -m jupyter server extension enable --py jupyter_http_over_ws
```

### E) Colab can’t connect / CORS issues

**Fix:** Ensure you launched with both flags:

```bash
--NotebookApp.allow_origin='https://colab.research.google.com' \
--NotebookApp.allow_credentials=True
```

Try another port if 8888 is busy: `--port=8890` and use that in Colab.

### F) Missing “Connect to local runtime” UI

- Refresh the page; try **Chrome**.
- Look under **Connect** dropdown (top‑right). In some UIs it’s under **Runtime → Change runtime type**.
- Firefox users may need `about:config` → set `network.websocket.allowInsecureFromHTTPS=true`.

### G) macOS firewall prompt

Click **Allow** when you first launch the server.

### H) Shell parse weirdness when copy‑pasting

Avoid pasting prompts like `(gym) …` and keep commands as plain lines. If using comments, put them on their own lines.

### I) Extension list still crashes on Server‑2‑only add‑ons

Temporarily remove modern add‑ons you don’t need for Colab:

```bash
python -m pip uninstall -y jupyter_server_ydoc jupyter_server_fileid jupyterlab
```

(You can reinstall later.)

---

## Useful diagnostics

```bash
# Show which Python/pip you’re using
which python; which pip; python -V; pip -V

# Check versions
python -c "import notebook, jupyter_server; print('notebook', notebook.__version__); print('server', jupyter_server.__version__)"

# See config search paths
python - <<'PY'
from jupyter_core.paths import jupyter_config_path
print("Jupyter config paths (highest priority first):")
for p in jupyter_config_path():
    print(" -", p)
PY

# List server extensions (after cleaning up incompatible ones)
python -m jupyter server extension list
```

---

## Security notes

- The `--allow_origin` flag tightly scopes cross‑origin access to **Colab only**.
- Consider keeping the default token auth; only set `--NotebookApp.token=''` on **trusted, local** setups.
- Keep the server bound to localhost; do not expose it publicly.

---

## Appendix: Why classic Notebook (6.x) instead of 7?

`jupyter_http_over_ws` integrates with the classic Notebook’s Tornado handlers (e.g., `notebook.base`). Notebook 7 is rebuilt atop **Jupyter Server 2** with different module layout; until the bridge is updated, **Notebook 6.x + Server 1.x** is the stable route for Colab local runtimes.

---

### Done ✨

Once connected, Colab uses your local Conda env (packages, GPU, files) while keeping the familiar Colab UI.

