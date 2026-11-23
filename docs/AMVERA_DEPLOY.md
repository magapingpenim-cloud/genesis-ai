# Amvera deployment

The Amvera web app serves the static chat UI from the repository root. It deliberately does not import PyTorch: the chat frontend is not yet connected to an inference model.

The root `amvera.yaml` configures Amvera to run `python3 app.py` on port 80. The server binds to `0.0.0.0` and serves `index.html`, `styles.css`, and `script.js`.

After pushing these files to the branch connected to Amvera, trigger a **new build** (not only a restart). The build should no longer execute `src/genesis/inference/generate_image.py`.

If the Amvera dashboard has a separate manually configured startup command that overrides repository configuration, update it to use the repository's `amvera.yaml` / `app.py` entry point.
