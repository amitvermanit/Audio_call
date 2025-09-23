#!/bin/bash

# === CONFIG ===
CONDA_PATH="/home/test/miniconda3/etc/profile.d/conda.sh"
PROFILE_DIR="/tmp/.chrome_script_profile"

# === FRONTEND ===
gnome-terminal -- bash -ic "cd /home/cb-translator/frontend && echo '[INFO] Starting Frontend...' && npm run dev ; exec bash"

# === BACKEND ===
gnome-terminal -- bash -ic "cd /home/cb-translator/backend && echo '[INFO] Starting Backend...' && npm run start:dev ; exec bash"


# === WHISPER-S2T ===
gnome-terminal -- bash -c "
    source $CONDA_PATH
    conda deactivate || true
    conda activate S2T_ENV
    cd ./ml2 || exit 1
    echo '[INFO] Running Whisper main...'
    python _main.py
    exec bash
"
# === SEAMLESS-T2S ===
gnome-terminal -- bash -c "
    source $CONDA_PATH
    conda deactivate || true
    conda activate T2S_ENV
    cd ./ml2/pipelines/seamless_t2s || exit 1
    echo '[INFO] Running Seamless-worker ...'
    python seamless_t2S_worker.py
    exec bash
"
# === VOICE-CLONE ===
gnome-terminal -- bash -c "
    source $CONDA_PATH
    conda deactivate || true
    conda activate VC_ENV
    cd ./ml2/pipelines/voice_clone || exit 1
    echo '[INFO] Running voice-clone-worker ...'
    python voice_clone_worker.py
    exec bash
"

# === OPEN CHROMIUM BROWSERS ===
chromium --user-data-dir="$PROFILE_DIR" "http://localhost:5173/" --new-window &
chromium --user-data-dir="$PROFILE_DIR" --incognito "http://localhost:5173/" &

