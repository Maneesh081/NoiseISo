#!/usr/bin/env bash
# List the project's task scripts.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cat <<'EOF'
Speech enhancement - task scripts

  ./run.sh          Start the web dashboard at http://localhost:7860
  ./stop.sh         Stop a running dashboard
  ./train.sh        Train the U-Net (40 epochs, early stopping)
  ./eval.sh         Score the checkpoint on the official test set (STOI)
  ./prepare.sh      Resample the real VoiceBank+DEMAND set to 16 kHz + val split
  ./synth.sh        Generate the synthetic fallback corpus (optional)
  ./help.sh         Show this list

Every script accepts -h/--help for details. Run from any directory.
First-time order:  ./prepare.sh  ->  ./train.sh  ->  ./eval.sh  ->  ./run.sh
EOF
