# White Hmong VITS training and fine-tuning

## Choose how to initialize

- **Train from scratch.** Start from a VITS config with a serialized, ordered RPA character map and use a new output directory. Initialize model, optimizer and scheduler from scratch. Persist the config, tokenizer hash, corpus split and package versions alongside checkpoints.
- **Start a new fine-tune.** Load the verified source config and only the checkpoint's model tensors. Match the tokenizer by exact order and boundary flags before loading any weights. Keep the source architecture and audio settings. Start both optimizers, the schedulers, the epoch index and the global step fresh. Use a new run directory.
- **Resume an interrupted run.** Use a checkpoint belonging to that same run, with its original config and character order. A true resume also restores optimizer, scheduler and counters. Coqui's periodic checkpoint can be within an epoch and its stock resume can replay that epoch. The Experiment B launcher refuses to reuse an output directory. Use a separately validated same-run resume procedure if interruption recovery is required.

The first Experiment B attempt initialized from step 35,461 with an alphabetized map in place of the recovered map. It also reused the source checkpoint's counters and stopped after 12 additional epochs. Initial learning rates were not applied to both optimizers. Its validation metrics and generated audio are superseded.

## Repeat Experiment B on Radiance

Run these commands from the RPA checkout in PowerShell. Use the pinned environment
`.venv-vits` with `coqui-tts==0.28.0` and `coqui-tts-trainer==0.4.0`.

```powershell
$env:PYTHONPATH = 'src;scripts'
$ffmpeg = Get-ChildItem data/artifacts/ffmpeg-shared/extracted81 -Recurse -Filter ffmpeg.exe
if (@($ffmpeg).Count -ne 1) { throw 'Expected one shared FFmpeg 8.1 installation' }
$env:PATH = "$($ffmpeg.DirectoryName);$env:PATH"
$python = '.venv-vits/Scripts/python.exe'
$run = 'data/runs/expb-cleaned-v2'
& $python -u scripts/vits_expb_train.py prepare --run $run --epochs 100 --batch-size 16 --lr 0.0001
& $python -u scripts/vits_expb_train.py train --run $run --check
& $python -u scripts/vits_expb_train.py train --run $run
```

The default bundle is `data/artifacts/rpa-vits-step35461-listener-verified`. Keep its
`best_model_35461.pth`, `config.json` and manifest together. The launcher authenticates both
files, validates the ordered vocabulary digest, and checks the instantiated tokenizer
and model tensor shapes before training. Preflight decodes one WAV from each split using
Coqui’s actual Windows FFmpeg/TorchCodec stack. A legacy checkpoint without a serialized map
can inherit the order only from this authenticated recovery config. A same-size map with
different token IDs fails validation.

The cleaned input is `data/derived/mwwhdv-nt/cleaned-v001`. The run records its manifest
hash, split counts, batch size and both optimizer rates. Preparation applies the
existing RPA text projection to every transcript before writing Coqui metadata and
hashes both split files. Preflight validates every transcript against the ordered
vocabulary so Coqui cannot silently drop uppercase or unsupported characters. Keep the dataset and run
configuration stable for the full job. Place logs in the new run directory and launch
in a persistent terminal on Windows. Check `/api/ps` on Ollama before starting and leave
the DSH-TUI model unloaded during training.

## Verify a launch and qualify a candidate

1. Read `training-state.json` for the process ID, trainer output directory, requested epoch count and completed epoch count. A fresh start must show epoch index 0 and step 0, both optimizers empty, and both learning rates at 0.0001. Coqui prints the zero-based range as `EPOCH: 0/99` for the first epoch.
2. Verify live training steps and a fresh TensorBoard event file. Compare the first trainer preview with the listener-verified recovery checkpoint for the same phrase before treating the run as healthy.
3. Verify all 100 new epochs using the completed epoch count, the TensorBoard epoch events and the final checkpoint. The best validation checkpoint may be earlier than the final checkpoint.
4. Before adopting any resulting model, generate matched audio from the baseline and candidate using their verified configs. Have a White Hmong listener check tones, enunciation and pacing. Validation loss alone cannot establish speech quality.
