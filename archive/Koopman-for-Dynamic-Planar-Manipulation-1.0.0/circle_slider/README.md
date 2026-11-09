# Koopman zero-control fine-tuning patch

This replaces the previous hard constraint `||A||_2 <= 1` with a data-anchored
approach that preserves the pretrained Koopman model as much as practical.

## What changes

- Generates physically simulated trajectories with the pusher control fixed at
  exactly zero, using the same `Pusher.dynamics_bodyu` simulator and physical
  parameters as `GenerateData_main.py`.
- Each zero-input trajectory contains 31 samples, i.e. 30 transitions, with a
  sample spacing of 0.125 s to match the spacing produced by the original
  `traj_len=5`, `dt=0.1` generator (`linspace(0, 5*0.1, 5)`).
- Fine-tunes only `model.A.weight` from the existing `model.pt` checkpoint.
  The encoder and `B_bilinear` network are frozen and remain exactly unchanged.
- Uses both the original controlled trajectories and long zero-control
  trajectories in the original multi-step prediction-loss structure.
- Does not clip eigenvalues or force `||A||_2 <= 1`.

This is not a mathematical guarantee that every possible zero-input state is
bounded. It instead teaches the existing model the actual simulator's
zero-input behavior over the same 30-step horizon used by ANR. Held-out
zero-control validation loss and the change in A are printed each epoch.

## Files

- `GenerateData_subset.py`: original data generator with an added
  `control_mode="zero"` option. Default behavior remains random control.
- `GenerateData_zero_control.py`: generates 2,000 zero-control trajectories.
- `helpers/learn_modules_zeroaug.py`: warm-start fine-tuning; only A is trainable.
- `training_zeroaug.py`: loads both datasets and saves `model_zeroaug.pt`.

## Run

From the Koopman model directory (the folder containing `model.pt`,
`planar_pusher.py`, and the original `GenerateData_subset.py`):

1. Back up the original `GenerateData_subset.py`.
2. Replace it with the included version (or merge only its `control_mode`
   argument and branch).
3. Put `GenerateData_zero_control.py` in that directory and run:

   ```bash
   python GenerateData_zero_control.py
   ```

4. Put `learn_modules_zeroaug.py` in `helpers/` and run:

   ```bash
   python training_zeroaug.py
   ```

5. The output is `model_zeroaug.pt`. Keep `model.pt` untouched. Point the ANR
   script to `model_zeroaug.pt` only after reviewing validation loss and the
   relative change in A.

The generation step uses the existing simulator. If the simulator does not
complete all requested integrations, inspect its output before training.
