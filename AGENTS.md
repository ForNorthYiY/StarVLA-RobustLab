# StarVLA-RobustLab workspace rules

These rules apply to the entire repository and to project operations on the `ict80` server.

## Remote server layout

- SSH alias: `ict80`
- Remote user/home: `yiyang`, `/data/yiyang`
- Project source code: `/data/yiyang/git/`
- Personal datasets: `/data/dataset/yiyang/` only
- Base/pretrained models: `/data/yiyang/models/`
- Personal training checkpoints: `/data/yiyang/checkpoints/`
- Hugging Face cache: `/data/yiyang/cache/huggingface/`
- Conda installation: `/data/yiyang/miniconda3/`

## Storage rules

- Never store datasets under `/data/yiyang`, `$HOME`, a Git repository, or another user's directory.
- All dataset downloads, conversions, and extracted files must stay below `/data/dataset/yiyang/`.
- All model weights and checkpoints must stay in directories owned by `yiyang`; use the model/checkpoint paths above.
- Before downloading data, models, or checkpoints, check free space and estimate the required storage.
- Do not commit datasets, model weights, checkpoints, caches, run outputs, or secrets to Git.

## Git and backup rules

- Keep project code in a Git repository and configure a remote backup before substantial development.
- Make small, periodic commits, with one logical change per commit.
- Check `git status` before and after edits; preserve unrelated user changes.
- Push completed, tested milestones to the configured remote in a timely manner.
- Never put passwords, private SSH keys, API tokens, or machine-specific secrets in commits.

## Experiment discipline

- Run preflight and smoke tests before expensive training or evaluation.
- Do not start long or multi-GPU jobs without explicit authorization.
- Record the project commit, StarVLA commit, resolved config, environment, seed, checkpoint, and command for every run.
- Never fabricate experiment results; use `TBD` until a real run completes.
