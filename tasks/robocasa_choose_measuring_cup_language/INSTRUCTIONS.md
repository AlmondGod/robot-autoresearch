# robocasa_choose_measuring_cup_language Instructions

Write outputs under
`runs/autorobobench/robocasa_choose_measuring_cup_language/<run>/`. Do not edit
eval files or split files for scored runs.

## Task

- Optimize one language-conditioned policy over four variants:
  `ChooseMeasuringCupLeftLarger`, `ChooseMeasuringCupLeftSmaller`,
  `ChooseMeasuringCupRightLarger`, `ChooseMeasuringCupRightSmaller`.
- Metric: language-conditioned success. Also report wrong-language success and
  conditioning gap.
- Do not collapse variants into one unlabeled task.

## Train

```bash
python3 tasks/robocasa_choose_measuring_cup_language/train.py \
  --manifest data/autorobobench/robocasa_choose_measuring_cup_language_manifest.json \
  --split data/autorobobench/robocasa_choose_measuring_cup_language_splits.json \
  --out-dir runs/autorobobench/robocasa_choose_measuring_cup_language/<run> \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_choose_measuring_cup_language/eval.py \
  --checkpoint runs/autorobobench/robocasa_choose_measuring_cup_language/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_choose_measuring_cup_language/<run>/eval.json \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_choose_measuring_cup_language/visualize.py \
  --run-dir runs/autorobobench/robocasa_choose_measuring_cup_language/<run>
```
