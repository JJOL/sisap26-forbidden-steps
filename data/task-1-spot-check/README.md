---
configs:
- config_name: inputs
  data_files:
  - split: train
    path: ["benchmark-dev-gooaq-small.h5", "config.json"]
- config_name: truths
  data_files:
  - split: train
    path: ["benchmark-dev-gooaq-small.h5", "config.json"]

tira_configs:
  resolve_inputs_to: "."
  resolve_truths_to: "."
  baseline:
    link: https://github.com/maumueller/sisap26-small-example/tree/main/
    command: python /app/search.py --input $inputDataset/*.h5 --task-description $inputDataset/*.json  --output $outputDir/results.h5
    format:
      name: ["sisap-predictions"]
  input_format:
    name: "arbitrary"
  truth_format:
    name: "arbitrary"
  evaluator:
    image: mam10eks/sisap-protype:eval-0.0.2
    command: /eval.py $inputDataset/*.json $inputDataset/*.h5 $inputRun  $outputDir/evaluation.prototext
---

# SISAP 2026: gooaq

(attention, the baseline is private, so the following requires additional cloning the repo to `~/.tira/git-repositories/sisap26-small-example)

tira-cli dataset-submission --path task-1-spot-check --task sisap-2026 --split train --dry-run

If everything work, this should yield:

<img width="2939" height="368" alt="Screenshot_20260507_161741" src="https://github.com/user-attachments/assets/ebf12131-8f71-4764-b917-f2dd6c53ac5a" />

To upload the data to tira, remove the `--dry-run` flag.
