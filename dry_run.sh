tira-cli code-submission \
    --path . \
    --command '/app/build/main.exe --inputFolder $inputDataset --outputFolder $outputDir --strategy blocked-inverted --kTop 30 --task task3 --dataset fiqa-dev  --threads -1' \
    --task sisap-2026 \
    --dataset task-3-spot-check-20260529-training \
    --dry-run
    # --verbose