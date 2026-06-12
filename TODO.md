<!-- To upload submissions: https://www.tira.io/submit/sisap-2026/user/orbidden-steps/code-submission


# TO TEST:
tira-cli code-submission \
--path . \
--command '/app/own_main.exe $inputDataset inverted $outputDir/cpp_own.h5 fiqa-small task3' \
--task sisap-2026 \
--dataset task-3-spot-check-20260529-training \
--dry-run --verbose -->



- Paper
    - 
- Solution
    - Try Brute Force [X]
    - Try Sparse Efficient Brute Force [X]
    - Try IVF 
    - Try Terms Ordered (Cached scores)
    - Try Product Quantization
    - Try HNSW
    - Learned Index? Tabu-Search lowering 10% to 10%
        Or for the best scores, keep the percentil where their second or third best term lies in

    - Then try to parallelize Top-3 and choose best option


CRITERION:
- Best Top-3 Algorithms
- 