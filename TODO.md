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
    - Try IVF [X]
    - Try Terms Ordered (Cached scores) [X]
    - Try Product Quantization [X]
        * Results: NO -- NEEDS TO CONVERT TO DENSE
    - Try HNSW [X]
        * Results: NO: ENORMOUSE INDEX BUILD TIME, NO GAIN ON SPEED WHEN MAINTAINING RECALL > 0.90
    - Learned Index? Tabu-Search lowering 10% to 10%
        - Weak-AND:
        * Results: Access Pattern based on many pointer increments and skips has huge latency and overhead compared to IVF
    - Try Hyprid MaxScores with Inverted Files or Block-Max MaxScore: https://medium.com/@shivamprasad484/maxscore-algorithm-making-search-lightning-fast-19ce92bf1d41, https://www.elastic.co/search-labs/blog/more-skipping-with-bm-maxscore
    - Parallelize using OpenMP


CRITERION:
- Best Top-3 Algorithms