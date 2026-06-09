echo Running Task 3
mkdir -p results/task-3-spot-check
docker run \
    --rm \
    --user "$(id -u):$(id -g)" \
    --cpus=4 \
    --memory=16g \
    --memory-swap=16g \
    --memory-swappiness 0 \
    --volume $(pwd)/data:/app/data:ro \
    --volume $(pwd)/results:/app/results:rw \
    sisap-own ./own_main.exe data/task-3-spot-check/benchmark-dev-fiqa-small.h5 inverted results/task-3-spot-check/cpp_own.h5 fiqa-small task3