for task in 1 2 3; do
    echo Running Task $task
    mkdir -p results/task-$task-spot-check
    docker run \
        --rm \
        --user "$(id -u):$(id -g)" \
        --cpus=4 \
        --memory=16g \
        --memory-swap=16g \
        --memory-swappiness 0 \
        --volume $(pwd)/search.py:/app/search.py:ro \
        --volume $(pwd)/data:/app/data:ro \
        --volume $(pwd)/results:/app/results:rw \
        sisap-baseline python search.py --input data/task-$task-spot-check/*.h5 --task-description data/task-$task-spot-check/config.json --output results/task-$task-spot-check/
done
