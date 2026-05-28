for task in 1 2 3; do
    echo Running Task $task
    docker run \
        -it \
        --cpus=4 \
        --memory=16g \
        --memory-swap=16g \
        --memory-swappiness 0 \
        --volume $(pwd)/search.py:/app/search.py:ro \
        --volume $(pwd)/data:/app/data:ro \
        --volume $(pwd)/results:/app/results:rw \
        -t sisap-baseline --input data/task-$task-spot-check/*.h5 --task-description data/task-$task-spot-check/config.json --output results/task-$task-spot-check/
done
