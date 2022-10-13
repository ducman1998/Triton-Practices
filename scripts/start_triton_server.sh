cd docs/examples
# export CUDA_VISIBLE_DEVICES=1
docker run --shm-size 128m -it -p 3000:8000 -p 3001:8001 -p 3002:8002 --gpus device=1 --rm -v ${PWD}/model_repository:/models  nvcr.io/nvidia/tritonserver:22.09-py3 \
tritonserver --model-repository=/models  --log-verbose 1 
# docker run -it -p 3000:8000 -p 3001:8001 -p 3002:8002 --gpus device=1 --rm -v ${PWD}/model_repository:/models  nvcr.io/nvidia/tritonserver:22.09-py3 
cd ../..
