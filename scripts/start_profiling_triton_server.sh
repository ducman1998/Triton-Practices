rm -rf demo/analyzer_conf_res
model-analyzer profile \
    --model-repository ${PWD}/docs/examples/model_repository \
    --profile-models "ensemble_model" --triton-launch-mode=remote \
    --output-model-repository-path ${PWD}/demo/analyzer_conf_res \
    --export-path ${PWD}/demo/analyzer_output \
    --triton-http-endpoint localhost:3000 \
    --triton-grpc-endpoint localhost:3001 \
    --triton-metrics-url localhost:3002 \
    --batch-sizes "1,2,3,4" \
    --use-local-gpu-monitor \
    --latency-budget 250
    