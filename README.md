# Triton-Practices
**Triton Inference Server**

1. **Triton Architecture**
2. **Deploy Triton with container**
3. **Python backend**
4. **Model pipelines**
- [**Ensemble:** https://github.com/triton-inference- server/server/blob/main/docs/user_guide/architecture.md#ensemble-models](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)
- **Business Logic Scripting (BLS)**
5. **Add custom operations to Triton**
- [Follow git: https://github.com/triton-inference- server/server/blob/main/docs/user_guide/custom_operations.md](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/custom_operations.md)
6. **Models And Schedulers**
- [Git resource: https://github.com/triton-inference- server/server/blob/main/docs/user_guide/architecture.md#models-and- schedulers](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#models-and-schedulers)
- Stateless models
  - A *stateless* model does not maintain state between inference requests. Each inference performed on a stateless model is independent of all other inferences using that model.
  - Dynamic batcher can be used.
- Stateful models
  - A *stateful* model does maintain state between inference requests. The model is expecting multiple inference requests that together form a sequence of inferences that must be routed to the same model instance so that the state being maintained by the model is correctly updated.
  - Sequence batcher can be used.
- Ensemble models
  - Implicit State Management
    - Implicit state management allows a stateful model to store its state inside Triton. When using implicit state, the stateful model does not need to store the state required for inference inside the model.
    - Implicit state management requires backend support. Currently, only [onnxruntime_backend](https://github.com/triton-inference-server/onnxruntime_backend) and [tensorrt_backend](https://github.com/triton-inference-server/tensorrt_backend) support implicit state.
    - Below is a portion of the model configuration that indicates the model is using implicit state.
- Scheduling Strategies
  - Direct
  - Oldest
7. **Optimize performance using Model Analyzer Tool**
- **Install**
  - Using Docker (inside container start, will init triton server and benchmark models at server)
  - Install with pip and profiling an already running Triton Inference Server.
    - pip install triton-model-analyzer
- **Start profiling remote server**
  - You should also make sure that same GPUs are available to the Inference Server and Model Analyzer and they are on the same machine. Model Analyzer does not currently support profiling remote GPUs.

model-analyzer profile \ ![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.013.png)
8. **Manage loading & unloading models**
- To reduce service downtime, Triton loads new models in the background while continuing to serve inferences on existing models. Based on use case and performance requirements, the optimal amount of resources dedicated to loading models may differ. Triton exposes a --model-load-thread-count option to configure the number of threads dedicated to loading models, which defaults to twice the number of CPU cores (2\*num\_cpus) visible to the server.
- Three modes:
  - NONE (Model auto try to load all models at startup, cannot load/unload using API)
  - EXPLICIT (you can load/unload using model control API)

![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.014.png) POOL (auto reload changes in model repo in period, not recommend in production because lack of sync)

9. **Customize Triton Inference Server container**
- [Git source: https://github.com/triton-inference- server/server/blob/main/docs/customization_guide/compose.md](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md)
  - You can remove some backend those don't need.
10. **Create custom backends in C++/Python**
10. **Triton repository agent**
- [Git source: https://github.com/triton-inference- server/server/blob/main/docs/customization_guide/repository_agents.md](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/repository_agents.md)
- With repo agent, you can modify process of load model from repo
  - Triton follows these steps when loading a model:
    - Load the model's configuration file (config.pbtxt) and extract the *ModelRepositoryAgents* settings. Even if a repository agent modifies the config.pbtxt file, the repository agent settings from the initial config.pbtxt file are used for the entire loading process.
    - For each repository agent specified:
      - Initialize the corresponding repository agent, loading the shared library if necessary. Model loading fails if the shared library is not available or if initialization fails.
      - Invoke the repository agent's *TRITONREPOAGENT\_ModelAction* function with action TRITONREPOAGENT\_ACTION\_LOAD. As input the agent can access the model's repository as either a cloud storage location or a local filesystem location.
      - The repository agent can return *success* to indicate that no changes where made to the repository, can return *failure* to indicate that the model load should fail, or can create a new repository for the model (for example, by decrypting the input repository) and return *success* to indicate that the new repository should be used.
      - If the agent returns *success* Triton continues to the next agent. If the agent returns *failure*, Triton skips invocation of any additional agents.
    - If all agents returned *success*, Triton attempts to load the model using the final model repository.
    - For each repository agent that was invoked with TRITONREPOAGENT\_ACTION\_LOAD, in reverse order:
    - Triton invokes the repository agent's *TRITONREPOAGENT\_ModelAction*  function with action TRITONREPOAGENT\_ACTION\_LOAD\_COMPLETE if the model loaded successfully or TRITONREPOAGENT\_ACTION\_LOAD\_FAIL if the model failed to load.
- Triton follows these steps when unloading a model:
  - Triton uses the repository agent settings from the initial config.pbtxt file, even if during loading one or more agents modified its contents.
  - For each repository agent that was invoked with TRITONREPOAGENT\_ACTION\_LOAD, in the same order:
    - Triton invokes the repository agent's *TRITONREPOAGENT\_ModelAction*  function with action TRITONREPOAGENT\_ACTION\_UNLOAD.
  - Triton unloads the model.
  - For each repository agent that was invoked with TRITONREPOAGENT\_ACTION\_UNLOAD, in reverse order:
    - Triton invokes the repository agent's *TRITONREPOAGENT\_ModelAction*  function with action TRITONREPOAGENT\_ACTION\_UNLOAD\_COMPLETE.
12. **Deploy Triton on Jetson nano**
12. **GRPC protocol in Triton**
