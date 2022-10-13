# Triton-Practices
**Triton Inference Server**

1. **Triton Architecture**
1. **Deploy Triton with container**
- Github source: <https://github.com/triton-inference-server/server>
  - Triton doesn't support response outputs in format FP16.
- Follow source to deploy on container: 

CMD to run triton:

docker run --shm-size 128m -it -p 3000:8000 -p 3001:8001 -p 3002:8002 --gpus device=1 --rm -v ${PWD}/model\_repository:/models  nvcr.io/nvidia/tritonserver:22.09-py3 \![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.001.png)

tritonserver --model-repository=/models  --log-verbose 1 

- Add --shm-size when running on Docker to increase default shared memory from 64MB to 128MB to support 2 models in ensembles tutorials.
3. **Python backend**
- Github source: <https://github.com/triton-inference-server/python_backend.git>
- The goal of Python backend is to let you serve models written in Python by Triton Inference Server without having to write any C++ code.
- Use can use custom python execution environments (follow git source)
- Can use BLS (Business Logic Scripting)
  - Triton's [ensemble](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models) feature supports many use cases where multiple models are composed into a pipeline (or more generally a DAG, directed acyclic graph). However, there are many other use cases that are not supported because as part of the model pipeline they require loops, conditionals (if-then-else), data-dependent control-flow and other custom logic to be intermixed with model execution. We call this combination of custom logic and model executions *Business Logic Scripting (BLS)*.
- Example code of Python backend:

import triton\_python\_backend\_utils as pb\_utils import numpy as np![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.002.png)

import json 

class TritonPythonModel:

`    `def initialize(self, args):

- You must parse model\_config. JSON string is not parsed here![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.003.png)

`        `self.model\_config = model\_config = json.loads(args['model\_config'])

- Get OUTPUT0 configuration

`        `output\_config = pb\_utils.get\_output\_config\_by\_name(model\_config, "output\_classifier")

- Convert Triton types to numpy types

`        `self.output\_dtype = pb\_utils.triton\_string\_to\_numpy(output\_config['data\_type'])

`    `def execute(self, requests):

`        `"""`execute` MUST be implemented in every Python model. `execute`

`        `function receives a list of pb\_utils.InferenceRequest as the only

`        `argument. This function is called when an inference request is made

`        `for this model. Depending on the batching configuration (e.g. Dynamic

`        `Batching) used, `requests` may contain multiple requests. Every

`        `Python model, must create one pb\_utils.InferenceResponse for every

`        `pb\_utils.InferenceRequest in `requests`. If there is an error, you can

`        `set the error argument when creating a pb\_utils.InferenceResponse

`        `Parameters

`        `----------

`        `requests : list

`          `A list of pb\_utils.InferenceRequest

`        `Returns

`        `-------

`        `list

`          `A list of pb\_utils.InferenceResponse. The length of this list must

`          `be the same as `requests`

`        `"""

`        `output\_dtype = self.output\_dtype

`        `responses = []

- Every Python backend must iterate over everyone of the requests
- and create a pb\_utils.InferenceResponse for each of them.

`        `for request in requests:

- Get INPUT0

`            `input\_tensor = pb\_utils.get\_input\_tensor\_by\_name(request, "input\_classifier")             output\_tensor = input\_tensor.as\_numpy()

`            `output\_tensor = np.exp(output\_tensor)/sum(np.exp(output\_tensor))

- Create output tensors. You need pb\_utils.Tensor
- objects to create pb\_utils.InferenceResponse.

`            `output\_tensor = pb\_utils.Tensor("output\_classifier", output\_tensor.astype(output\_dtype))

- Create InferenceResponse. You can set an error here in case
- there was a problem with handling this inference request.
- Below is an example of how you can set errors in inference
- response:

`            `#

- pb\_utils.InferenceResponse(
- output\_tensors=..., TritonError("An error occured"))

`                `output\_tensors=[output\_tensor])![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.004.png)

`            `responses.append(inference\_response)

- You should return a list of pb\_utils.InferenceResponse. Length
- of this list must match the length of `requests` list.

`        `return responses![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.005.png)

![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.006.png)

4. **Model pipelines**
- [**Ensemble:** https://github.com/triton-inference- server/server/blob/main/docs/user_guide/architecture.md#ensemble-models](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)
  - Folder structure: Ensemble\_Model |----1 {empty} |----config.pbtxt |----label.txt
  - Example code

![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.007.jpeg)

name: "ensemble\_model" ![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.008.png)

platform: "ensemble" ![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.003.png)

max\_batch\_size: 1 

input [ 

`  `{ 

`    `name: "IMAGE" 

`    `data\_type: TYPE\_STRING 

`    `dims: [ 1 ] 

`  `} 

] 

output [ 

`  `{ 

`    `name: "CLASSIFICATION" 

`    `data\_type: TYPE\_FP32 

`    `dims: [ 1000 ] 

`  `}, 

`  `{ 

`    `name: "SEGMENTATION" 

`    `data\_type: TYPE\_FP32 

`    `dims: [ 3, 224, 224 ] 

`  `} 

] 

ensemble\_scheduling { 

`  `step [ 

`    `{ 

`      `model\_name: "image\_preprocess\_model"       model\_version: -1 

`      `input\_map { 

`        `key: "RAW\_IMAGE" 

`        `value: "IMAGE" 

`      `} 

`      `output\_map { 

`        `key: "PREPROCESSED\_OUTPUT" 

`        `value: "preprocessed\_image" 

`      `} 

`    `}, 

`    `{ 

`      `model\_name: "classification\_model"       model\_version: -1 

`      `input\_map { 

`        `key: "FORMATTED\_IMAGE" 

`        `value: "preprocessed\_image" 

`      `} 

`      `output\_map { 

`        `key: "CLASSIFICATION\_OUTPUT" 

`        `value: "CLASSIFICATION" 

`      `} 

`    `}, 

`      `model\_name: "segmentation\_model"       model\_version: -1 ![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.009.png)

`      `input\_map { 

`        `key: "FORMATTED\_IMAGE" 

`        `value: "preprocessed\_image" 

`      `} 

`      `output\_map { 

`        `key: "SEGMENTATION\_OUTPUT" 

`        `value: "SEGMENTATION" 

`      `} 

`    `} 

`  `] 

}

- **Business Logic Scripting (BLS)**
  - Example code of BLS:

import triton\_python\_backend\_utils as pb\_utils ![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.010.png)

class TritonPythonModel: 

`  `... 

`    `def execute(self, requests): 

`      `... 

- Create an InferenceRequest object. `model\_name`, 
- `requested\_output\_names`, and `inputs` are the required arguments and 
- must be provided when constructing an InferenceRequest object. Make sure 
- to replace `inputs` argument with a list of `pb\_utils.Tensor` objects. 

`      `inference\_request = pb\_utils.InferenceRequest( 

`          `model\_name='model\_name', 

`          `requested\_output\_names=['REQUESTED\_OUTPUT\_1', 'REQUESTED\_OUTPUT\_2'], 

`          `inputs=[<pb\_utils.Tensor object>]) 

- `pb\_utils.InferenceRequest` supports request\_id, correlation\_id, and model 
- version in addition to the arguments described above. These arguments 
- are optional. An example containing all the arguments: 
- inference\_request = pb\_utils.InferenceRequest(model\_name='model\_name', 
- requested\_output\_names=['REQUESTED\_OUTPUT\_1', 'REQUESTED\_OUTPUT\_2'], 
- inputs=[<list of pb\_utils.Tensor objects>], 
- request\_id="1", correlation\_id=4, model\_version=1, flags=0) 
- Execute the inference\_request and wait for the response 

`      `inference\_response = inference\_request.exec() 

- Check if the inference response has an error 

`      `if inference\_response.has\_error(): 

`          `raise pb\_utils.TritonModelException(inference\_response.error().message())       else: 

- Extract the output tensors from the inference response. 

`          `output1 = pb\_utils.get\_output\_tensor\_by\_name(inference\_response, 'REQUESTED\_OUTPUT\_1') 

- Decide the next steps for model execution based on the received output ![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.011.png)
- tensors. It is possible to use the same output tensors to for the final 
- inference response too.
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

`            `sequence\_batching { ![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.012.png)

`              `state [ 

`                `{ 

`                  `input\_name: "INPUT\_STATE"                   output\_name: "OUTPUT\_STATE"                   data\_type: TYPE\_INT32 

`                  `dims: [ -1 ] 

`                `} ![](Aspose.Words.a37f41ee-aeca-4923-818a-e14573295bb4.011.png)

`              `] 

`            `}        

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

`    `--model-repository ${PWD}/docs/examples/model\_repository \ 

`    `--profile-models "ensemble\_model" --triton-launch-mode=remote \     --output-model-repository-path ${PWD}/demo/analyzer\_conf\_res \     --export-path ${PWD}/demo/analyzer\_output \ 

`    `--triton-http-endpoint localhost:3000 \ 

`    `--triton-grpc-endpoint localhost:3001 \ 

`    `--triton-metrics-url localhost:3002 \ 

`    `--batch-sizes "1,2,3,4"

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
