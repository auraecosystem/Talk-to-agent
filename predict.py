from kubu_hai.artifacts import ModelArtifactManager
from kubu_hai.engine import HybridEngine, InferenceRequest
from kubu_hai.registry import ModelRegistry

# Define this architecture in trusted, reviewed application code.
def build_model():
    import torch
    return torch.nn.Linear(4, 2)

artifacts = ModelArtifactManager("./model-artifacts")
adapter = artifacts.load_pytorch(
    "linear/model.json", model_factory=build_model, device="auto"
)
registry = ModelRegistry()
registry.register(adapter)
engine = HybridEngine(registry)
result = engine.infer(InferenceRequest(
    task="tabular.classification", inputs={"input": [[0.1, 0.2, 0.3, 0.4]]},
    preferred_backend=adapter.backend,
))
print(result.model_id, result.backend, result.output)
