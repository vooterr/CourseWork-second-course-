from model.vlm.model import LlavaVLM

class ModelFactory:

    REGISTRY = {
        "llava": LlavaVLM,
        # "qwen": QwenVLM,
        # "internvl": InternVLVLM
    }

    @staticmethod
    def create(name: str, **kwargs):
        if name not in ModelFactory.REGISTRY:
            raise ValueError(f"Unknown model: {name}")

        return ModelFactory.REGISTRY[name](**kwargs)
