import base64
import io
import logging

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    # AutoProcessor may return a VL processor for text-only models (e.g.
    # Qwen3.5-27B gets Qwen3VLProcessor in transformers>=5.x).  Detect this
    # by checking whether the model actually has a vision component.
    if proc is not None:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=kwargs.get("trust_remote_code", False))
            model_type = getattr(config, "model_type", "")
            # Only keep processor for genuine vision/multimodal models
            if model_type and "vl" not in model_type.lower() and "vision" not in model_type.lower():
                logger.info(f"Discarding processor ({type(proc).__name__}) for text-only model_type={model_type}")
                proc = None
        except Exception:
            pass

    return proc


def process_vision_info(prompt, processor):
    # temporary solution, will write image utils for slime later
    from qwen_vl_utils import process_vision_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
