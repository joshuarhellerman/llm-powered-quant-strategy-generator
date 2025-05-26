# Save as tiktoken_patch_enhanced.py
"""
Enhanced patch to prevent tiktoken-related segfaults.
This must be imported before any other imports.
"""
import sys
import types
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tiktoken_patch")


# Create a complete fake tiktoken module with all necessary attributes
class FakeEncoding:
    def encode(self, text):
        # Estimate tokens (4 chars per token)
        if not text:
            return []
        return [0] * (len(text) // 4 or 1)

    def decode(self, tokens):
        return "".join(["x"] * len(tokens))

    def encode_ordinary(self, text):
        return self.encode(text)

    def decode_single_token_bytes(self, token):
        return b"x"


class FakeTiktoken(types.ModuleType):
    def __init__(self):
        super().__init__("tiktoken")
        self.__version__ = "0.0.0"
        self.__file__ = __file__
        self.__spec__ = types.SimpleNamespace(
            name="tiktoken",
            origin="tiktoken_patch",
            submodule_search_locations=[]
        )
        self.__path__ = []
        self.__package__ = "tiktoken"

    def get_encoding(self, encoding_name):
        return FakeEncoding()

    def list_encoding_names(self):
        return ["cl100k_base", "p50k_base", "r50k_base"]

    # Include any other functions that might be called
    def encoding_for_model(self, model_name):
        return FakeEncoding()


# Check if tiktoken is already imported
if "tiktoken" in sys.modules:
    logger.warning("Tiktoken already imported, replacing with fake module")
    # Replace the existing module with our fake one
    sys.modules["tiktoken"] = FakeTiktoken()
else:
    # Install our fake module
    sys.modules["tiktoken"] = FakeTiktoken()
    logger.info("Installed enhanced fake tiktoken module to prevent segfaults")

# Ensure garbage collection is enabled
try:
    import gc

    gc.collect()
    logger.info("Forced garbage collection")
except:
    pass

# Disable CPU parallelism which can cause issues
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger.info("Set environment variables to enhance stability")