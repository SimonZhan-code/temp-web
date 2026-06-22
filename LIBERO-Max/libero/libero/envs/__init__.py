from .bddl_base_domain import TASK_MAPPING
from .base_object import OBJECTS_DICT
from .problems import *
from .robots import *
from .arenas import *
from .env_wrapper import OffScreenRenderEnv, SegmentationRenderEnv
from .venv import SubprocVectorEnv, DummyVectorEnv

# Register SafeLIBERO object aliases (must come after OBJECTS_DICT is populated)
from .safety import _register_safelibero_object_aliases
_register_safelibero_object_aliases()
