from .kg_gen import KGGen 
from .models import Graph

# Optional SAP AI Core support
try:
    from .aicore_provider import AICoreLM, create_aicore_lm
    __all__ = ["KGGen", "Graph", "AICoreLM", "create_aicore_lm"]
except ImportError:
    __all__ = ["KGGen", "Graph"]