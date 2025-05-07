"""
Trading strategy templates package
"""

from .momentum import TEMPLATE as MOMENTUM_TEMPLATE
from .mean_reversion import TEMPLATE as MEAN_REVERSION_TEMPLATE
from .reinforcement_learning import TEMPLATE as REINFORCEMENT_LEARNING_TEMPLATE
from .transformer import TEMPLATE as TRANSFORMER_TEMPLATE

# Map of template names to actual templates
TEMPLATES = {
    "momentum": MOMENTUM_TEMPLATE,
    "mean_reversion": MEAN_REVERSION_TEMPLATE,
    "reinforcement_learning": REINFORCEMENT_LEARNING_TEMPLATE,
    "transformer": TRANSFORMER_TEMPLATE
}

def get_template(strategy_type):
    """Get the template for a specific strategy type"""
    return TEMPLATES.get(strategy_type, MOMENTUM_TEMPLATE)  # Default to momentum