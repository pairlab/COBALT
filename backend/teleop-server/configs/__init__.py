from configs.config import CCRConfig
from configs.base_client_config import BaseClientConfig
from configs.base_config_robosuite import BaseServerConfigRobosuite
from configs.base_config_yam import BaseServerConfigYAM

_ISAAC_IMPORT_ERROR = None

try:
    from configs.base_config_isaac import BaseServerConfigIsaac
except Exception as exc:  # pragma: no cover - optional dependency path
    BaseServerConfigIsaac = None
    _ISAAC_IMPORT_ERROR = exc


def get_isaac_import_error():
    """Return the import error raised while loading Isaac config, if any."""
    return _ISAAC_IMPORT_ERROR
