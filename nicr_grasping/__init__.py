import logging

logger: logging.Logger = logging.getLogger('nicr_grasping')
logging.basicConfig(format='[%(name)s: %(asctime)s] (%(levelname)s) %(message)s', level=logging.WARNING, force=True)


# add logging filter to silence autolab_core warnings about not finding ros packages
class AutolabCoreFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return 'autolab_core' not in record.pathname


logging.getLogger().addFilter(AutolabCoreFilter())

import importlib.util

from .version import __version__


############################################################################
# MODIFY HERE FOR NEW PAKAGE INTEGRATIONS
############################################################################

# GraspNet
graspnet_spec = importlib.util.find_spec('graspnetAPI')
GRASPNET_INSTALLED = graspnet_spec is not None

# GGCNN
grasp_detection_spec = importlib.util.find_spec('grasp_detection')
GRASP_DETECTION_INSTALLED = grasp_detection_spec is not None

from .utils.paths import graspnet_dataset_path
