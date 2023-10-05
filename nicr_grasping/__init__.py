import logging
import importlib.util

from .version import __version__

logging.basicConfig(format='[%(name)s: %(asctime)s] (%(levelname)s) %(message)s', level=logging.INFO)
logger = logging.getLogger('nicr_grasping')

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
