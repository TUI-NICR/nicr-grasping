# set loglevel for default logger to ERROR to silence anoying warning from autolab_core
# about not finding ros packages
import logging
logging.getLogger().setLevel(logging.ERROR)
