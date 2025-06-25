from .apps import Barbet
import barbet
import sys

sys.modules['bloodhound'] = barbet
