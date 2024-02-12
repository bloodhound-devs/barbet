from torchapp.testing import TorchAppTestCase
from gambit.apps import Gambit


class TestGambit(TorchAppTestCase):
    app_class = Gambit
