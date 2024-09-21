from torchapp.testing import TorchAppTestCase
from gambit._old_apps_fastai import Gambit


class TestGambit(TorchAppTestCase):
    app_class = Gambit
