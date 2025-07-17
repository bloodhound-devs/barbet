.. image:: https://github.com/bloodhound-devs/barbet/blob/main/docs/images/barbet-banner.jpg?raw=true

.. start-badges

|pypi badge| |testing badge| |coverage badge| |docs badge| |black badge| |torchapp badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/barbet?color=blue
   :alt: PyPI - Version
   :target: https://pypi.org/project/barbet/

.. |testing badge| image:: https://github.com/bloodhound-devs/barbet/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/bloodhound-devs/barbet/actions

.. |docs badge| image:: https://github.com/bloodhound-devs/barbet/actions/workflows/docs.yml/badge.svg
    :target: https://bloodhound-devs.github.io/barbet
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/09aad5114164b54daabe1f5efd02a009/raw/coverage-badge.json
    :target: https://bloodhound-devs.github.io/barbet/coverage/

.. |torchapp badge| image:: https://img.shields.io/badge/torch-app-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/
    
.. end-badges

.. start-quickstart

Installation
==================================

Install using pip:

.. code-block:: bash

    pip install git+https://github.com/bloodhound-devs/barbet.git


Usage
==================================

See the options for making inferences with the command:

.. code-block:: bash

    barbet --help


Run
==================================

.. code-block:: bash

    barbet --input GCA_000006945.2.fna --output-dir outputs

Or using the large model:

.. code-block:: bash

    barbet --input GCA_000006945.2.fna --output-dir outputs-large --large


Training
==================================

You can train the model on releases from GTDB or your own custom dataset.
See the instructions in the documentation for `preprocessing <https://bloodhound-devs.github.io/barbet/preprocessing.html>`_ and `training <https://bloodhound-devs.github.io/barbet/training.html>`_.

.. end-quickstart


Credits
==================================

.. start-credits

`Robert Turnbull <https://robturnbull.com>`_, Mar Quiroga, Gabriele Marini, Torsten Seemann, Wytamma Wirth

For more information contact: <wytamma.wirth@unimelb.edu.au>

Created using torchapp (https://github.com/rbturnbull/torchapp).

.. end-credits

