:layout: landing
:description: Shibuya is a modern, responsive, customizable theme for Sphinx.

PyLossless :octicon:`pulse`
===========================

.. rst-class:: lead

    EEG Processing Pipeline that is non-destructive, automated, and built on Python.

.. container:: buttons

    `Docs <install.html>`_
    `GitHub <https://github.com/lina-usc/pylossless>`_


.. grid:: 1 1 2 3
    :gutter: 2
    :padding: 0
    :class-row: surface

    .. grid-item-card:: :octicon:`zap` Automated

        Fast, Open-source, and built on python.

    .. grid-item-card:: :octicon:`pin` Non-destructive

        Keeps EEG continuous, noting bad channels, times, and independent components
        so you can reject them and epoch your data however and whenever you want to.

    .. grid-item-card:: :octicon:`telescope-fill` Streamlined Review

         Web dashboard built with helps you review the output and make
         informed decisions about your data.

.. image:: https://raw.githubusercontent.com/scott-huberty/wip_pipeline-figures/main/dashboard.png
   :alt: pylossless-qc-dashboard-screenshot
   :align: center


.. toctree::
   :maxdepth: 1
   :hidden:

   install
   auto_examples/index.rst
   API/API_index
   contributing
   Paper <https://www.biorxiv.org/content/10.1101/2024.01.12.575323v1>