<p align="center"><img src="docs/images/torchlanguage_complete.png" /></p>

--------------------------------------------------------------------------------
TorchLanguage is the equivalent of TorchVision for Natural Language Processing. It gives you access to text transformers (tokens, index, n-grams, etc) and data sets.

.. image:: https://www.travis-ci.org/nschaetti/TorchLanguage.svg?branch=master
    :target: https://www.travis-ci.org/nschaetti/TorchLanguage

.. image:: https://codecov.io/nschaetti/torchlanguage/text/branch/master/graph/badge.svg
    :target: https://codecov.io/nschaetti/torchlanguage/text

.. image:: https://readthedocs.org/projects/torchlanguage/badge/?version=latest
    :target: http://torchlanguage.readthedocs.io/en/latest/?badge=latest

torchlanguage
+++++++++

This repository consists of:

* `torchlanguage.datasets <#datasets>`_: Pre-built datasets for common NLP tasks
* `torchlanguage.model <#models>` _: Generic pretrained models for common NLP tasks
* `torchlanguage.transforms <#transforms>` _: Common transformation for text
* `torchlanguage.utils <#tools>` _: Tools, functions and measures for NLP

Installation
============

Make sure you have Python 2.7 or 3.5+ and PyTorch 0.2.0 or newer. You can then install torchlanguage using pip::

    pip install torchlanguage

Optional requirements
---------------------

If you want to use English tokenizer from `SpaCy <http://spacy.io/>`_, you need to install SpaCy and download its English model::

    pip install spacy
    python -m spacy download en

Alternatively, you might want to use Moses tokenizer from `NLTK <http://nltk.org/>`_. You have to install NLTK and download the data needed::

    pip install nltk
    python -m nltk.downloader perluniprops nonbreaking_prefixes

Data
====

The data module provides the following:

* Ability to download and load a corpus from a directory. The file must be name Class_Title.txt:

  .. code-block:: python

      >>> dataset = torchlanguage.datasets.FileDirectory(
      ...    root='./data',
      ...    download=True,
      ...    download_url="http://urltozip/file.zip",
      ...    transform=transformer
      ...    )

* Wrapper for dataset splits (train, validation) and cross-validation:

  .. code-block:: python

      >>> TEXT = data.Field()
      >>> LABELS = data.Field()
      >>> cross_val_dataset = {'train': torchlanguage.utils.CrossValidation(dataset, k=k),
      ...     'test': torchlanguage.utils.CrossValidation(dataset, k=k, train=False)}
      >>> for k in range(k):
      >>>     for data in cross_val_dataset['train']:
      >>>         inputs, label = data
      >>>     # end for
      >>>     for data in cross_val_dataset['test']:
      >>>         inputs, label = data
      >>>     # end for
      >>>     cross_val_dataset['train'].next_fold()
      >>>     cross_val_dataset['test'].next_fold()
      >>> # end for

Datasets
========

The datasets module currently contains:

* FileDirectory: Load a corpus from a directory
* ReutersC50Dataset: The Reuters C50 dataset for authorship attribution
* SFGram: A set of science-fiction magazine with five authors.

Others are planned or a work in progress:

* Traduction
* Question answering

See the ``examples`` directory for examples of dataset usage.