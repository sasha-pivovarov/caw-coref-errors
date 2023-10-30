
## Conjunction-Aware Word-Level Coreference Resolution
Conjunction-Aware Word-Level Coreference Resolution (🦅 CAW-coref) improves upon the Word-Level Coreference Resolution architecture (WL-coref) and was accepted to CRAC 2023. The paper is available [here](https://arxiv.org/abs/2310.06165). CAW-coref improves upon WL-coref by 0.9% CONLL F1 score, without sacrificing computational efficiency.

### Disclaimer
This repository is a fork of the Word-Level Coreference Resolution [repository](https://github.com/vdobrovolskii/wl-coref), which contains the code to reproduce the experiments described in the paper of the same name. This original paper is available [here](https://aclanthology.org/2021.emnlp-main.605/).

### List of Changes w.r.t. Word-Level Coreference Resolution
- Updated the Python and Torch version.
- Changed `convert_to_heads.py` to pick the Coordinating Conjunction as head-word for conjoined entities, as described in the Conjunction-Aware Word-Level Coreference Resolution paper.
- [Share CAW-coref model weights](https://www.dropbox.com/scl/fi/yhtf9h9sml91qs8sazdx6/roberta_-e20_2023.09.08_16.14-_release.pt?rlkey=kf60obnpqjyelsg7019g92kv5&dl=0), which improves upon WL-coref by 0.9% CONLL F1 score.


### Table of contents
1. [Preparation](#preparation)
2. [Training](#training)
3. [Evaluation](#evaluation)
5. [Prediction](#prediction)
6. [Citation](#citation)

### Preparation

The following instruction has been tested with Python 3.11.

You will need:
* **OntoNotes 5.0 corpus** (download [here](https://catalog.ldc.upenn.edu/LDC2013T19), registration needed)
* **Python 2.7** to run conll-2012 scripts
* **Java runtime** to run [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml)
* **Python 3.7+** to run the model
* **Perl** to run conll-2012 evaluation scripts
* **CUDA**-enabled machine (48 GB to train, 4 GB to evaluate)

1. Extract OntoNotes 5.0 arhive. In case it's in the repo's root directory:

        tar -xzvf ontonotes-release-5.0_LDC2013T19.tgz
2. Switch to Python 2.7 environment (where `python` would run 2.7 version). This is necessary for conll scripts to run correctly. To do it with with conda:

        conda create -y --name py27 python=2.7 && conda activate py27
3. Run the conll data preparation scripts (~30min):

        sh get_conll_data.sh ontonotes-release-5.0 data
4. Download conll scorers and Stanford Parser:

        sh get_third_party.sh
5. Prepare your environment. To do it with conda:

        conda create -y --name wl-coref python=3.7 openjdk perl
        conda activate wl-coref
        python -m pip install -r requirements.txt
6. Build the corpus in jsonlines format (~20 min):

        python convert_to_jsonlines.py data/conll-2012/ --out-dir data
        python convert_to_heads.py

You're all set!

### Training

If you have completed all the steps in the previous section, then just run:

    python run.py train roberta

Use `-h` flag for more parameters and `CUDA_VISIBLE_DEVICES` environment variable to limit the cuda devices visible to the script. Refer to `config.toml` to modify existing model configurations or create your own.

### Evaluation

Make sure that you have successfully completed all steps of the [Preparation](#preparation) section.

1. [Download](https://www.dropbox.com/scl/fi/yhtf9h9sml91qs8sazdx6/roberta_-e20_2023.09.08_16.14-_release.pt?rlkey=kf60obnpqjyelsg7019g92kv5&dl=0) and save the pretrained model to the `data` directory.

        https://www.dropbox.com/scl/fi/yhtf9h9sml91qs8sazdx6/roberta_-e20_2023.09.08_16.14-_release.pt?rlkey=kf60obnpqjyelsg7019g92kv5&dl=0

2. Generate the conll-formatted output:

        python run.py eval roberta --data-split test

3. Run the conll-2012 scripts to obtain the metrics:

        python calculate_conll.py roberta test 20

### Prediction

To predict coreference relations on an arbitrary text, you will need to prepare the data in the jsonlines format (one json-formatted document per line).
The following fields are requred:

        {
                "document_id": "tc_mydoc_001",
                "cased_words": ["Hi", "!", "Bye", "."],
                "sent_id": [0, 0, 1, 1]
        }

You can optionally provide the speaker data:

        {
                "speaker": ["Tom", "Tom", "#2", "#2"]
        }

`document_id` can be any string that starts with a two-letter genre identifier. The genres recognized are the following:
* bc: broadcast conversation
* bn: broadcast news
* mz: magazine genre (Sinorama magazine)
* nw: newswire genre
* pt: pivot text (The Bible)
* tc: telephone conversation (CallHome corpus)
* wb: web data

You can check [a sample input file](sample_input.jsonlines) for reference.

Then run:

        python predict.py roberta input.jsonlines output.jsonlines

This will utilize the latest weights available in the data directory for the chosen configuration. To load other weights, use the `--weights` argument.

### Citation
For CAW-coref, cite:

    @misc{doosterlinck2023cawcoref,
    title={CAW-coref: Conjunction-Aware Word-level Coreference Resolution}, 
    author={Karel D'Oosterlinck and Semere Kiros Bitew and Brandon Papineau and Christopher Potts and Thomas Demeester and Chris Develder},
    year={2023},
    eprint={2310.06165},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url = "https://arxiv.org/abs/2310.06165",
    }

For the original WL-coref work, cite:

    @inproceedings{dobrovolskii-2021-word,
    title = "Word-Level Coreference Resolution",
    author = "Dobrovolskii, Vladimir",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.605",
    pages = "7670--7675"}
