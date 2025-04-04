# SciGisPy: a Novel Metric for Biomedical Text Simplification via Gist Inference Score

This repository contains the code implementation for our paper presented at **EMNLP 2024**, in the **Third Workshop on Text Simplification, Accessibility, and Readability (TSAR)**:

> **SciGisPy: a Novel Metric for Biomedical Text Simplification via Gist Inference Score**  
> [Chen Lyu](mailto:chen.lyu@warwick.ac.uk), [Gabriele Pergola](mailto:gabriele.pergola.1@warwick.ac.uk)  
> *[Proceedings of TSAR Workshop, EMNLP 2024](https://aclanthology.org/2024.tsar-1.10/)*

---

## Overview

**SciGisPy** is a novel evaluation metric specifically designed for assessing the quality of generated simplifications in biomedical text. Unlike traditional metrics that primarily evaluate surface-level or syntactic simplifications (e.g., BLEU, SARI, FKGL), SciGisPy leverages the Gist Inference Score (GIS) concept derived from Fuzzy-Trace Theory. It is tailored to ensure the preservation of essential meaning—'gist'—crucial in biomedical texts aimed at non-expert readers.

---

## Key Contributions

- **Biomedical Adaptation**: Specifically tailored for biomedical contexts.
- **Cognitive-Theoretic Basis**: Grounded in Fuzzy-Trace Theory, measuring gist rather than surface textual changes.
- **Enhanced GIS Components**: Integration of semantic chunking, information content theory, domain-specific embeddings, and domain-aware coherence metrics.
- **Reference-free Evaluation**: Does not require reference simplifications, facilitating practical, real-world deployment.
- **Extensive Validation**: Comprehensive evaluation and ablation studies conducted on the Cochrane biomedical dataset, demonstrating significant performance gains over baseline metrics.

---

## Installation

This code requires Python 3.8.10.

1. **Clone the repository:**
```bash
git clone https://github.com/your-repo/SciGisPy.git
cd SciGisPy
```

2. **Download Pre-trained Models:**
Download [BioWordVec and BioSentVec models](https://github.com/ncbi-nlp/BioSentVec) and place them in the root directory of the repository.

3. **Install dependencies:**
```bash
pip install -r requirement.txt
python -m spacy download en_core_web_trf
```

---

## Usage

Run the provided Jupyter notebook for a detailed usage demonstration and example scenarios:

```bash
jupyter notebook Run_SciGisPy.ipynb
```

---

## Results

Evaluations on the Cochrane biomedical simplification dataset show that SciGisPy:

- Correctly classified simplified texts in 84% of the cases, compared to 44.8% using original GIS.
- Outperformed surface-level metrics such as FKGL, ARI, and SARI by significant margins.
- Ablation studies indicate strong individual contributions from enhancements such as `PCREF_chunk`, `WRDIC`, and `Mean Sentence Length`.

---

## Citation

If you use SciGisPy in your research, please cite our paper:

```bibtex
@inproceedings{lyu-pergola-2024-scigispy,
    title = "SciGisPy: a Novel Metric for Biomedical Text Simplification via Gist Inference Score",
    author = "Lyu, Chen  and
      Pergola, Gabriele",
    booktitle = "Proceedings of the Third Workshop on Text Simplification, Accessibility and Readability (TSAR)",
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.tsar-1.10/"
}
```

---

## Contact

- **Chen Lyu**: [chen.lyu@warwick.ac.uk](mailto:chen.lyu@warwick.ac.uk)
- **Gabriele Pergola**: [gabriele.pergola.1@warwick.ac.uk](mailto:gabriele.pergola.1@warwick.ac.uk)

Please reach out with any questions or feedback.

