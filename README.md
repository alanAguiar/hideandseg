# HideAndSeg: Automated Octopus Segmentation in Natural Habitats


**Alan de Aguiar**, **Michaella Pereira Andrade**, **Charles Morphy D. Santos**, **João Paulo Gois**

🐙[[`Paper`](https://arxiv.org/abs/2511.04426)] [[`BibTeX`](#citing-hideandseg)]


![HideAndSeg Output](assets/full_output.png?raw=true)

---

**HideAndSeg** is a minimally supervised framework for segmenting octopuses in underwater videos recorded in natural habitats. The method addresses the challenges posed by octopus camouflage, rapid appearance changes, non-rigid deformations, frequent occlusions, and adverse underwater imaging conditions, while operating in scenarios where large-scale annotated datasets are unavailable.

### How It Works
The pipeline integrates **SAM 2 (Segment Anything Model 2)** with a custom-trained **YOLOv11** object detector to progressively automate the segmentation process:
1. Initial point-based prompts generate segmentation masks.
2. These masks are used to train the YOLOv11 detector.
3. Once trained, the detector provides automated bounding box prompts back to SAM 2.
4. The system produces fully automated segmentation without further manual intervention.
---

## Getting Started

### Prerequisites
HideAndSeg requires:
* **Python** $\ge$ 3.10
* **PyTorch** $\ge$ 2.5.1
* **TorchVision** $\ge$ 0.20.1

*Note: Please follow the [official PyTorch instructions](https://pytorch.org/get-started/locally/) to install the correct versions for your hardware before proceeding.*

### Installation

**1. Install SAM 2**
HideAndSeg relies on SAM 2, which must be installed from its source repository:
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

**2. Install HideAndSeg**
Clone this repository and install:

```bash
git clone https://github.com/alanAguiar/hideandseg.git
cd hideandseg
pip install -e .
cd ..
```

### Usage
The pipeline is designed to be used as a Python library:

```python
from hideandseg import HideAndSeg

# Initialize the pipeline
hands = HideAndSeg(
    sam2_model_size="large"
)

# Run segmentation
hands.segment(video_path="video.mp4", output_path="output", n_annotated_frames=5)
```

📚 For a complete walkthrough, check out our [example Jupyter Notebook](https://github.com/alanAguiar/hideandseg/tree/main/examples/video_segmentation.ipynb).

---

## Citing HideAndSeg

If you use HideAndSeg in your research, please cite:

```bibtex
@misc{deaguiar2025hideandsegaibasedtoolautomated,
      title={HideAndSeg: an AI-based tool with automated prompting for octopus segmentation in natural habitats}, 
      author={Alan de Aguiar and Michaella Pereira Andrade and Charles Morphy D. Santos and João Paulo Gois},
      year={2025},
      eprint={2511.04426},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.04426}, 
}
```