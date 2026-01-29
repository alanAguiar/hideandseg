# HideAndSeg: Automated Octopus Segmentation in Natural Habitats

**HideAndSeg: An AI-based tool with automated prompting for octopus segmentation in natural habitats**

**Alan de Aguiar**, **Michaella Pereira Andrade**, **Charles Morphy D. Santos**, **João Paulo Gois**

[[`Paper`](https://arxiv.org/abs/2511.04426)] [[`BibTeX`](#citing-hideandseg)]

---

**HideAndSeg** is a minimally supervised framework for segmenting octopuses in underwater videos recorded in natural habitats. The method addresses the challenges posed by octopus camouflage, rapid appearance changes, non-rigid deformations, frequent occlusions, and adverse underwater imaging conditions, while operating in scenarios where large-scale annotated datasets are unavailable.

HideAndSeg integrates **SAM 2** with a custom-trained **YOLOv11** object detector to progressively automate the segmentation process. Initial point-based prompts are used to generate segmentation masks, which serve as pseudo-labels to train the detector. Once trained, the detector provides bounding box prompts to SAM 2, enabling fully automated segmentation without further manual intervention.

---

## Method

The HideAndSeg pipeline consists of three stages:

1. **Prompted Initialization**  
   Users provide point prompts on selected frames. SAM 2 generates initial segmentation masks from these inputs.

2. **Detector Training**  
   The generated masks are used as training data for a YOLOv11 object detector, which learns to localize the octopus across frames.

3. **Automated Segmentation**  
   The trained detector supplies bounding box prompts to SAM 2, enabling automatic and temporally consistent segmentation throughout the video.

This design allows HideAndSeg to recover segmentation after complete occlusions, a scenario where manually prompted segmentation fails.

---

## Evaluation Without Ground Truth

In the absence of annotated datasets, HideAndSeg introduces two unsupervised metrics to evaluate segmentation quality:

- **Temporal Consistency** – measures stability of segmentation masks across consecutive frames.
- **New Component Count** – quantifies the emergence of spurious connected components as an indicator of segmentation noise.

These metrics enable quantitative comparison between manually prompted and automated segmentation approaches.

---

## Results

HideAndSeg achieves:

- Reduced segmentation noise compared to manually prompted SAM 2
- Improved temporal coherence in challenging underwater videos
- Successful re-identification and segmentation after full occlusions
- Robust performance in real-world natural environments

The framework provides a practical tool for reducing manual effort in large-scale behavioral studies of wild cephalopods.

---

## Applications

- Octopus and cephalopod behavioral analysis  
- Long-term monitoring in natural habitats  
- Minimally supervised video segmentation  
- Marine biology and ecological research  

---

## Citing HideAndSeg

If you use HideAndSeg in your research, please cite:

```bibtex
@misc{deaguiar2025hideandsegaibasedtoolautomated,
      title={HideAndSeg: an AI-based tool with automated prompting for octopus segmentation in natural habitats}, 
      author={Alan de Aguiar and Michaella Pereira Andrade and Charles Morphy D. Santos and João Paulo Gois},
      year={2025},
      eprint={2511.04426},
      archivhttps://arxiv.org/help/api/indexePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.04426}, 
}
