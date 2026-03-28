# Interpretable Similarity of Synthetic Image Utility (IUS)
This is the official implementation of IUS measure [**"Interpretable Similarity of Synthetic Image Utility"**](https://arxiv.org/pdf/2512.17080).

## ⌨️ Code Implementation
The official **PyTorch** implementation of the *IUS* measure is currently being prepared and will be publicly released in April 2026. 

## 📄 Paper Abstract
<p align="justify">
Synthetic medical image data can unlock the potential of deep learning (DL)-based clinical decision support (CDS) systems through the creation of large scale, privacy-preserving, training sets. Despite the significant progress in this field, there is still a largely unanswered research question: “How can we quantitatively assess the similarity of a synthetically generated set of images with a set of real images in a given application domain?”. Today, answers to this question are mainly provided via user evaluation studies, inception-based measures, and the classification performance achieved on synthetic images. This paper proposes a novel measure to assess the similarity between synthetically generated and real sets of images, in terms of their utility for the development of DLbased CDS systems. Inspired by generalized neural additive models, and unlike inception-based measures, the proposed measure is interpretable (Interpretable Utility Similarity, IUS), explaining why a synthetic dataset could be more useful than another one in the context of a CDS system based on clinically relevant image features. The experimental results on publicly available benchmark datasets from various color medical imaging modalities including endoscopic, dermoscopic and fundus imaging, indicate that selecting synthetic images of high utility similarity using IUS can result in relative improvements of up to 54.6% in terms of classification performance. The generality of IUS for synthetic data assessment is demonstrated also for grayscale X-ray and ultrasound imaging modalities. IUS implementation is available at https://github.com/innoisys/ius. 
</p>

## 📚 Citation
If you use this code or find our work useful in your research, please cite:
```bibtex
@article{
  author    = {Panagiota Gatoula and George Dimas and Dimitris K. Iakovidis},
  title     = {Interpretable Similarity of Synthetic Image Utility},
  journal   = {IEEE Transactions on Medical Imaging},
  year      = {2026},
  publisher = {IEEE},
}
```

## ⚖️  License
This project is licensed under the MIT License.
