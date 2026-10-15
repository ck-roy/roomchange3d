# Third-party software and model notices

RoomChange3D code is MIT licensed. Dependencies are installed from their upstream packages; no third-party source or model weights are checked into this repository.

## VGGT code and checkpoint

The official source is [facebookresearch/vggt](https://github.com/facebookresearch/vggt). Its source uses Meta's custom VGGT License (see [upstream LICENSE.txt](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt)). The upstream README states that the code license was updated for commercial use subject to the license and acceptable-use terms. The default Hugging Face checkpoint [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B) is marked **CC BY-NC 4.0**. It is non-commercial. The upstream says a separate commercial checkpoint requires an application. This project does not redistribute either checkpoint.

By default, the first analysis calls `VGGT.from_pretrained("facebook/VGGT-1B")`; Hugging Face Hub downloads the model to the user's local cache. The official project lists a separate `facebook/VGGT-1B-Commercial` checkpoint, available by application. After access is granted, set `ROOMCHANGE_VGGT_MODEL=facebook/VGGT-1B-Commercial` before launching to select it. Review and accept the selected checkpoint's terms before use or publishing outputs.

## Torchvision object recognition

`torchvision` Faster R-CNN COCO weights and ResNet-50 ImageNet-1K verification weights are downloaded by the library on first use. The ResNet verifier rejects ambiguous crop labels before they can become reported events. See the [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE) and [COCO dataset terms](https://cocodataset.org/#termsofuse) for details. No detector weights or COCO images are redistributed by this project.

## Open-vocabulary object recognition

The optional `google/owlv2-base-patch16-ensemble` checkpoint is downloaded from [Hugging Face](https://huggingface.co/google/owlv2-base-patch16-ensemble) on first use. Its model card specifies Apache-2.0. We do not redistribute the weights. OWLv2 scores are category hypotheses; the pipeline applies a separate threshold and requires repeated consecutive-frame evidence.

## Main runtime packages

PyTorch (BSD-style), torchvision (BSD-style), Transformers (Apache-2.0), Streamlit (Apache-2.0), Plotly (MIT), NumPy (BSD), SciPy (BSD), OpenCV Python (Apache-2.0), Pillow (HPND), pandas (BSD), pytest (MIT), Hugging Face Hub (Apache-2.0). Transitive dependencies have their own licenses; consult installed package metadata and upstream notices.

