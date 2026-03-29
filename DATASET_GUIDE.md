# Dataset Guide: Drowsiness Detection Models

To train the state-of-the-art `MobileNetV2` binary eye state classifier, this pipeline provides native support for several public datasets.

## 1. MRL Eye Dataset (Built-In Support)
We highly recommend the **MRL Eye Dataset**, containing ~84,000 images representing various eye states, lighting conditions, and subjects. The dataset pipeline in `train/dataset.py` automatically parses MRL metadata natively.

### Instructions:
You can automatically download it by initializing the data loader with `download=True` or explicitly calling `MRLEyeDataset(download=True)`.

**Manual Method**:
1. Download `mrlEyes_2018_01.zip` from [MRL's Official Website](http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip).
2. Extract the contents inside `train/data/mrlEyes/`.
3. The dataloader extracts Eye State labels by splitting the filename (0 = Closed, 1 = Open).

---

## 2. CEW (Closed Eyes in the Wild)
The CEW dataset consists of pre-cropped images of eyes, explicitly separated into `ClosedFace` and `OpenFace` folders. 

### Instructions:
If you download the CEW dataset, place the directories like so:
```
train/data/CEW/
 │
 ├── OpenFace/
 │   ├── image_0001.jpg
 │   └── image_0002.jpg
 │
 └── ClosedFace/
     ├── image_0001.jpg
     └── image_0002.jpg
```

To integrate with `train.py`, replace `MRLEyeDataset` with PyTorch's native `ImageFolder`:
```python
from torchvision.datasets import ImageFolder
dataset = ImageFolder(root="data/CEW/", transform=train_transform)
```

---

## 3. NTHU Drowsy Driver Video Dataset
NTHU provides live video feeds of drivers. Since our CNN operates on eye crops, video frames must be preprocessed.

### Instructions:
1. Download the MP4 videos from the NTHU repository.
2. Iterate through each video using `cv2.VideoCapture`.
3. Feed each frame into the `PreprocessingPipeline` initialized in `main.py` to extract `left_eye` and `right_eye` ROI crops.
4. Save the returned `left_eye` and `right_eye` outputs as `.png` files into folders (e.g. `Closed/` and `Open/`) based on the NTHU ground truth timestamps.
5. Use the `ImageFolder` procedure mentioned in the CEW guide to train.
