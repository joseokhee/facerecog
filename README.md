# facerecog

Face recognition pipeline: **Google Cloud Vision API** for face detection, followed by a custom **CNN classifier** (TensorFlow 1.x) on the cropped faces. My first deep learning project (2020).

<p align="center"><img src="https://user-images.githubusercontent.com/33644885/103069846-998e0600-4603-11eb-9448-8cf0851129ed.png" width="70%"></p>

## Model
- Input: 96 × 96 × 3 face crops
- Stacked conv blocks (figure above) with **overlapping max pooling** and **dropout** (as in AlexNet) to reduce overfitting
- **Xavier initialization**

## Files
| File | Purpose |
|---|---|
| `face.py` | Input queue, model definition, training |
| `predict_model.py` | Inference on new images |

## Notes
- Written for TensorFlow 1.x (`tf.app.flags`, queue runners); will not run on TF 2 without `tf.compat.v1`.
- Data paths in `face.py` (`TRAINING_FILE`, `VALIDATION_FILE`) must be set to your own CSV lists (`path,_,label`).

## Reference
Based on the tutorial by Byungwook Cho: http://bcho.tistory.com/1178
