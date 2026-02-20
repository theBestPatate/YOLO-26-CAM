# EigenCAM for YOLO 26 Interpretability

A package for applying EigenCAM (it is like GradCAM) and generating heatmaps for YOLO models. Simply clone the package and import the modules to get started.

The basic structure is close to [Jacob Gil&#39;s package for AI explainability](https://github.com/jacobgil/pytorch-grad-cam) and modified to be used for YOLO models.

## Use Cases

It can be used on YOLO classification, segmentation and object detection models. **Now supports YOLO 26, YOLO V12, YOLO V11, YOLO V8, and older models** - all you have to do is just pass the model and see it work automatically. Example notebooks for V8, V11, and V26 provided.

You can also send pull request for adding more functions to it.

## What is EigenCAM

EigenCAM is a technique that involves computing the first principle component of the 2D activations in a neural network, without taking class discrimination into account, and has been found to produce effective results.

#### Image:

<img src="example/images/puppies.jpg" alt="puppies" width="240" height="240">

#### GrayScale Heatmaps:

| Object Detection         | Classification             | Segmentation               |
| ------------------------ | -------------------------- | -------------------------- |
| ![od3.png](example/images/od3.png) | ![cls3.png](example/images/cls3.png) | ![seg3.png](example/images/seg3.png) |

#### Combined

| Object Detection         | Classification             | Segmentation               |
| ------------------------ | -------------------------- | -------------------------- |
| ![od1.png](example/images/od1.png) | ![cls1.png](example/images/cls1.png) | ![cls1.png](example/images/seg1.png) |

### Object Detection model

![od2.png](example/images/od2.png)

### Classification model

![cls2.png](example/images/cls2.png)

### Segmentation model

![seg2.png](example/images/seg2.png)

## Getting Started

#### Simply clone this repository or just download the yolo_cam folder. You must have the yolo_cam folder in the same location as your notebook

#### Import the libraries first:

```python
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
```
#### Choose the target layers and the Yolo Model and load it on cpu
```python
model = YOLO('models/yolo11n-cls.pt') 
model = model.cpu()
target_layers =[model.model.model[-1].conv]
```
#### Call the function and print the image (tasks supported = 'detection', 'classification' and 'segmentation')

```python
with EigenCAM(model, target_layers,task='classificaton') as cam:
    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.show()
```

#### For the Object Detection Task, just change the task to 'detection' and the rest is same.

```python
cam = EigenCAM(model, target_layers,task='detection')
```

The default task is 'detection' so it is fine even if you don't specify the task then

#### Check out the Jupyter Notebooks (YOLO V8n EigenCAM, YOLO V11 EigenCAM, and YOLO V26 EigenCAM) to understand it better and also handle any issues.

## Supported YOLO Versions

✅ **YOLO 26** - Latest model with full support for classification, object detection, and segmentation  
✅ **YOLO V12** - Full support  
✅ **YOLO V11** - Full support  
✅ **YOLO V8** - Full support  
✅ **Older YOLO versions** - Should work with most models

## ToDo:

See the [open issues](https://github.com/rigvedrs/Yolo-V8-CAM/issues) for a list of proposed features (and known issues).

- [X] Solve the issue with having to re-run the cells
- [X] Add support for segmentation model
- [X] Add support for YOLO 26 models
- [ ] Add support for pose detection model
- [ ] Solve pending issues

## Contributing

The open source community thrives on contributions, making it an incredible space for learning, inspiration, and creativity. Please feel free to share any contributions you have for this project.

- Create your Feature Branch (`git checkout -b feature/CoolFeature`)
- Commit your Changes (`git commit -m 'Add some CoolFeature'`)
- Push to the Branch (`git push origin feature/CoolFeature`)
- Open a Pull Request

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=rigvedrs/YOLO-V12-CAM&type=Date)](https://star-history.com/#rigvedrs/YOLO-V12-CAM&Date)



## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white "My email ID")](mailto:rigvedrs@gmail.com)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white "Visit my LinkedIn profile")](https://www.linkedin.com/in/rigvedrs/)
