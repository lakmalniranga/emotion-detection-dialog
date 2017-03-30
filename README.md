# emotion-detection-dialog
This scripted is created to detect emotions in faces

## Installation
1. Clone repo
2. Download shape predictor and place within same directory
```bash
  https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat
  ```
3. Install dependencies
   [Recommend: setup virtual enviroment with python 2.7 and pip] 
```bash
  dlib        - http://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/
  openCV 3.x  - pip install opencv-python
  numpy       - pip install numpy
  skimage     - pip install scikit-image
  sklearn     - pip install -U scikit-learn
  ```
4. Set your camera port [cam_port]
5. Run main.py
```bash
  python2 main.py
  ```
