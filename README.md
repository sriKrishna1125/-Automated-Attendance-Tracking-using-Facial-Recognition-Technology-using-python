dlib==19.24.0
face-recognition==1.3.0
numpy==1.24.2
opencv-python


How it works

create a separate folder for each person who appears in your training images. Then you can put all the images into their appropriate folders:

face_recognizer/
│
├── output/
│
├── training/
│   └── ben_affleck/
│       ├── img_1.jpg
│       └── img_2.png
│
├── validation/
│   ├── ben_affleck1.jpg
│   └── michael_jordan1.jpg
│
├── detector.py
├── requirements.txt
