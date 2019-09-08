# Automatic Number Plate Recognition

# Dependencies
```bash
	Python3, tensorflow 1.0, numpy, opencv 4
```

# OpenCV 4 on Raspberry pi 3 B+

## Update the system
```bash
	sudo apt-get update && sudo apt-get -y dist-upgrade
	sudo reboot
	sudo apt-get update && sudo apt-get upgrade
	sudo reboot
```

## Installing Dependencies
```bash
	sudo apt-get install -y build-essential
	sudo apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
	sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev
	sudo apt-get install -y libgtk-3-dev 
	sudo apt-get install -y libpng12-0 libpng12-dev
	sudo apt-get install -y libpnglite-dev
	sudo apt-get install -y zlib1g-dbg zlib1g zlib1g-dev 
	sudo apt-get install -y pngtools libtiff5-dev libtiff4 libtiffxx0c2 libtiff-tools 
	sudo apt-get install -y libjpeg8 libjpeg8-dev libjpeg8-dbg libjpeg-progs 
	sudo apt-get install -y ffmpeg  
	sudo apt-get install -y libgstreamer0.10-0-dbg libgstreamer0.10-0  libgstreamer0.10-dev 
	#sudo apt-get install -y libxine1-ffmpeg  libxine-dev libxine1-bin 
	sudo apt-get install -y libunicap2 libunicap2-dev 
	sudo apt-get install -y libdc1394-22-dev libdc1394-22 libdc1394-utils swig 
	sudo apt-get install -y libv4l-0 libv4l-dev 
	#sudo apt-get install -y libpython2.6 python2.6-dev 
	#sudo apt-get install -y pkg-config[*]gvg
	sudo apt-get install -y libxvidcore-dev libx264-dev
	sudo apt-get install -y libatlas-base-dev gfortran
	sudo apt-get install -y python3-dev
	sudo apt-get install -y python3-pip
	sudo pip3 install numpy scipy 
	sudo pip3 install matplotlib
```
```bash
	sudo apt-get install tcl-dev tk-dev python-tk python3-tk
	git clone https://github.com/opencv/opencv.git
	git clone https://github.com/opencv/opencv_contrib.git
```
```bash
	cd ~/opencv
	mkdir build
	cd build
```
```bash
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D ENABLE_NEON=ON \
    -D BUILD_opencv_python2=ON \
    -D BUILD_opencv_python3=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF ..
```
```bash
    sudo nano /etc/dphys-swapfile
	CONF_SWAPSIZE=100 -> CONF_SWAPSIZE=1024
	sudo /etc/init.d/dphys-swapfile stop
	sudo /etc/init.d/dphys-swapfile start
	free -m
```
```bash
	make
```
```bash
	sudo nano /etc/dphys-swapfile

	CONF_SWAPSIZE=1024 -> CONF_SWAPSIZE=100

	sudo /etc/init.d/dphys-swapfile stop
	sudo /etc/init.d/dphys-swapfile start
```
```bash
	pip3 install mahotas
	pip3 install scikit-learn
	pip3 install -U scikit-image
```
```bash
	sudo make install
```

# INSTALLING TESSERACT 4
```bash
	sudo apt-get install libqtgui4 libqt4-test

	cd ~
	git clone https://github.com/thortex/rpi3-tesseract
	cd rpi3-tesseract/release
	./install_requires_related2leptonica.sh
	./install_requires_related2tesseract.sh
	./install_tesseract.sh

	cd ~
	wget https://github.com/tesseract-ocr/tessdata/raw/master/eng.traineddata
	sudo mv -v eng.traineddata /usr/local/share/tessdata/

	pip3 install opencv-contrib-python imutils pytesseract pillow
```

# YOLO Algorithm 
## You can download the yolo pre-trained weights from:

	License Plate Detection https://drive.google.com/file/d/1PQlDcpopizwEVTplGBVvs2kKdR1guFkY/view?usp=sharing

	Objects Recognition https://drive.google.com/file/d/1OaNglrGbvZEi-_uuJ6IRWu4Xi_jEsCEQ/view?usp=sharing


## To run the ANPR algorithm use:
```bash
	$ python3 yolo.py --image images/test.jpg --yolo lp-yolo/
```