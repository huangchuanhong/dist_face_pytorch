g++ -std=c++11 -shared -fPIC face_image_aug.cc -o face_image_aug.so `pkg-config opencv --libs --cflags opencv`
