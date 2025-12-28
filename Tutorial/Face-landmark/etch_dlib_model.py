import urllib.request

url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
print("Download complete!")