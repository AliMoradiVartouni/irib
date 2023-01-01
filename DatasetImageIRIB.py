import torch
print(torch.__version__)
print(torch.cuda.is_available(),torch.cuda.device_count(),torch.version.cuda)#,torch.cuda.current_device(),torch.cuda.get_device_name(0))
# print(torch.__config__.show())

import csv
import pandas as pd
import os
import shutil
import zipfile
from io import StringIO, BytesIO
import ffmpeg
from tkinter import *
from tkinter import ttk
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, status, Response, Request, Query
# from fastapi.staticfiles import StaticFiles
import uvicorn
# from pydantic import BaseModel
# from typing import Union
from fastapi.responses import RedirectResponse, FileResponse, StreamingResponse, PlainTextResponse
from fastapi_login import LoginManager
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_login.exceptions import InvalidCredentialsException
# from pathlib import Path
from FairFace import predict_bbox
import face_recognition
import cv2
import imutils
import json
from datetime import datetime

# print (cv2.__version__)
# from deepface import DeepFace
from fer import FER
import matplotlib.pyplot as plt

from minio import Minio
from dotenv import load_dotenv
from minio.commonconfig import Tags

client = Minio("127.0.0.1:9000",secure=False,access_key="ali",secret_key="ali123456")
"""
To activate minio service for this:
sudo ./minio server /home/avsd/minio_repository
"""


SECRET = 'your-secret-key'

# Initialize an instance of FastAPI
app = FastAPI()
# app.flag_capture_video = 'opencv'

# Define the default route
@app.get("/")
def root():
    return PlainTextResponse("Welcome to DataFace FastAPI\n\n"
                             "This API includes Fairface, Face verification, and Emotion detection"
                             "\n\nYou must upload video. Also, for verification you can select base images file."
                             "\n\n\nAdd /docs end of uri: 172.16.13.194:8000/docs")

manager = LoginManager(SECRET, token_url='/auth/token')
fake_db = {'moradi_ali@irib.ir': {'name': 'Ali Moradi','password': '123456'},
           's_ghanbari@irib.ir':{'name': 'Shirin Ghanbari','password':'123456'}}

@manager.user_loader()
def load_user(email: str):  # could also be an asynchronous function
    user = fake_db.get(email)
    return user

# the python-multipart package is required to use the OAuth2PasswordRequestForm
@app.post('/auth/token')
def login(data: OAuth2PasswordRequestForm = Depends()):
    email = data.username
    password = data.password

    user = load_user(email)  # we are using the same function to retrieve the user
    if not user:
        raise InvalidCredentialsException  # you can also use your own HTTPException
    elif password != user['password']:
        raise InvalidCredentialsException

    access_token = manager.create_access_token(
        data=dict(sub=email)
    )
    return {'access_token': access_token, 'token_type': 'bearer'}

def unzip_base_image(name):
    myfile = client.get_object("face-verify-images", name)
    shutil.rmtree('./FaceVerification/input_image')
    os.mkdir("./FaceVerification/input_image")
    with zipfile.ZipFile(BytesIO(myfile.read()), 'r') as zip_file:
        for member in zip_file.namelist():
            filename = os.path.basename(member)
            # skip directories
            if not filename:
                continue

            # copy file (taken from zipfile's extract)
            source = zip_file.open(member)
            target = open(os.path.join("./FaceVerification/input_image", filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)

def objs():
    objects = client.list_objects("face-verify-images", recursive=True)
    objs= []
    for element in objects:
        for key, value in element.__dict__.items():
            if key == "_object_name":
                objs.append(value)
    return objs

@app.get('/SelectBaseImage',tags=["known file for verification"])
async def selection_and(film : str= Query("{}".format(objs()[0]), enum = ["{}".format(objs()[i]) for i in range(len(objs()))])):
    unzip_base_image(film.title())
    return({"film": film})
# @app.get("/SetBaseImage")
# async def images4verify(user=Depends(manager)):
#     objects = client.list_objects("face-verify-images", recursive=True)
#     objs= []
#     for element in objects:
#         for key, value in element.__dict__.items():
#             if key == "_object_name":
#                 objs.append(value)
#
#
#     root = Tk()
#     frm = ttk.Frame(root, padding=10)
#     frm.grid()
#     ttk.Label(frm, text="Select a following film!").grid(column=0, row=0)
#
#     varbutt = []
#     for i,element in enumerate(objs):
#         varbutt.append(IntVar())
#         Checkbutton(root, text=element, variable=varbutt[i]).grid(row=i+1, sticky=W)
#
#     def var_states():
#         for j,var in enumerate(varbutt):
#             if var.get() == 1:
#                 name = objs[j]
#                 print(name)
#                 return root.destroy(),unzip_base_image(name)
#
#     ttk.Button(root, text='Set', command=var_states).grid(row=i+2, column=0, sticky=W, pady=3)
#     # ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=i+2, sticky=W, pady=3)
#     root.mainloop()
#
#     # return client.list_objects("face-verify-images")



@app.post("/UploadBaseImageVerify",tags=["known file for verification"])
async def create_image_file(myfile: UploadFile,name,user=Depends(manager)):
    load_dotenv()
    client.fput_object("face-verify-images",name, myfile.file.fileno())
    print("It is successfully uploaded to bucket")

@app.post("/UploadVideo",tags=["tagging"])
async def create_upload_file(myfile: UploadFile,  film_season, film_episod,
                             model : str= Query("{}".format('fps'), enum = ["fps","opencv"]),user=Depends(manager)):
    file_location = f"./Aux_Files_Face/Films/{myfile.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(myfile.file.read())
    # return {"info": f"file '{myfile.filename}' saved at '{file_location}'"}
    yourfile = './Aux_Files_Face/Films/' + myfile.filename
    print("{}".format(yourfile))
    dir_output = './Aux_Files_Face/Images'
    if os.path.exists(dir_output):
        os.system("rm -rf " + dir_output)
    os.makedirs(dir_output)

    print(user)

    tags = Tags.new_object_tags()
    tags["film_name"] = myfile.filename
    tags["film_season"] = film_season
    tags["film_episod"] = film_episod
    tags["user"] = [k for k, v in fake_db.items() if v == user][0]

    if model == 'fps':
        create_frames_fps(yourfile, dir_output,20)
        tags["model_frame"] = "FPS20"
    if model == 'opencv':
        create_frames_opencv(yourfile, dir_output)
        tags["model_frame"] = "opencv"

    if not "film_frames_spec" in client.list_objects("keeper"):
        client.put_object("keeper", "film_frames_spec", BytesIO(b"Ali Moradi"), 10)
    client.set_object_tags("keeper", "film_frames_spec", tags)

    from_folder = "Aux_Files_Face/Images/"  # @param {type:"string"}
    # BASE_DIR = from_folder+'.zip'
    files_in = sorted(os.listdir(from_folder))

    images = [i for i in files_in]
    df = pd.DataFrame()
    df['img_path'] = [from_folder + str(x) for x in images]

    df.to_csv('Aux_Files_Face/test_imgs.csv', header='img_path')
    return {"filename": yourfile}

def create_frames_fps(yourfile, dir, intervals =20):
    probe = ffmpeg.probe(yourfile)
    print(probe)
    # time = float(probe['streams'][0]['duration']) // 2
    time = float(probe['format']['duration'])
    width = probe['streams'][0]['width']

    # Set how many spots you want to extract a video from.
    # parts = 7
    # intervals = time // parts
    intervals = int(intervals)

    print(time)
    parts = int(time / intervals)

    interval_list = [(i * intervals, (i + 1) * intervals) for i in range(parts)]
    i = 0

    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    for item in interval_list:
        (
            ffmpeg
                .input(yourfile, ss=item[1])
                .filter('scale', width, -1)
                .output('./Aux_Files_Face/Images/Image' + str(i + 1) + '.jpg', vframes=1)
                .run()
        )
        i += 1

def create_frames_opencv(yourfile, dir):
    # initialize the background subtractor
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
    # initialize a boolean used to represent whether or not a given frame
    # has been captured along with two integer counters -- one to count
    # the total number of frames that have been captured and another to
    # count the total number of frames processed
    min_percent = 1
    max_percent = 10
    warmup = 200
    captured = False
    total = 0
    frames = 0
    # open a pointer to the video file initialize the width and height of
    # the frame
    vs = cv2.VideoCapture(yourfile)
    (W, H) = (None, None)

    # loop over the frames of the video
    while True:
        # grab a frame from the video
        (grabbed, frame) = vs.read()
        # if the frame is None, then we have reached the end of the video file
        if frame is None:
            break
        # clone the original frame (so we can save it later), resize the
        # frame, and then apply the background subtractor
        orig = frame.copy()
        frame = imutils.resize(frame, width=600)
        mask = fgbg.apply(frame)
        # apply a series of erosions and dilations to eliminate noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # if the width and height are empty, grab the spatial dimensions
        if W is None or H is None:
            (H, W) = mask.shape[:2]
        # compute the percentage of the mask that is "foreground"
        p = (cv2.countNonZero(mask) / float(W * H)) * 100
        # print('p=',p, ' captured=', captured, ' frames=', frames)

        # if there is less than N% of the frame as "foreground" then we
        # know that the motion has stopped and thus we should grab the
        # frame
        if p < min_percent and not captured and frames > warmup:
            # show the captured frame and update the captured bookkeeping
            # variable
            # cv2.imshow("Captured", frame)
            captured = True
            # construct the path to the output frame and increment the
            # total frame counter
            filename = "Image{}.jpg".format(total)
            path = os.path.sep.join([dir, filename])
            total += 1
            # save the  *original, high resolution* frame to disk
            print("[INFO] saving {}".format(path))
            cv2.imwrite(path, orig)
        # otherwise, either the scene is changing or we're still in warmup
        # mode so let's wait until the scene has settled or we're finished
        # building the background model
        elif captured and p >= max_percent:
            captured = False

        # display the frame and detect if there is a key press
        # cv2.imshow("Frame", frame)
        # cv2.imshow("Mask", mask)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # increment the frames counter
        frames += 1
    # do a bit of cleanup
    vs.release()

def predict_RaceGenderAge():
    vectorizer = pd.read_csv('Aux_Files_Face/test_imgs.csv')
    boxes= predict_bbox.boxes(vectorizer['img_path'])
    predict_bbox.prediction(boxes)
    # frame = pd.read_csv("./Aux_Files_Face/RAG_outputs.csv")
    # output = BytesIO()
    # with pd.ExcelWriter(output) as writer:
    #     frame.to_excel(writer)
    # headers = {
    #     'Content-Disposition': 'attachment; filename="RAG_outputs.csv"'
    # }
    # return StreamingResponse(iter([output.getvalue()]), headers=headers)

def verify():
    Result_Path = './FaceVerification/result_image'
    KNOWN_FACES_DIR = './FaceVerification/input_image'
    UNKNOWN_FACES_DIR = 'Aux_Files_Face/Images'
    CSV_PATH = './Aux_Files_Face/'

    TOLERANCE = 0.6
    FRAME_THICKNESS = 3
    FONT_THICKNESS = 2
    MODEL = 'cnn'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

    data_known = ['']
    for name in os.listdir(KNOWN_FACES_DIR):
        # tname = input('Enter Character Name for Known Image (' + name + ') :')
        tname = name
        data_known.append([name, tname])
    print(data_known)

    with open(CSV_PATH + 'facesverify.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data_known)

    del data_known[0]
    if os.path.exists(Result_Path) == True:
        shutil.rmtree(Result_Path)
    os.mkdir(Result_Path)

    # Returns (R, G, B) from name
    def name_to_color(name):
        # Take 3 first letters, tolower()
        # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
        color = [(ord(c.lower()) - 97) * 8 for c in name[:3]]
        return color

    print('Loading known faces...')
    known_faces = []
    known_names = []

    # We oranize known faces as subfolders of KNOWN_FACES_DIR
    # Each subfolder's name becomes our label (name)
    for name in os.listdir(KNOWN_FACES_DIR):
        # Next we load every file of faces of known person
        # for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        # for filename in os.listdir(KNOWN_FACES_DIR):

        # Load an image
        image = face_recognition.load_image_file(KNOWN_FACES_DIR + '/' + name)

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encode_res = face_recognition.face_encodings(image);
        if encode_res is not None:
            encoding = encode_res[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)

    print('Processing unknown faces...')
    # Now let's loop over a folder of faces we want to label
    for filename_un in os.listdir(UNKNOWN_FACES_DIR):

        # Load image
        print(f'Filename {filename_un}', end='')
        image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename_un}')

        # This time we first grab face locations - we'll need them to draw boxes
        locations = face_recognition.face_locations(image, model=MODEL)

        # Now since we know loctions, we can pass them to face_encodings as second argument
        # Without that it will search for faces once again slowing down whole process
        encodings = face_recognition.face_encodings(image, locations)

        # We passed our image through face_locations and face_encodings, so we can modify it
        # First we need to convert it from RGB to BGR as we are going to work with cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
        print(f', found {len(encodings)} face(s)')
        dt_1 = [0] * (len(data_known) + 1)
        for face_encoding, face_location in zip(encodings, locations):

            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face withing a tolerance
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                for i, item in enumerate(data_known):
                    if (results[i] == True): dt_1[i + 1] = 1

                match = known_names[results.index(True)]
                # for i, dt in enumerate(data_known):
                #     if (results[i].index == True): dt.append(filename_un)
                print(f' - {match} from {results}')

                # Each location contains positions in order: top, right, bottom, left
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                # Get color by name using our fancy function
                color = name_to_color(match)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

                # Wite a name
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), FONT_THICKNESS)

        dt_1[0] = filename_un
        with open(CSV_PATH + 'facesverify.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(dt_1)


        # Show image
        # cv2.imshow(image)
        cv2.imwrite(Result_Path + '/' + filename_un , image)
        cv2.waitKey(0)
        # cv2.destroyWindow(filename)

    print("Finished!!!")

def predict_Emotion():
    CSV_PATH = './Aux_Files_Face/'
    IMG_PATH = './Aux_Files_Face/Detected_Faces'
    header = ['ImageName', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    with open(CSV_PATH + '/Emotion.csv', 'w', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for img_name in os.listdir(IMG_PATH):
        test_image = plt.imread(IMG_PATH + '/' + img_name)
        emo_detector = FER(mtcnn=True)
        captured_emotions = emo_detector.detect_emotions(test_image)
        for i in captured_emotions:
            data = []
            data.append(img_name)
            for name in i["emotions"]:
                data.append(str(i["emotions"][name]))
            print(data)
            with open(CSV_PATH + '/Emotion.csv', 'a', encoding='UTF-8') as f:
                writer = csv.writer(f)
                writer.writerow(data)
    print("---------------------------------")
    print("Finished!!!")
    print("All Result Saved")

def key_return(MainDict):
    return [k[k.find("'") + 1:k.find(".")] for k, v in MainDict.items() if v == '1']

# @app.get("/TetsYolo")
def yolo5():
    # Model

    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
    # model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
    model = torch.hub.load('./yolov5-master', 'custom', path='./yolov5-master/models/yolov5s.pt', source='local')

    # Images
    # imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
    # imgs =  dlib.load_rgb_image("Aux_Files_Face/Images_1/Image3.jpg")
    dir = "Aux_Files_Face/Images/"
    imgs = [dir+f for f in os.listdir(dir)]

    # Inference
    results = model(imgs)

    # # Results
    # results.print()
    # results.save()  # or .show()
    # results.xyxy[0]  # img1 predictions (tensor)
    ODYolo = [{"image":"{}".format(imgs[i][22:]), "objects":(results.pandas().xyxy[i].to_dict(orient='records'))} for i in range(len(imgs))] # img1 predictions (pandas)
    # print(ODYolo)
    return ODYolo

@app.post("/SendToDatabase",tags=["tagging"])
async def create_image_file(film_name = client.get_object_tags("keeper","film_frames_spec")["film_name"]):
    # model_frame = client.get_object("keeper","model_frame")
    tags = client.get_object_tags("keeper","film_frames_spec")

    predict_RaceGenderAge()
    verify()
    predict_Emotion()
    odyolo = yolo5()

    # current dateTime
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d%H%M")
    dst_path = r"/media/avsd/ac3b1711-bb3f-4882-8bfb-3fdf71a4463a/Output_DataFace/"
    # print(os.path.dirname(dst_path))
    directory = "{}{}_{}_{}_{}".format(dst_path,film_name, tags["film_season"], tags["film_episod"], date_time_str)
    if os.path.exists(directory):
        os.system("rm -rf " + directory)
    os.makedirs(directory)

    shutil.copytree("./Aux_Files_Face/Images", directory+"/Images")
    shutil.copytree("./Aux_Files_Face/Detected_Faces", directory+"/Faces")

    spec_json = {}
    spec_json['creation_date'] = now.strftime("%Y-%m-%d-%H-%M-%S")
    spec_json['user'] = tags["user"]
    spec_json['film_name'] = film_name
    spec_json['film_season'] = tags["film_season"]
    spec_json['film_episod'] = tags["film_episod"]
    spec_json['model_frame'] = tags["model_frame"]
    frames = []

    with open('./Aux_Files_Face/facesverify.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print(row)
            object_detect = {}
            object_detect['image'] = row['']
            object_detect['persons'] = key_return(row)

            faces = []
            with open('./Aux_Files_Face/RAG_outputs.csv', newline='') as csvfile_rag:
                reader_rag = csv.DictReader(csvfile_rag)
                for row2 in reader_rag:
                    if list(row2.values())[0].split('/')[2][:list(row2.values())[0].split('/')[2].find("_")] == row[''][:-4]:
                        face_detect = {}
                        face_detect['face'] = list((row2.values()))[0].split('/')[2]
                        face_detect['race'] = row2['race']
                        face_detect['gender'] = row2['gender']
                        face_detect['age'] = row2['age']
                        face_detect['bbox'] = row2['bbox']
                        with open('./Aux_Files_Face/Emotion.csv', newline='') as csvfile_emotion:
                            reader_emotion = csv.DictReader(csvfile_emotion)
                            face_detect['emotion'] = {}
                            for row3 in reader_emotion:
                                if list(row2.values())[0].split('/')[2] == row3['ImageName']:
                                    del row3['ImageName']
                                    face_detect['emotion'] = row3
                        faces.append(face_detect)
            object_detect['faces'] = faces

            for od in odyolo:
                if od['image'] == row['']:
                    object_detect['objects'] = od['objects']
                    continue

            frames.append(object_detect)
    spec_json['frames'] = frames

    # print(spec_json)
    developer_images = json.dumps(spec_json)
    print(developer_images)

    with open("{}/FilmsFrameDataset.json".format(directory), "w") as write_file:
        json.dump(spec_json, write_file)  # encode dict into JSON


# uvicorn FairFace:app --reload
if __name__ == "__main__":
    uvicorn.run("DatasetImageIRIB:app", host="172.16.13.194", port=8000)
    # uvicorn.run("DatasetImageIRIB:app", host="192.168.43.225", port=8000)
    # uvicorn.run(app, host='127.0.0.1', port=8005)
