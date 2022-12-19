import os
import cv2
import time
import numpy as np
import torch
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
from ibug.face_parsing.utils import label_colormap


def face_parsing_test(input_images, blurring=0, facebox=False, benchmark=False, threshold=0.8, encoder='rtnet50', decoder='fcn', max_num_faces=2, num_classes=11, weights=None, device='cuda:0'):
    '''
    IN
    --input_images: Array of input images
    --blurring: Whether to blur the skin or not (0 only seg, 1 seg+face blur, 2 only blur) (default: 0)
    --facebox: Whether to add red rectangle around face in output or not (default: False)
    --benchmark: Enable benchmark mode for CUDNN (default: False)
    --threshold: Detection threshold (default: 0.8)
    --encoder: Method to use, can be either rtnet50, resnet50, or rtnet101 (default: 'rtnet50')
    --decoder: Method to use, can be either fcn or deeplabv3plus (default: 'fcn')
    --max_num_faces: Max number of faces (default: 2)
    --num_classes: Face parsing classes (default: 11)
    --weights: Weights to load, can be resnet50 or mobilenet0.25 when using RetinaFace (default: None)
    --device: Device to be used by the model (default: cuda:0)


    OUT
    --output_images: Array of segmented images in same order
    '''

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = benchmark
    face_detector = RetinaFacePredictor(threshold=threshold, device=device,
                                        model=(RetinaFacePredictor.get_model('mobilenet0.25')))
    face_parser = RTNetPredictor(
        device=device, ckpt=weights, encoder=encoder, decoder=decoder, num_classes=num_classes)

    colormap = label_colormap(num_classes)
    print('Face detector created using RetinaFace.')

    alphas = np.linspace(0.75, 0.25, num=max_num_faces)
    output_images = np.zeros(shape=input_images.shape)

    # Process the frames
    image_number = 0
    window_title = os.path.splitext(os.path.basename(__file__))[0]
    print('window title', window_title)

    print('Processing started, press \'Q\' to quit.')
    print('Processing images in ')
    for im in input_images:
        # Get a new frame
        frame = np.ndarray.astype(np.round(im.copy()*255, 0), np.uint8)
        # Detect faces
        start_time = time.time()
        faces = face_detector(frame, rgb=False)
        elapsed_time = time.time() - start_time

        # Textural output
        print(f'Image #{image_number} processed in {elapsed_time * 1000.0:.04f} ms: ' +
            f'{len(faces)} faces detected.')
        if len(faces) == 0:
            print('problem')
            #cv2.imshow('image', frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            assert 0
        else:
            # Parse faces
            start_time = time.time()
            masks = face_parser.predict_img(frame, faces, rgb=False)
            elapsed_time = time.time() - start_time


            # Textural output
            print(f'Image #{image_number} processed in {elapsed_time * 1000.0:.04f} ms: ' +
                f'{len(masks)} faces parsed.')
            
            # # Rendering
            dst = frame.copy()
            if blurring:
                blur = cv2.blur(frame, (5,5))
                blur_index = np.zeros((len(faces), frame.shape[0], frame.shape[1], frame.shape[2]))
            for i, (face, mask) in enumerate(zip(faces, masks)):
                if facebox:
                    bbox = face[:4].astype(int)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(
                        0, 0, 255), thickness=2)
                alpha = alphas[i]
                index = mask > 0
                if blurring:
                    b = mask <= 1 # blurring skin + background
                    blur_index[i] = np.stack((b, b, b), axis=-1)
                res = colormap[mask]
                dst[index] = (1 - alpha) * frame[index].astype(float) + \
                    alpha * res[index].astype(float)
            dst = np.clip(dst.round(), 0, 255).astype(np.uint8)
            if blurring == 1:
                for i in range(0, len(faces)):
                    frame = np.where(blur_index[i], blur, dst)
            elif blurring == 2:
                for i in range(0, len(faces)):
                    frame = np.where(blur_index[i], blur, frame)
            else:
                frame = dst

            output_images[image_number] = frame
            image_number += 1
            #cv2.imshow('image', frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

    return output_images
