import numpy as np
import cv2
from . import audio
from tqdm import tqdm
import torch
from . import face_detection
#import .face_detection
from .models import Wav2Lip
import torch.nn.functional as F
import subprocess
import platform
import os
import pickle
import hashlib
import cv2
import glob
from pathlib import Path

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

# Global cache configuration
CACHE_DIR = "wav2lip_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "last_face_detect_cache.pkl")

def compute_video_hash(images, params_dict=None):
    """Compute hash from first and last frame of video and parameters"""
    if params_dict is None:
        params_dict = {}
    
    # Create hash object
    video_hash = hashlib.sha256()
    
    # Hash parameters first (consistent ordering)
    if params_dict:
        # Convert params to sorted string representation
        params_str = ','.join(f"{k}={v}" for k, v in sorted(params_dict.items()))
        video_hash.update(params_str.encode('utf-8'))
    
    # Hash video frames if available
    if len(images) > 0:
        # Process first frame
        first_frame = images[0]
        if len(first_frame.shape) == 3:
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        # Process last frame
        last_frame = images[-1]
        if len(last_frame.shape) == 3:
            last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        
        # Update hash with frame data
        video_hash.update(first_frame.tobytes())
        video_hash.update(last_frame.tobytes())

    return video_hash.hexdigest()

def clear_previous_caches():
    """Remove all previous cache files"""
    for cache_file in glob.glob(os.path.join(CACHE_DIR, "*.pkl")):
        try:
            if cache_file != CACHE_FILE:  # Don't delete current cache
                os.remove(cache_file)
        except OSError:
            pass
            
def face_detect_with_cache(images, face_detect_batch, pad_bottom=10, fps=25.0):
    """Face detection with caching mechanism"""
    # Create cache directory if needed
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Prepare parameters for cache key
    params = {
        'pad_bottom': pad_bottom,
        'fps': fps
        # Add other parameters here as needed
    }
    
    # Compute combined hash of frames and parameters
    video_hash = compute_video_hash(images, params)
    
    # Check cache existence
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cached_hash, results = pickle.load(f)
            if cached_hash == video_hash:
                print("\nUsing cached face detection results")
                return results
        except Exception as e:
            print(f"Cache loading failed: {str(e)}")
    
    # Compute face detection if cache missing/invalid
    results = face_detect(images, face_detect_batch, pad_bottom)
    
    # Clear previous caches before saving new one
    clear_previous_caches()
    
    # Save new results to cache
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump((video_hash, results), f)
    
    print("Saved new face detection results to cache")
    return results

# Your original face detection function (unchanged)
def face_detect(images, face_detect_batch, pad_bottom=10):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                           flip_input=False, device=device)
    batch_size = face_detect_batch
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size), position=0, leave=True):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = [0, pad_bottom, 0, 0]
    for rect, image in zip(predictions, images):
        try:
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)
                raise ValueError('Face not detected! Ensure the video contains a face in all frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])
        except:
            pass

    boxes = np.array(results)
    boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

def datagen(frames, mels, face_detect_batch, mode, pad_bottom=10, fps=25.0):
    img_size = 96
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    frame_size = len(frames)

    face_det_results = face_detect_with_cache(frames, face_detect_batch, pad_bottom, fps)
    
    repeat_frames = len(mels) / frame_size 
    for i, m in enumerate(mels):
        try:
            if mode == "sequential":
                face_idx = int(i//repeat_frames)
            else:
                face_idx = i%frame_size

            frame_to_save = frames[face_idx].copy()
            face, coords = face_det_results[face_idx].copy()

            face = cv2.resize(face, (img_size, img_size))
                
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= 128:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        except:
            print("box error, no face is found, skipping frame")

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(model_path):
    if device == 'cuda':
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def get_onnx_input_names(model):
    """Get the input names from an ONNX model"""
    return [input.name for input in model.get_inputs()]

def load_model(path):
    if str(path).lower().endswith('.onnx'):
        import onnxruntime as ort
        model = ort.InferenceSession(str(path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # Store the input names mapping in the model object
        input_names = get_onnx_input_names(model)
        if len(input_names) != 2:
            raise ValueError(f"Expected 2 inputs for ONNX model, got {len(input_names)}: {input_names}")
        model.input_names_mapping = {
            'mel': input_names[0],  # First input is typically for audio features
            'img': input_names[1]   # Second input is typically for image
        }
        return model
        
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()
    
def wav2lip_(images, audio_path, face_detect_batch, mode, model_path, pad_bottom=10, fps=25.0):
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    batch_size = 128
    gen = datagen(images.copy(), mel_chunks, face_detect_batch, mode, pad_bottom, fps)

    o=0

    print(f"Load model from: {model_path}")
    model = load_model(model_path)
    
    out_images = []
    out_images_BGR = []
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if (len(mel_batch) < batch_size):
            padding_len = batch_size - len(mel_batch)
            if padding_len:              
                last_img = img_batch[-1][np.newaxis, ...]  # Add an extra dimension to allow concatenation
                img_padding = np.repeat(last_img, padding_len, axis=0)
                img_batch = np.concatenate((img_batch, img_padding), axis=0)
                last_mel = mel_batch[-1][np.newaxis, ...]  # Add an extra dimension to allow concatenation
                mel_padding = np.repeat(last_mel, padding_len, axis=0)
                mel_batch = np.concatenate((mel_batch, mel_padding), axis=0)
                
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device) # 0.001 s
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device) # 0.001 s
        
       
        try:
            with torch.no_grad():
                if isinstance(model, torch.nn.Module):
                    pred = model(mel_batch, img_batch)
                else:  # ONNX model
                    # Use the stored input names mapping
                    input_dict = {
                        model.input_names_mapping['mel']: mel_batch.cpu().numpy(),
                        model.input_names_mapping['img']: img_batch.cpu().numpy()
                    }
                    pred = model.run(None, input_dict)[0]
                    pred = torch.from_numpy(pred)
                
        except RuntimeError as err:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return -1
            
        pred = pred * 255.0
        # Convert frames and coords to PyTorch tensors and move them to the GPU
        frames_gpu = []
        frames_gpu = [torch.from_numpy(frame).to(device) for frame in frames]
        i = 0
        for p, f, c in zip(pred, frames_gpu, coords): # p[3, 96, 96]. 0.003 s
            y1, y2, x1, x2 = c            
            p = p.unsqueeze(dim=0)                       # Add a singleton dimension along axis 0
            if (p.shape[-1] and p.shape[-2]):
                p = F.interpolate(p, size=(int(y2 - y1), int(x2 - x1)), mode='bilinear').squeeze(dim=0).permute(1, 2, 0) # -> [96, 96, 3]    mode='bilinear' or nearest           
                f[y1:y2, x1:x2] = p     # Assign the processed patch to the cloned frame
                frames_gpu[i] = f
                i+=1
                
        # After processing all frames, move them back to CPU memory. 0.003s
        out_images = torch.stack(frames_gpu).cpu().numpy()
        
        for im in out_images:
            # Step 4: Convert RGB → BGR for saving/display with comfy/vhs
            im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            out_images_BGR.append(im_bgr)       
       
    return out_images_BGR