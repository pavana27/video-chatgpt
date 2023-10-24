import os
from transformers import CLIPVisionModel, CLIPImageProcessor
import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.")
    parser.add_argument("--infer_batch", required=False, type=int, default=32,help="Number of frames/images to perform batch inference.")

    args = parser.parse_args()

    return args

def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq

def load_video(vis_path, num_frm=100):
    print("inside load video function")
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    print("total number of frmaes")
    print(total_frame_num)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)
                                                                                                                                                      1,1           Top
