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
    print(img_array.shape)
    a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        print("insideif")
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))
    #print(clip_imgs)
    return clip_imgs

def get_spatio_temporal_features(features, num_temporal_tokens=100):
    print("inside spatio tenpral features funtion")
    t, s, c = features.shape

    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = np.mean(features, axis=0)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

    return sp_features
def main():
    args = parse_args()
    video_dir_path = args.video_dir_path
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    infer_batch = args.infer_batch
    os.makedirs(clip_feat_path, exist_ok=True)

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32', torch_dtype=torch.float16)
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32', torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).cuda()
    vision_tower.eval()

    all_videos = os.listdir(video_dir_path)
    video_clip_features = {}
    all_videos = os.listdir(video_dir_path)
    #video_clip_features = {}
    counter = 0
    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        try:
            video = load_video(video_path)#working here
            video_tensor = image_processor.preprocess(video, return_tensors='pt')['pixel_values']
            print("printing length if video_tensore before half")
            print(len(video_tensor))
            video_tensor = video_tensor.half()

            n_chunk = len(video_tensor)
            print("printing n_chunk")
            print(n_chunk)
            video_features = torch.FloatTensor(n_chunk, 256, 1024).fill_(0)
            print("length of video_features")
            print(len(video_features))
            print("displying video_features")
            print(video_features)
            n_iter = int(math.ceil(n_chunk / float(infer_batch)))
            print("number of iteration")
            print(n_iter)
            for i in range(n_iter):
                print("inside try for")#working here
                min_ind = i * infer_batch
                max_ind = (i + 1) * infer_batch
                video_batch = video_tensor[min_ind:max_ind].cuda()
                print("printing video batch")#
                print(video_batch)#this is printing
                print("length of video_batch")
                print(len(video_batch))
                print("printing min and max id")
                print(min_ind)
                print(max_ind)
                image_forward_outs = vision_tower(video_batch, output_hidden_states=True)
                #print("printing image forward puts")
                #print(image_forward_outs)
                select_hidden_state_layer = -2
                select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                #working with large-patch32
                #print("displaying select hidden state")
                #print(select_hidden_state)
                batch_features = select_hidden_state[:, 1:]# this is a tensor
                #this is also displaying
                print(batch_features)
                #print(len(batch_features)) #answer is 32 
                print("checking if detach on batch features is displaying")
                print(batch_features.detach()) # this is also displaying
                video_features = video_features[min_ind:max_ind].cuda() #this is the line i added 
                video_features=batch_features.detach().cpu() #problem was with this line changed a bit
                print("printing video features")
                print(video_features)
            print("outside try for")
            video_clip_features[video_id] = get_spatio_temporal_features(video_features.numpy().astype("float16"))
            counter += 1
            print("processing video done")
        except Exception as e:
            print(f"Can't process {video_path}")
        if counter % 512 == 0:  # Save after every 512 videos, update this number as per your requirements
            for key in video_clip_features.keys():
                features = video_clip_features[key]
                with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
                    pickle.dump(features, f)
            video_clip_features = {}
    for key in video_clip_features.keys():
        features = video_clip_features[key]
        with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
            pickle.dump(features, f)
if __name__ == "__main__":
    main()

    
                                                                                                                                                      1,1           Top
