from afforddp.env.CabinetManip import CabinetManipEnv
from afforddp.utils.seed import set_seed
import numpy as np
from afforddp.gym_util.utils import read_yaml_config
import torch
import glob
from isaacgym import gymapi
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
import sys
import tqdm
import os
import shutil
from isaacgym import gymutil
import argparse
import cv2
from PIL import Image
import multiprocessing as mp
from multiprocessing import Lock
import time

torch.set_printoptions(precision=4, sci_mode=False)

def save_config(data, save_config_path):

    with open(save_config_path, 'w') as file:
        json.dump(data, file, indent=4)

def create_video_from_images(video_dir, output_path, fps=20):
    """Create MP4 video from images in video_dir"""
    image_files = sorted(glob.glob(os.path.join(video_dir, "step-*.png")))

    if not image_files:
        return None

    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0])
    height, width, _ = first_img.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_file in image_files:
        img = cv2.imread(img_file)
        video_writer.write(img)

    video_writer.release()
    return output_path

def convert_video_to_gif(video_path, gif_path, fps=20):
    """Convert MP4 video to GIF"""
    # Read video
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if frames:
        # Save as GIF
        duration = int(1000 / fps)  # duration per frame in milliseconds
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                      duration=duration, loop=0, optimize=False)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='PullDrawer.yaml', help='environment config name')
    parser.add_argument('--save_dir', type=str, default='record_test', help='the path where the expert demonstrations are stored')
    parser.add_argument('--obj_id', type=int, default=47024, help='gapartnet asset id')
    parser.add_argument('--part_id', type=int, default=-1, help='select part to manipulation')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--num_parallel', type=int, default=1, help='number of parallel processes for demonstration collection')

    args = parser.parse_args()

    return args

def get_next_traj_id(save_data_dir, lock=None):
    """Thread-safe function to get next available trajectory ID"""
    if lock:
        lock.acquire()
    try:
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
            return 0

        path = os.listdir(save_data_dir)
        path = [item for item in path if item.startswith("traj_") and item != "config.json"]

        if not path:
            return 0

        path.sort(key=lambda x: int(x.split('_')[-1]))
        return int(path[-1].split('_')[-1]) + 1
    finally:
        if lock:
            lock.release()

def collect_demo_worker(args, worker_id, demos_to_collect, lock, progress_queue):
    """Worker function to collect demonstrations in parallel"""

    # Set unique seed for each worker
    set_seed(args.seed + worker_id)

    task_name = args.config_name.split(".")[0]
    if not task_name in ['PullDrawer', 'OpenDoor']:
        raise ValueError(f'Invalid task_type: {task_name}')

    config_path = os.path.join(os.getcwd(), "afforddp/config/env", args.config_name)
    cfgs = read_yaml_config(config_path)
    obj_id = args.obj_id
    cfgs['asset']['arti']['arti_gapartnet_ids'] = [obj_id]

    # Force headless mode for parallel workers (except worker 0 if only 1 worker)
    if args.num_parallel > 1 or worker_id > 0:
        cfgs['HEADLESS'] = True

    gym = CabinetManipEnv(cfgs)

    save_data_dir = f"{args.save_dir}/{task_name}/{obj_id}"

    # First worker saves config
    if worker_id == 0:
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        config_save_path = f'{save_data_dir}/config.json'
        if not os.path.exists(config_save_path):
            save_config(cfgs, config_save_path)

    # Wait a bit for directory creation
    time.sleep(0.5)

    count = 0
    while count < demos_to_collect:
        try:
            gym.reset(to_reset="all")
            gym.get_gapartnet_anno()

            # render bbox for visualization and debug (only if not headless)
            if not cfgs["HEADLESS"] and True:
                gym.gym.clear_lines(gym.viewer)

            gym.cal_handle(bbox_id=args.part_id)

            # Get next trajectory ID in a thread-safe manner
            traj_id = get_next_traj_id(save_data_dir, lock)
            save_root = f'{save_data_dir}/traj_{traj_id}'

            # Create directory
            try:
                os.makedirs(save_root)
            except FileExistsError:
                # Another process created this, try next ID
                continue

            success = gym.motion_planning(save_video=True, save_root=save_root, task_type=task_name)

            if not success:
                shutil.rmtree(save_root)
            else:
                # Generate video and GIF only if successful
                video_dir = os.path.join(save_root, 'video')
                if os.path.exists(video_dir):
                    mp4_path = os.path.join(save_root, 'trajectory.mp4')
                    gif_path = os.path.join(save_root, 'trajectory.gif')
                    create_video_from_images(video_dir, mp4_path, fps=20)
                    convert_video_to_gif(mp4_path, gif_path, fps=20)

                count += 1
                # Send progress update to queue
                progress_queue.put(1)

        except Exception as e:
            print(f"Worker {worker_id} encountered error: {e}")
            continue

    del gym
    return count

def collect_demo(args):
    """Main function to orchestrate parallel demonstration collection"""

    task_name = args.config_name.split(".")[0]
    config_path = os.path.join(os.getcwd(), "afforddp/config/env", args.config_name)
    cfgs = read_yaml_config(config_path)
    total_demos = cfgs['num_demos']

    if args.num_parallel == 1:
        # Single process mode - original behavior with progress bar
        manager = mp.Manager()
        progress_queue = manager.Queue()

        with tqdm.tqdm(total=total_demos) as pbar:
            pbar.set_description('##### Collecting demonstrations:')

            # Create a wrapper to update progress bar
            class ProgressTracker:
                def __init__(self, pbar):
                    self.pbar = pbar
                def put(self, val):
                    self.pbar.update(val)

            collect_demo_worker(args, 0, total_demos, None, ProgressTracker(pbar))
        return

    # Parallel mode
    print(f"Starting parallel collection with {args.num_parallel} workers")
    print(f"Total demonstrations to collect: {total_demos}")

    # Calculate demos per worker
    demos_per_worker = total_demos // args.num_parallel
    extra_demos = total_demos % args.num_parallel

    # Create shared lock and progress queue
    manager = mp.Manager()
    lock = manager.Lock()
    progress_queue = manager.Queue()

    # Create worker processes
    processes = []
    for i in range(args.num_parallel):
        # Distribute extra demos among first workers
        worker_demos = demos_per_worker + (1 if i < extra_demos else 0)

        p = mp.Process(
            target=collect_demo_worker,
            args=(args, i, worker_demos, lock, progress_queue)
        )
        p.start()
        processes.append(p)
        print(f"Worker {i} started - collecting {worker_demos} demonstrations")

    # Monitor progress
    with tqdm.tqdm(total=total_demos) as pbar:
        pbar.set_description('Collecting demonstrations')
        completed = 0
        while completed < total_demos:
            try:
                progress_queue.get(timeout=1)
                completed += 1
                pbar.update(1)
            except:
                # Check if all processes are still alive
                if all(not p.is_alive() for p in processes):
                    break

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print(f"\nCollection complete! Total demonstrations collected: {total_demos}")

if __name__ =='__main__':
    args = parse_args()
    collect_demo(args)
