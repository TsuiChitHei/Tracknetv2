from pathlib import Path
import cv2
from enum import Enum
import pandas as pd
import numpy as np
import os
import argparse
import sys
import time

# python tools/label_tool.py match/videos/clip.mp4 --csv_dir match/labels
# python tools/label_tool.py match/videos/clip.mp4 --csv_dir match/labels --step 3
# 若不加csv_dir,则会默认从mp4文件同级目录读取csv
#
# With --step N: n/p jump N frames. Left-click marks a keyframe; frames between
# two visible keyframes are filled by linear interpolation.


# state 0:hidden  1:visible
state_name = ['HIDDEN', 'VISIBLE']

keybindings = {
    'next':          [ ord('n') ],
    'prev':          [ ord('p')],

    'piece_start':   [ ord('s'), ],     # 裁剪开始帧
    'piece_end':     [ ord('e'), ],     # 裁剪结束帧

    'first_frame':   [ ord('z'), ],
    'last_frame':    [ ord('x'), ],

    'forward_frames':   [ ord('f'), ],      #   前进36帧
    'backward_frames':  [ ord('b'), ],      #   后退36帧

    'circle_grow':   [ ord('='), ord('+') ],
    'circle_shrink': [ ord('-'), ],

    'mark_hidden':   [ ord('h'), ],

    'quit':          [ ord('q'), ],
}



class VideoPlayer():
    def __init__(self, opt) -> None:
        self.jump = 36
        self.step = max(1, int(opt.step))
        self.keyframes = set()

        self.cap = cv2.VideoCapture(opt.video_path)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #self.fourcc = int(self.cap.get(cv.CAP_PROP_FOURCC))
        # self.fourcc = cv2.VideoWriter_fourcc('H', 'E', 'V', 'C')
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.bitrate = self.cap.get(cv2.CAP_PROP_BITRATE)

        # Check video lens!
        # ret, frame = self.cap.read()
        # frame_cnt = 1
        # while ret:
        #     ret, frame = self.cap.read()
        #     if ret:
        #         frame_cnt += 1
        #     else:
        #         print("Waringing: {} frame decode error!".format(frame_cnt))
        
        # if self.frames != frame_cnt:
        #     print(" self.frames: {}, frame_cnt: {} ".format(self.frames, frame_cnt))
        #     self.frames = frame_cnt

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


        self.video_path = Path(opt.video_path)
        self.circle_size = 5
        if opt.csv_dir is None:
            self.csv_path = self.video_path.with_suffix('.csv')
        else:
            self.csv_path = Path(opt.csv_dir) / Path(self.video_path.stem).with_suffix('.csv')

        self.window = cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', 1280, 720)

        ret, self.frame = self.cap.read()
        if not ret or self.frame is None:
            raise RuntimeError(
                f'Failed to read the first frame from {opt.video_path}. '
                'In Git Bash use forward slashes, e.g. match/videos/clip.mp4'
            )
        self.frame_num = 0

        self.piece_start = 0
        self.piece_end = 0
        self._finished = False

        if os.path.exists(self.csv_path):
            self.info = pd.read_csv(self.csv_path)

            if len(self.info.index) != self.frames:
                print("pd len: {}, camera len: {}".format(len(self.info.index), self.frames))
                print("Number of frames in video and dictionary are not the same!")
                print("Fail to load!")
                exit(1)


                # self.info = {'frame_num':[], 'visible':[], 'x':[], 'y':[]}

                # for idx in range(self.frames):
                #     self.info['frame_num'].append(idx)
                #     self.info['visible'].append(0)
                #     self.info['x'].append(0)
                #     self.info['y'].append(0)
                
                # print("pandas dataframe len: {}".format(len(self.info)))

            else:
                self.info = {k: list(v.values()) for k, v in self.info.to_dict().items()}
                # Existing visible labels count as keyframes so reopening a CSV keeps them.
                self.keyframes = {
                    i for i in range(self.frames) if int(self.info['visible'][i]) == 1
                }
                print("Load labeled {} successfully.".format(self.csv_path))
        else:
            print("Create new dictionary")
            
            self.info = {'frame_num':[], 'visible':[], 'x':[], 'y':[]}

            for idx in range(self.frames):
                self.info['frame_num'].append(idx)
                self.info['visible'].append(0)
                self.info['x'].append(0)
                self.info['y'].append(0)
                
            print("Created labels for {} frames.".format(self.frames))

        if self.step > 1:
            print(f"Step mode: n/p jump {self.step} frames. Left-click interpolates between keyframes.")

        cv2.setMouseCallback('Frame',self.markBall)
        self.display()


    def save_piece(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.piece_start)

        out = cv2.VideoWriter('{}_{}.mp4'.format(self.piece_start+1, self.piece_end+1), self.fourcc, self.fps, (self.width, self.height))

        frame_cnt = self.piece_start
        while frame_cnt <= self.piece_end:
            ret, frame = self.cap.read()
            out.write(frame)

            frame_cnt += 1
        
        out.release()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
        print("save piece succefully!")

    def seek(self, frame_num):
        frame_num = max(0, min(int(frame_num), self.frames - 1))
        self.frame_num = frame_num
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
        ret, self.frame = self.cap.read()
        if not ret or self.frame is None:
            print(f"Warning: failed to read frame {self.frame_num}")

    def interpolate_between_keyframes(self):
        """Fill non-keyframe frames by linear interpolation between visible keyframes."""
        if len(self.keyframes) < 2:
            return

        keys = sorted(self.keyframes)
        for a, b in zip(keys, keys[1:]):
            if b <= a + 1:
                continue

            va = int(self.info['visible'][a])
            vb = int(self.info['visible'][b])

            if va == 1 and vb == 1:
                xa, ya = self.info['x'][a], self.info['y'][a]
                xb, yb = self.info['x'][b], self.info['y'][b]
                span = b - a
                for i in range(a + 1, b):
                    if i in self.keyframes:
                        continue
                    t = (i - a) / span
                    self.info['frame_num'][i] = i
                    self.info['x'][i] = xa + t * (xb - xa)
                    self.info['y'][i] = ya + t * (yb - ya)
                    self.info['visible'][i] = 1
            else:
                # One or both ends hidden: clear interpolated frames in the gap.
                for i in range(a + 1, b):
                    if i in self.keyframes:
                        continue
                    self.info['frame_num'][i] = i
                    self.info['x'][i] = 0
                    self.info['y'][i] = 0
                    self.info['visible'][i] = 0

    def mark_hidden(self):
        self.info['frame_num'][self.frame_num] = self.frame_num
        self.info['x'][self.frame_num] = 0
        self.info['y'][self.frame_num] = 0
        self.info['visible'][self.frame_num] = 0
        self.keyframes.add(self.frame_num)
        self.interpolate_between_keyframes()
        self.display()

    def markBall(self, event, x, y, flags, param):
        nx = x / self.width
        ny = y / self.height
        if event == cv2.EVENT_LBUTTONDOWN:
            self.info['frame_num'][self.frame_num] = self.frame_num
            self.info['x'][self.frame_num] = nx
            self.info['y'][self.frame_num] = ny
            self.info['visible'][self.frame_num] = 1
            self.keyframes.add(self.frame_num)
            self.interpolate_between_keyframes()
            if self.frame_num < self.frames - 1:
                self.seek(self.frame_num + self.step)
            self.display()

        # Single right-click: OpenCV often never fires RBUTTONDBLCLK on Windows.
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mark_hidden()


    def display(self):
        if self.frame is None:
            return
        res_frame = self.frame.copy()
        res_frame = cv2.putText(res_frame, state_name[self.info['visible'][self.frame_num]], (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        res_frame = cv2.putText(res_frame, "Frame: {}, Total: {}".format(int(self.frame_num+1), int(self.frames)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        res_frame = cv2.putText(res_frame, "Piece: {}-{}".format(int(self.piece_start+1), int(self.piece_end+1)), (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        res_frame = cv2.putText(res_frame, "Step: {}".format(self.step), (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2, cv2.LINE_AA)

        if self.info['visible'][self.frame_num]:
            x = int(self.info['x'][self.frame_num] * self.width)
            y = int(self.info['y'][self.frame_num] * self.height)
            color = (0, 255, 255) if self.frame_num in self.keyframes else (0, 0, 255)
            cv2.circle(res_frame, (x, y), self.circle_size, color, -1)

        cv2.imshow('Frame', res_frame)

    #print("frame num: {}".format(self.frame_num))
    #print(type(self.frame))
    #assert(ret==True)
    #    frame_num   0---->frames-1
    def main_loop(self):
        key = cv2.waitKeyEx(1)
        if key in keybindings['first_frame']:
            self.seek(0)

        elif key in keybindings['last_frame']:
            self.seek(self.frames - 1)

        elif key in keybindings['next']:
            if self.frame_num < self.frames - 1:
                self.seek(self.frame_num + self.step)

        elif key in keybindings['prev']:
            time.sleep(0.01)
            if self.frame_num > 0:
                self.seek(self.frame_num - self.step)


        elif key in keybindings['forward_frames']:
            if self.frame_num < self.frames - 1:
                self.seek(self.frame_num + self.jump)

        
        elif key in keybindings['backward_frames']:
            self.seek(self.frame_num - self.jump)


        elif key in keybindings['circle_grow']:
            self.circle_size += 1
        elif key in keybindings['circle_shrink']:
            self.circle_size -= 1


        elif key in keybindings['mark_hidden']:
            self.mark_hidden()

        elif key in keybindings['piece_start']:
            self.piece_start = self.frame_num

        elif key in keybindings['piece_end']:
            self.piece_end = self.frame_num
            self.save_piece()


        elif key in keybindings['quit']:
            self.finish()
            return


        self.display()


    def finish(self):
        if getattr(self, '_finished', True):
            return
        self._finished = True
        if getattr(self, 'cap', None) is not None:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if not hasattr(self, 'info'):
            return
        df = pd.DataFrame.from_dict(self.info).sort_values(by=['frame_num'], ignore_index=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_path, index=False)
        print(f"Saved labels to {self.csv_path}")


    def __del__(self):
        if hasattr(self, 'info'):
            self.finish()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, nargs='?', default=None, help='Path to the video file.')
    parser.add_argument('--csv_dir', type=str, default=None, help='Path to the directory where csv file should be saved. If not specified, csv file will be saved in the same directory as the video file.')
    parser.add_argument('--step', type=int, default=1, help='Frames to jump with n/p. Left-click interpolates between keyframes. Example: --step 3')
    parser.add_argument('--remove_duplicate_frames', type=bool, default=False, help='Should identical consecutie frames be reduces to one frame.')
    opt = parser.parse_args()
    return opt


def remove_duplicate_frames(video_path, output_path):
    # Open the video file
    vid = cv2.VideoCapture(video_path)

    # Set the frame width and height
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object for the output video file
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Read and process the frames one by one
    previous_frame = None
    while True:
        # Read the next frame
        success, frame = vid.read()

        # If we reached the end of the video, break the loop
        if not success:
            break

        # If the current frame is not a duplicate, write it to the output video
        if previous_frame is None or cv2.PSNR(frame, previous_frame) < 40.:
            out.write(frame)

        # Update the previous frame
        previous_frame = frame
    print('finished removing duplicates')



if __name__ == '__main__':
    opt = parse_opt()

    if opt.video_path is None:
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        elif __file__:
            application_path = os.path.dirname(__file__)
        p  = Path(application_path)
        video_path = next(p.glob('*.mp4'))
        toRemove = input('Should duplicated, consecutive frames be deleted? Insert "y" or "n": \n')
        if toRemove == 'y':
            bez_duplikatow_video_path = str(video_path.with_stem(video_path.stem + '_no_dups'))
            remove_duplicate_frames(str(video_path), bez_duplikatow_video_path)
            video_path = bez_duplikatow_video_path
        opt.video_path = str(video_path)

    # run as a CLI script
    elif opt.remove_duplicate_frames == True:
        remove_duplicate_frames(opt.video_path, opt.video_path)

    player = VideoPlayer(opt)
    while(player.cap.isOpened()):
        player.main_loop()
