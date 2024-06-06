"""
@Author: Conghao Wong
@Date: 2023-07-12 17:38:42
@LastEditors: Conghao Wong
@LastEditTime: 2024-06-05 19:24:21
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os
import sys
import tkinter as tk
from copy import copy, deepcopy
from tkinter import filedialog
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageTk
from utils import TK_BORDER_WIDTH, TK_TITLE_STYLE, TextboxHandler

sys.path.insert(0, os.path.abspath('.'))

import qpid
import socialCircle
from main import main
from qpid.args import TEMPORARY, EmptyArgs
from qpid.constant import DATASET_CONFIGS, INPUT_TYPES
from qpid.dataset.agent_based import Agent
from qpid.mods import segMaps, vis
from qpid.utils import dir_check, get_mask, get_relative_path, move_to_device

OBS = INPUT_TYPES.OBSERVED_TRAJ
NEI = INPUT_TYPES.NEIGHBOR_TRAJ
SEG = segMaps.INPUT_TYPES.SEG_MAP

DATASET = 'ETH-UCY'
SPLIT = 'zara1'
CLIP = 'zara1'
MODEL_PATH = 'static'

TEMP_IMG_PATH = './temp_files/socialcircle_toy_example/fig.png'
TEMP_SEG_MAP_PATH = './temp_files/socialcircle_toy_example/seg.png'
TEMP_RGB_IMG_PATH = './temp_files/socialcircle_toy_example/fig_rgb.png'
LOG_PATH = './temp_files/socialcircle_toy_example/run.log'

DRAW_MODE_PLT = 'PLT'
DRAW_MODE_QPID = 'Interactive (SC)'
DRAW_MODE_QPID_PHYSICAL = 'Interactive (PC)'
DRAW_MODES_ALL = [DRAW_MODE_QPID, DRAW_MODE_PLT, DRAW_MODE_QPID_PHYSICAL]

OBSTACLE_IMAGE_PATH = get_relative_path(__file__, 'mask.png')

MAX_HEIGHT = 480
MAX_WIDTH = 640

MARKER_CIRCLE_RADIUS = 3
MARKER_RADIUS = 5
MARKER_TAG = 'indicator'

SEG_MAP_R = 0xff
SEG_MAP_G = 0xa3
SEG_MAP_B = 0x7f

dir_check(os.path.dirname(LOG_PATH))


class ToyArgs(EmptyArgs):

    @property
    def draw_seg_map(self) -> int:
        """
        Choose whether to draw segmentation maps on the canvas.
        """
        return self._arg('draw_seg_map', 1, TEMPORARY)
    
    @property
    def points(self) -> int:
        """
        The number of points to simulate the trajectory of manual
        neighbor. It only accepts `2` or `3`.
        """
        return self._arg('points', 2, TEMPORARY)
    
    @property
    def lite(self) -> int:
        """
        Choose whether to show the lite version of tk window.
        """
        return self._arg('lite', 0, TEMPORARY)
    

qpid.register_args(ToyArgs, 'Toy Example Args')


class SocialCircleToy():
    def __init__(self, args: list[str]) -> None:
        # Manager objects
        self.t: qpid.training.Structure | None = None
        self.image: tk.PhotoImage | None = None
        self.image_shape = None
        self.image_vis_scale = None
        self.mask_image: ImageTk.PhotoImage | None = None
        self.vis_mgr: vis.Visualization | None = None

        # Data containers
        self.inputs: list[torch.Tensor] | None = None
        self.outputs: list[torch.Tensor] | None = None
        self._agents: list = []
        self.input_and_gt: list[list[torch.Tensor]] | None = None
        self.input_types = None

        # Args
        self.args = ToyArgs(sys.argv)

        # Settings
        self.draw_mode_count = 0
        self.click_count = 0
        self.marker_count: int | None = None

        # Variables
        self.image_scale = 1.0
        self.image_margin = [0.0, 0.0]

        # Try to load models from the init args
        self.load_model(args)

        # The maximum number of manual agents
        self.interp_model = None

        # TK variables
        self.tk_vars: dict[str, tk.StringVar] = {}
        self.tk_vars['agent_id'] = tk.StringVar(value='0')
        for p in range(self.args.points):
            for i in ['x', 'y']:
                self.tk_vars[f'p{i}{p}'] = tk.StringVar()

    @property
    def draw_mode(self) -> str:
        return DRAW_MODES_ALL[self.draw_mode_count]

    @property
    def agents(self):
        if self.t:
            agents = self.t.agent_manager.agents
            if len(agents):
                pass
            elif len(self._agents):
                agents = self._agents
            else:
                raise ValueError('No Agent Data!')
        else:
            raise ValueError(self.t)
        return agents

    def init_model(self):
        """
        Init models and managers, then load all needed data.
        """
        if not self.t:
            raise ValueError('Structure Not Initialized!')

        # Create model(s)
        self.t.create_model()
        if not INPUT_TYPES.NEIGHBOR_TRAJ in self.t.model.input_types:
            self.t.model.input_types.append(INPUT_TYPES.NEIGHBOR_TRAJ)

        old_input_types = self.input_types
        self.input_types = (self.t.model.input_types,
                            self.t.args.obs_frames,
                            self.t.args.pred_frames)
        self.t.agent_manager.set_types(self.t.model.input_types,
                                       self.t.model.label_types)

        # Load dataset files
        if ((self.input_and_gt is None) or
                (self.input_types != old_input_types)):
            self.t.log('Reloading dataset files...')
            ds = self.t.agent_manager.clean().make(self.t.args.force_clip, training=False)
            self._agents = self.t.agent_manager.agents
            self.input_and_gt = list(ds)[0]

        # Create vis manager
        if not self.vis_mgr:
            self.vis_mgr = vis.Visualization(manager=self.t,
                                             dataset=self.t.args.force_dataset,
                                             clip=self.t.args.force_clip)

    def load_model(self, args: list[str]):
        """
        Create new models and training structures from the given args.
        """
        try:
            t = main(args, run_train_or_test=False)
            self.t = t
            self.t.args._set_default('force_dataset', DATASET)
            self.t.args._set_default('force_split', SPLIT)
            self.t.args._set_default('force_clip', CLIP)
            self.init_model()
            self.t.log(f'Model `{t.model.name}` and dataset ({CLIP}) loaded.')
        except Exception as e:
            print(e)

    def get_input_index(self, input_type: str):
        if not self.t:
            raise ValueError
        return self.t.model.input_types.index(input_type)

    def run_on_agent(self, agent_index: int,
                     extra_neighbor_position=None):

        if not self.input_and_gt:
            raise ValueError

        inputs = self.input_and_gt[0]
        inputs = [i[agent_index][None] for i in inputs]

        if (p := extra_neighbor_position) is not None:
            if self.draw_mode in [DRAW_MODE_PLT, DRAW_MODE_QPID]:
                nei = self.add_one_neighbor(inputs, p)
                inputs[self.get_input_index(NEI)] = nei

            elif self.draw_mode == DRAW_MODE_QPID_PHYSICAL:
                try:
                    seg_map = inputs[self.get_input_index(SEG)]
                    seg_map = self.add_obstacle(seg_map)
                    inputs[self.get_input_index(SEG)] = seg_map
                except ValueError:
                    pass

        self.forward(inputs)

        # Draw results on images
        m = self.draw_mode
        if m in [DRAW_MODE_QPID, DRAW_MODE_QPID_PHYSICAL]:
            self.draw_results(agent_index, draw_with_plt=False,
                              image_save_path=TEMP_RGB_IMG_PATH,
                              resize_image=True)
        elif m == DRAW_MODE_PLT:
            self.draw_results(agent_index, draw_with_plt=True,
                              image_save_path=TEMP_IMG_PATH,
                              resize_image=False)
        else:
            raise ValueError(m)

    def get_neighbor_count(self, neighbor_obs: torch.Tensor):
        '''
        Input's shape should be `(1, max_agents, obs, dim)`.
        '''
        nei = neighbor_obs[0]

        if issubclass(type(nei), np.ndarray):
            nei = torch.from_numpy(nei)

        nei_mask = get_mask(torch.sum(nei, dim=[-1, -2]))
        return int(torch.sum(nei_mask))

    def add_one_neighbor(self, inputs: list[torch.Tensor],
                         position: list[tuple[float, float]]):
        '''
        Shape of `nei` should be `(1, max_agents, obs, 2)`
        '''
        obs = inputs[self.get_input_index(OBS)]
        nei = inputs[self.get_input_index(NEI)]

        nei = copy(nei.numpy())
        steps = nei.shape[-2]

        if len(position) == 2:
            xp = np.array([0, steps-1])
            fp = np.array(position)
            x = np.arange(steps)
            traj = np.column_stack([np.interp(x, xp, fp[:, 0]),
                                    np.interp(x, xp, fp[:, 1])])

        elif len(position) == 3:
            xp = np.array([0, steps//2, steps-1])
            fp = np.array(position)
            x = np.arange(steps)

            from qpid.model.layers.interpolation import \
                LinearSpeedInterpolation
            if self.interp_model is None:
                self.interp_model = LinearSpeedInterpolation()

            traj = self.interp_model.forward(
                index=torch.tensor(xp),
                value=torch.tensor(fp),
                init_speed=torch.tensor((fp[2:] - fp[:1])/steps)
            ).numpy()
            traj = np.concatenate([fp[:1], traj], axis=0)

        else:
            raise ValueError(len(position))

        nei_count = self.get_neighbor_count(nei)
        nei[0, nei_count] = traj - obs.numpy()[0, -1:, :]
        return torch.from_numpy(nei)

    def add_obstacle(self, seg_map: torch.Tensor):
        """
        Add a rectangle obstacle to the segmentation map.

        :param seg_map: Seg map, shape = (..., 100, 100).
        """
        if not self.t:
            raise ValueError

        from qpid.mods.segMaps.settings import NORMALIZED_SIZE

        r = []
        for _i in ['0', '1']:
            for _j in ['px', 'py']:
                r.append(NORMALIZED_SIZE *
                         int(float(self.tk_vars[_j + _i].get())))

        if not self.image_shape:
            seg_img_path = self.t.agent_manager.split_manager.\
                clips_dict[self.t.args.force_clip].\
                other_files[DATASET_CONFIGS.RGB_IMG]
            img = Image.open(seg_img_path)
            self.image_shape = img.size
            img.close()

        if not self.image_vis_scale:
            s = self.t.agent_manager.split_manager.scale_vis
            self.image_vis_scale = s

        xs = [int(self.image_vis_scale * r[0]/self.image_shape[1]),
              int(self.image_vis_scale * r[2]/self.image_shape[1])]
        ys = [int(self.image_vis_scale * r[1]/self.image_shape[0]),
              int(self.image_vis_scale * r[3]/self.image_shape[0])]

        xs.sort()
        ys.sort()

        new_map = deepcopy(seg_map)
        new_map[..., xs[0]:xs[1], ys[0]:ys[1]] = 1.0
        return new_map

    def forward(self, inputs: list[torch.Tensor]):
        if not self.t:
            raise ValueError

        self.inputs = inputs
        with torch.no_grad():
            self.outputs = self.t.model.implement(inputs, training=False)
        self.outputs = move_to_device(self.outputs, self.t.device_cpu)

    def switch_draw_mode(self, label: tk.Label | None = None):
        self.draw_mode_count += 1
        self.draw_mode_count %= len(DRAW_MODES_ALL)

        if label:
            label.config(text=f'Mode: {self.draw_mode}')

    def set_a_random_agent_id(self):
        try:
            n = len(self.agents)
            if self.t:
                n = min(n, self.t.args.batch_size)
            i = np.random.randint(0, n)
            self.tk_vars['agent_id'].set(str(i))
        except:
            pass

    def draw_results(self, agent_index: int,
                     draw_with_plt: bool,
                     image_save_path: str,
                     resize_image=False):

        if ((not self.inputs) or
            (not self.outputs) or
            (not self.vis_mgr) or
                (not self.t)):
            raise ValueError

        # Write predicted trajectories and new neighbors to the agent
        agent = Agent().load_data(
            deepcopy(self.agents[agent_index].zip_data()))
        agent.manager = self.t.agent_manager

        agent.write_pred(self.outputs[0].numpy()[0])
        agent.traj_neighbor = self.inputs[self.get_input_index(NEI)][0].numpy()
        agent.neighbor_number = self.get_neighbor_count(
            agent.traj_neighbor[None])

        self.vis_mgr.draw(agent=agent,
                          frames=[agent.frames[self.t.args.obs_frames-1]],
                          save_name=image_save_path,
                          save_name_with_frame=False,
                          save_as_images=True,
                          draw_with_plt=draw_with_plt)
        del agent

        # Resize the image
        if resize_image:
            import cv2
            f = cv2.imread(image_save_path)
            h, w = f.shape[:2]
            if ((h >= MAX_HEIGHT) and (h/w >= MAX_HEIGHT/MAX_WIDTH)):
                self.image_scale = h / MAX_HEIGHT
                self.image_margin = [0, (MAX_WIDTH - w/self.image_scale)//2]
            elif ((w >= MAX_WIDTH) and (h/w <= MAX_HEIGHT/MAX_WIDTH)):
                self.image_scale = w / MAX_WIDTH
                self.image_margin = [(MAX_HEIGHT - h/self.image_scale)//2, 0]
            else:
                raise ValueError

            f = cv2.resize(f, [int(w//self.image_scale),
                               int(h//self.image_scale)])
            _p = os.path.join(os.path.dirname(image_save_path),
                              'resized_' + os.path.basename(image_save_path))
            cv2.imwrite(_p, f)
            image_save_path = _p

        self.image = tk.PhotoImage(file=image_save_path)

    def draw_obstacle(self, canvas: tk.Canvas):
        # Get saved positions (image/pixel)
        res = []
        for _i in ['0', '1']:
            for _j in ['px', 'py']:
                _r = self.tk_vars[_j + _i].get()
                if not len(_r):
                    return

                res.append(float(_r))

        # Transform to canvas positions (canvas/pixel)
        x0_cp, y0_cp = self.image_pixel_to_canvas_pixel(*res[:2])
        x1_cp, y1_cp = self.image_pixel_to_canvas_pixel(*res[2:])

        _dx, _dy = (abs(int(x1_cp - x0_cp)), abs(int(y1_cp - y0_cp)))
        img = Image.open(OBSTACLE_IMAGE_PATH).resize((_dx, _dy))
        self.mask_image = ImageTk.PhotoImage(img)
        canvas.create_image(min(x0_cp, x1_cp) + _dx // 2,
                            min(y0_cp, y1_cp) + _dy // 2,
                            image=self.mask_image)

    def click(self, event: tk.Event, canvas: tk.Canvas):

        if ((not self.draw_mode in [DRAW_MODE_QPID,
                                    DRAW_MODE_QPID_PHYSICAL])
                or (not self.vis_mgr)):
            return

        x, y = [event.x, event.y]
        x_ip, y_ip = self.canvas_pixel_to_image_pixel(x, y)
        x_ir, y_ir = self.image_pixel_to_image_real(x_ip, y_ip)

        if self.click_count == 0:
            clear_indicator(canvas)
            draw_indicator(canvas, x, y, 'red', text='START')
            self.click_count = 1

            if self.draw_mode == DRAW_MODE_QPID:
                self.tk_vars['px0'].set(str(x_ir))
                self.tk_vars['py0'].set(str(y_ir))

            elif self.draw_mode == DRAW_MODE_QPID_PHYSICAL:
                self.tk_vars['px0'].set(str(x_ip))
                self.tk_vars['py0'].set(str(y_ip))

            else:
                pass

        elif self.click_count == 1:
            if self.args.points == 3 and self.draw_mode == DRAW_MODE_QPID:
                draw_indicator(canvas, x, y, 'orange', text='MIDDLE')
                self.click_count = 2
            else:
                draw_indicator(canvas, x, y, 'blue', text='END')
                self.click_count = 0

            if self.draw_mode == DRAW_MODE_QPID:
                self.tk_vars['px1'].set(str(x_ir))
                self.tk_vars['py1'].set(str(y_ir))

            elif self.draw_mode == DRAW_MODE_QPID_PHYSICAL:
                self.tk_vars['px1'].set(str(x_ip))
                self.tk_vars['py1'].set(str(y_ip))
                self.draw_obstacle(canvas)

            else:
                pass

        elif self.click_count == 2:
            draw_indicator(canvas, x, y, 'blue', text='END')
            self.click_count = 0

            self.tk_vars['px2'].set(str(x_ir))
            self.tk_vars['py2'].set(str(y_ir))

        else:
            raise ValueError

    def hover(self, event: tk.Event, canvas: tk.Canvas):
        """
        Draw a dot to the canvas when hovering on it.
        """
        if not self.draw_mode in [DRAW_MODE_QPID,
                                  DRAW_MODE_QPID_PHYSICAL]:
            return

        if self.marker_count is not None:
            canvas.delete(self.marker_count)

        self.marker_count = canvas.create_oval(event.x - MARKER_RADIUS,
                                               event.y - MARKER_RADIUS,
                                               event.x + MARKER_RADIUS,
                                               event.y + MARKER_RADIUS,
                                               fill='green')

    def run_prediction(self, with_manual_inputs: bool,
                       canvas: tk.Canvas,
                       social_circle: tk.Label,
                       nei_angles: tk.Label):

        if self.t is None:
            raise ValueError(self.t)

        # Check if the manual neighbor exists
        if (with_manual_inputs
            and len(x0 := self.tk_vars['px0'].get())
            and len(y0 := self.tk_vars['py0'].get())
            and len(x1 := self.tk_vars['px1'].get())
                and len(y1 := self.tk_vars['py1'].get())):
            extra_neighbor = [[float(x0), float(y0)],
                              [float(x1), float(y1)]]

            if (('px2' in self.tk_vars.keys())
                    and len(x2 := self.tk_vars['px2'].get())
                    and len(y2 := self.tk_vars['py2'].get())):
                extra_neighbor += [[float(x2), float(y2)]]

            self.t.log('Start running with an addition neighbor' +
                       f'from {extra_neighbor[0]} to {extra_neighbor[1]}...')

        else:
            extra_neighbor = None
            self.t.log('Start running without any manual inputs...')

        # Run the prediction model
        self.run_on_agent(int(self.tk_vars['agent_id'].get()),
                          extra_neighbor_position=extra_neighbor)

        # Show the visualized image
        if self.image:
            canvas.create_image(MAX_WIDTH//2, MAX_HEIGHT//2, image=self.image)

        # Draw segmentation map
        if self.draw_mode == DRAW_MODE_QPID_PHYSICAL:
            if ((self.args.draw_seg_map) and 
                (self.image) and
                (self.inputs is not None) and 
                (SEG in self.t.model.input_types)):

                seg_map = self.t.model.get_input(self.inputs, SEG)[0][..., None]
                seg_map_alpha = seg_map
                seg_map = torch.concat([SEG_MAP_R * seg_map, 
                                        SEG_MAP_G * seg_map, 
                                        SEG_MAP_B * seg_map,
                                        255 * 0.5 * seg_map_alpha], dim=-1)
                
                seg_map = Image.fromarray(seg_map.numpy().astype(np.uint8))
                seg_map = seg_map.resize((self.image.width(),
                                        self.image.height()))
                seg_map.save(TEMP_SEG_MAP_PATH)

                self.seg_map = ImageTk.PhotoImage(seg_map)
                canvas.create_image(MAX_WIDTH//2, MAX_HEIGHT//2, 
                                    image=self.seg_map)

            else:
                self.draw_obstacle(canvas)

        # Print model outputs
        time = int(1000 * self.t.model.inference_times[-1])
        self.t.log(f'Running done. Time cost = {time} ms.')

        # Set numpy format
        np.set_printoptions(formatter={'float': '{:0.3f}'.format})

        if (not self.outputs) or (not self.inputs):
            return

        # Print the SocialCircle
        try:
            sc = self.outputs[1][1].numpy()[0]
            social_circle.config(text=str(sc.T))
        except:
            pass

        # Print all neighbors' angles
        # count = self.get_neighbor_count(self.inputs[self.get_input_index(NEI)])
        # na = self.outputs[1][2].numpy()[0][:count]
        # nei_angles.config(text=str(na*180/np.pi))

    def clear_canvas(self, canvas: tk.Canvas):
        """
        Clear canvas when click refresh button
        """
        clear_indicator(canvas)
        self.tk_vars['px0'].set("")
        self.tk_vars['py0'].set("")
        self.tk_vars['px1'].set("")
        self.tk_vars['py1'].set("")

    def canvas_pixel_to_image_pixel(self, x: float, y: float) -> tuple[float, float]:
        return (self.image_scale * (y - self.image_margin[0]),
                self.image_scale * (x - self.image_margin[1]))

    def image_pixel_to_image_real(self, x: float, y: float) -> tuple[float, float]:
        if not self.vis_mgr:
            raise ValueError
        return self.vis_mgr.pixel2real(np.array([[x, y]]))[0]

    def image_pixel_to_canvas_pixel(self, x: float, y: float) -> tuple[float, float]:
        return (y / self.image_scale + self.image_margin[1],
                x / self.image_scale + self.image_margin[0])


def draw_indicator(canvas: tk.Canvas,
                   x: float, y: float,
                   color: str,
                   text: str | None = None):
    """
    Draw a circle indicator on the canvas.
    """
    if text:
        canvas.create_text(x - 2, y - 20 - 2, text=text,
                           tags=MARKER_TAG, anchor=tk.N, fill='black')
        canvas.create_text(x, y - 20, text=text,
                           tags=MARKER_TAG, anchor=tk.N, fill='white')

    canvas.create_oval(x - MARKER_CIRCLE_RADIUS,
                       y - MARKER_CIRCLE_RADIUS,
                       x + MARKER_CIRCLE_RADIUS,
                       y + MARKER_CIRCLE_RADIUS,
                       fill=color, tags=MARKER_TAG)


def clear_indicator(canvas: tk.Canvas):
    canvas.delete(MARKER_TAG)


if __name__ == '__main__':

    root = tk.Tk()
    root.title('Toy Example of SocialCircle Models')

    """
    Configs
    """
    # Left column
    l_args: dict[str, Any] = {
        # 'background': '#FFFFFF',
        'border': TK_BORDER_WIDTH,
    }

    # Right Column
    r_args: dict[str, Any] = {
        'background': '#FFFFFF',
        'border': TK_BORDER_WIDTH,
    }
    t_args: dict[str, Any] = {
        'foreground': '#000000',
    }

    # Button Frame
    b_args = {
        # 'background': '#FFFFFF',
        # 'border': TK_BORDER_WIDTH,
    }

    """
    Init base frames
    """
    (LF := tk.Frame(root, **l_args)).grid(
        row=0, column=0, sticky=tk.NW)
    (RF := tk.Frame(root, **r_args)).grid(
        row=0, column=1, sticky=tk.NW, rowspan=2)
    (BF := tk.Frame(root, **b_args)).grid(
        row=1, column=0, sticky=tk.N)

    """
    Init the log window
    """
    log_frame = tk.Frame(RF, **r_args)
    log_frame.grid(column=0, row=4, columnspan=2)

    logbar = tk.Text(log_frame, width=89, height=7, **r_args, **t_args)
    (scroll := tk.Scrollbar(log_frame, command=logbar.yview)).pack(
        side=tk.RIGHT, fill=tk.Y)
    logbar.config(yscrollcommand=scroll.set)
    logbar.pack()

    """
    Init the Training Structure
    """
    def args(path): return ['main.py',
                            '--sc', path,
                            '-bs', '4000',
                            '--test_mode', 'one',
                            '--draw_full_neighbors', '1'] + sys.argv

    qpid.set_log_path(LOG_PATH)
    qpid.set_log_stream_handler(TextboxHandler(logbar))
    qpid.add_arg_alias(['-sdd', '-SDD'],
                       ['--force_dataset', 'SDD',
                        '--force_split', 'sdd',
                        '--force_clip'])
    toy = SocialCircleToy(args(MODEL_PATH))

    """
    Init TK Components
    """
    # Left part
    i_l = -1

    if not toy.args.lite:
        tk.Label(LF, text='Settings', **TK_TITLE_STYLE, **l_args).grid(
            column=0, row=(i_l := i_l + 1), sticky=tk.W)

    tk.Label(LF, text='Agent ID', **l_args).grid(
        column=0, row=(i_l := i_l + 1))
    (id_frame := tk.Frame(LF, **l_args)).grid(
        column=0, row=(i_l := i_l + 1))

    tk.Entry(id_frame, textvariable=toy.tk_vars['agent_id'], width=10).grid(
        column=0, row=0)
    tk.Button(id_frame, text='Random', command=toy.set_a_random_agent_id).grid(
        column=1, row=0)

    tk.Label(LF, text='New Neighbor (x-axis, start)', **l_args).grid(
        column=0, row=(i_l := i_l + 1))
    tk.Entry(LF, textvariable=toy.tk_vars['px0']).grid(
        column=0, row=(i_l := i_l + 1))

    tk.Label(LF, text='New Neighbor (y-axis, start)', **l_args).grid(
        column=0, row=(i_l := i_l + 1))
    tk.Entry(LF,  textvariable=toy.tk_vars['py0']).grid(
        column=0, row=(i_l := i_l + 1))

    tk.Label(LF, text='New Neighbor (x-axis, end)', **l_args).grid(
        column=0, row=(i_l := i_l + 1))
    tk.Entry(LF, textvariable=toy.tk_vars['px1']).grid(
        column=0, row=(i_l := i_l + 1))

    tk.Label(LF, text='New Neighbor (y-axis, end)', **l_args).grid(
        column=0, row=(i_l := i_l + 1))
    tk.Entry(LF,  textvariable=toy.tk_vars['py1']).grid(
        column=0, row=(i_l := i_l + 1))

    # Right Part
    i_r = -1
    model_path = tk.Label(RF, width=60, wraplength=510,
                          text=MODEL_PATH, **r_args, **t_args)
    sc = tk.Label(RF, width=60, **r_args, **t_args)
    angles = tk.Label(RF, width=60, **r_args, **t_args)
    canvas = tk.Canvas(RF, width=MAX_WIDTH, height=MAX_HEIGHT, **r_args)

    if not toy.args.lite:
        tk.Label(RF, text='Predictions', **TK_TITLE_STYLE, **r_args, **t_args).grid(
            column=0, row=(i_r := i_r + 1), sticky=tk.W)

        tk.Label(RF, text='Model Path:', width=16, anchor=tk.E, **r_args, **t_args).grid(
            column=0, row=(i_r := i_r + 1))
        model_path.grid(column=1, row=i_r)

        tk.Label(RF, text='Social Circle:', width=16, anchor=tk.E, **r_args, **t_args).grid(
            column=0, row=(i_r := i_r + 1))
        sc.grid(column=1, row=i_r)

        # tk.Label(RF, text='Neighbor Angles:', width=16, anchor=tk.E, **r_args, **t_args).grid(
        #     column=0, row=(i_r := i_r + 1))
        # (angles := tk.Label(RF, width=60, **r_args, **t_args)).grid(
        #     column=1, row=i_r)

    canvas.grid(column=0, row=(i_r := i_r + 1), columnspan=2)
    canvas.bind("<Motion>", lambda e: toy.hover(e, canvas))
    canvas.bind("<Button-1>", lambda e: toy.click(e, canvas))

    tk.Button(BF, text='Run Prediction',
              command=lambda: toy.run_prediction(
                  True, canvas, sc, angles), **b_args).grid(
        column=0, row=10, sticky=tk.N)

    tk.Button(BF, text='Run Prediction (original)',
              command=lambda: toy.run_prediction(
                  False, canvas, sc, angles), **b_args).grid(
        column=0, row=11, sticky=tk.N)

    tk.Button(BF, text='Reload Model Weights',
              command=lambda: [toy.load_model(args(p := filedialog.askdirectory(initialdir='./'))),
                               model_path.config(text=p)]).grid(
        column=0, row=12, sticky=tk.N)

    tk.Button(BF, text='Clear Manual Inputs',
              command=lambda: toy.clear_canvas(canvas)).grid(
        column=0, row=13, sticky=tk.N)

    (mode_label := tk.Label(BF, text=f'Mode: {toy.draw_mode}', **l_args)).grid(
        column=0, row=15)

    tk.Button(BF, text='Switch Mode',
              command=lambda: toy.switch_draw_mode(mode_label)).grid(
        column=0, row=14, sticky=tk.N)

    root.mainloop()
