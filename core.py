import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import win32api
import win32con
import win32gui
import win32print
from pynput import mouse
from ultralytics import YOLO


class r5CVCore:
    def __init__(self, args, config, camera) -> None:
        self.args = args
        self.config = config
        self.camera = camera
        self.logger = logging.getLogger("r5CV")

        self.model = None
        self.trt_model = None

        # initialize the model
        self.logger.info(f"Loading the model: {self.args.model_name}")
        if self.args.model_name.endswith(".engine"):
            self.load_TRT_model()
        elif self.args.model_name.endswith(".pt"):
            self.load_model()

        # set variables
        self.screen_size = np.array([
            win32api.GetSystemMetrics(0),
            win32api.GetSystemMetrics(1)
        ])
        self.screen_center = self.screen_size // 2
        self.destination = self.screen_center
        self.last_destination = self.destination

        self.width = 0
        # these three variables cannot be changed by the user
        self.auto_fire = False
        # self.left_lock = False  # lock on target when the left mouse button is pressed
        # self.right_lock = False  # lock when pressing the right mouse button (scoping)

        self.fired_time = time.time()

        # scale: the ratio of the screen resolution to the default resolution (96 dpi)
        self.scale = win32print.GetDeviceCaps(win32gui.GetDC(0), win32con.LOGPIXELSX) / 96
        self.mouse_position = np.array(mouse.Controller().position)
        self.mouse_vector = np.array([0., 0.])

        # PID control
        self.pre_error = np.array([0., 0.])
        # self.error = np.array([0., 0.])
        self.integral = np.array([0., 0.])
        self.backforce = 0
        self.aim_fov = 4 / 3

        # debug
        self.exec_count = 0
        self.capture_time = 0
        self.inference_time = 0

    def load_model(self) -> None:
        model_path = os.path.join(self.args.model_dir, self.args.model_name)
        self.model = YOLO(model_path)

    def load_TRT_model(self) -> None:
        engine_path = os.path.join(self.args.model_dir, self.args.model_name)
        self.trt_model = TensorRTEngine(engine_path)

    def predict(self, image) -> np.ndarray:
        results = self.model(
            image,
            verbose=self.args.verbose,
            half=True,
            iou=self.config["iou_threshold"],
            conf=self.config["conf_threshold"],
        )
        return results[0]

    def predict_TRT(self, image) -> tuple:
        boxes, scores, cls_indices = self.trt_model.inference(
            image,
            iou=self.config["iou_threshold"],
            conf=self.config["conf_threshold"],
            classes=[1, 2]  # 1: Ally, 2: Enemy
        )
        return boxes, scores, cls_indices

    def calc_mouse_redirection(self, boxes) -> None:
        num_boxes = boxes.shape[0]
        if num_boxes == 0:  # no target detected
            self.width = -1
            self.last_destination = self.destination  # save the last destination
            self.destination = np.array([-1, -1])
            self.logger.debug(f"Detection: {0:2d}, Destination: {self.destination}")
        else:
            self.mouse_position = np.array(mouse.Controller().position)

            boxes_center = ((boxes[:, :2] + boxes[:, 2:]) / 2)
            boxes_center[:, 1] = (
                # boxes[:, 1] * 0.6 + boxes[:, 3] * 0.4  # torso
                boxes[:, 1] * 0.7 + boxes[:, 3] * 0.3  # chest
                # boxes[:, 1] * 0.85 + boxes[:, 3] * 0.15
            )

            crop_size = 540 / win32api.GetSystemMetrics(1)
            # map the box from the image coordinate to the screen coordinate
            start_point = self.screen_center - self.screen_size[1] * crop_size / 2
            start_point = list(map(int, start_point))
            boxes_center[:, 0] = boxes_center[:, 0] + start_point[0]
            boxes_center[:, 1] = boxes_center[:, 1] + start_point[1]

            # find the nearest box center
            distance = np.linalg.norm(boxes_center - self.mouse_position, axis=-1)
            min_index = np.argmin(distance)
            self.width = boxes[min_index, 2] - boxes[min_index, 0]
            self.last_destination = self.destination
            self.destination = boxes_center[np.argmin(distance)].astype(int)

            self.logger.debug(f"Detection: {num_boxes:2d}, Destination: {self.destination}")

    def pid_control(self, error) -> np.ndarray:
        self.integral += error
        derivative = error - self.pre_error
        self.pre_error = error

        output = self.config["Kp"] * error + self.config["Ki"] * self.integral + self.config["Kd"] * derivative
        output[1] += self.backforce
        return output.astype(int)

    def move_mouse(self, key_state, mouse_state) -> None:
        hold_state, toggle_state_1, toggle_state_2, _ = key_state
        mouse_left_state, _, mouse_right_toggle_state = mouse_state

        detecting = hold_state or toggle_state_1
        if toggle_state_2:
            if mouse_right_toggle_state:
                detecting = True
        # print("detecting: ", detecting)

        if not detecting:
            self.pre_error = np.array([0., 0.])
            self.integral = np.array([0., 0.])
            return

        self.logger.debug("Detecting: True")

        if self.destination[0] == -1:
            if self.last_destination[0] == -1:
                self.pre_error = np.array([0., 0.])
                self.integral = np.array([0., 0.])
                return
            else:
                self.mouse_vector = np.array([0, 0])
        else:
            self.mouse_vector = (self.destination - self.mouse_position) / self.scale

        norm = np.linalg.norm(self.mouse_vector)
        if norm > self.width * self.aim_fov:
            return

        # use proportional–integral–derivative control
        if self.config["use_pid"]:
            move = self.pid_control(self.mouse_vector)
            win32api.mouse_event(
                win32con.MOUSEEVENTF_MOVE,
                int(move[0]),
                int(move[1] / 2)
            )
            last_move = self.last_destination - self.mouse_position + self.mouse_vector
            if not self.auto_fire or time.time() - self.fired_time <= 0.125:
                return  # 125ms
            # norm <= width / 2  # higher divisor increases precision but limits fire rate
            # move[0] * last_mv[0] >= 0  # ensures tracking

            # scope fire
            if (hold_state and not toggle_state_2 and mouse_right_toggle_state and not mouse_left_state and norm <= self.width * 2 / 3 and move[0] * last_move[0] >= 0):
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                self.fired_time = time.time()
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

            # hip fire
            elif (((hold_state and not mouse_right_toggle_state) or (toggle_state_2 and mouse_right_toggle_state and not mouse_left_state)) and norm <= self.width * 3 / 4 and move[0] * last_move[0] >= 0):
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                self.fired_time = time.time()
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
            return

        # if destination is close to the center, do not move the mouse
        if norm <= 2 or (self.destination[0] == self.screen_center[0] and self.destination[1] == self.screen_center[1]):
            return

        # if the destination is close to the center, move the mouse slowly   TODO: Check this part
        if norm <= self.width * 4 / 3:
            win32api.mouse.event(
                win32con.MOUSEEVENTF_MOVE,
                int(self.mouse_vector[0] / 3),
                int(self.mouse_vector[1] / 3)
            )
            return

    def execute(self, key_state, mouse_state) -> None:
        self.exec_count += 1

        start_time = time.time()
        captured_image = self.camera.capture()
        self.capture_time += time.time() - start_time

        start_time = time.time()
        if self.args.model_name.endswith(".engine"):
            boxes, scores, classes = self.predict_TRT(captured_image)

        elif self.args.model_name.endswith(".pt"):
            boxes = self.predict(captured_image).boxes
            boxes = boxes[boxes[:].cls == 1].cpu().xyxy.numpy()  # 0: "Ally", 1: "Enemy", 2: "Tag"
        elif self.args.model_name.endswith("omnx."):
            # .omnx file must be converted to .engine file
            self.logger.error("ONNX model is not supported. Please convert it to TensorRT engine format.")
            sys.exit()
        else:
            self.logger.error("Invalid model format. Shutting down.")
            sys.exit()

        for i in range(0, boxes.shape[0]):
            break  # NOTE: As of now, nothing is shown on the screen
            self.show_target([
                int(boxes[i, 0]) + self.camera.x_offset,
                int(boxes[i, 1]) + self.camera.y_offset,
                int(boxes[i, 2]) + self.camera.x_offset,
                int(boxes[i, 3]) + self.camera.y_offset
            ])

        self.calc_mouse_redirection(boxes)
        self.move_mouse(key_state, mouse_state)

        self.inference_time += time.time() - start_time

        if self.exec_count % 1000 == 0:
            self.logger.info(f"Capture: {self.capture_time:.2f}ms, Inference: {self.inference_time:.2f}ms")
            self.capture_time = 0

    def show_target(self, box) -> None:  # FIXME: This function is not working
        hwnd = win32gui.GetDesktopWindow()
        hwndDC = win32gui.GetDC(hwnd)
        pen = win32gui.CreatePen(win32con.PS_SOLID, 3, win32api.RGB(255, 0, 255))
        brush = win32gui.GetStockObject(win32con.NULL_BRUSH)

        win32gui.SelectObject(hwndDC, pen)
        win32gui.SelectObject(hwndDC, brush)
        win32gui.Rectangle(hwndDC, box[0], box[1], box[2], box[3])

        win32gui.ReleaseDC(hwnd, hwndDC)
        return


class TensorRTEngine(object):
    def __init__(self, engine_path) -> None:
        with open(f"{os.path.realpath(os.path.dirname(__file__))}/class_names.json", "r") as file:
            self.cls_names = json.load(file)

        self.n_classes = 80

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, '')  # initialize TensorRT plugins
        with open(engine_path, 'rb') as f:
            if engine_path[-7:] == '.engine':
                meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
                metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata  # noqa: F841
            engine = runtime.deserialize_cuda_engine(f.read())  # read engine
            # serialized_engine = f.read()
        # engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get imgsz
        self.n_classes = engine.get_binding_shape(1)[1] - 4  # get n_classes from "(x1 y1 x2 y2) c1 c2 c3 cn"
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def inference(self, img, iou=0.45, conf=0.25, classes=[], end2end=False) -> tuple:
        cuda_img, ratio = TensorRTEngine.preprocess(img, self.imgsz)
        self.inputs[0]['host'] = np.ravel(cuda_img)
        # copy inputs to GPU
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # copy outputs from GPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        data = [out['host'] for out in self.outputs]

        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
            dets = np.concatenate([
                final_boxes[:num[0]],
                np.array(final_scores)[:num[0]].reshape(-1, 1),
                np.array(final_cls_inds)[:num[0]].reshape(-1, 1)
            ], axis=-1)
        else:
            # predictions = np.reshape(data, (1, 8400, -1), order="F")[0]
            predictions = np.reshape(data, (1, -1, int(4 + self.n_classes)), order="F")[0]
            dets = self.postprocess(predictions, ratio, iou, conf)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]  # final_boxes=[0,1,2,3] final_scores=[4] final_cls_inds=[5]
            score_mask = final_scores > conf
            if len(classes) > 0:
                class_mask = np.isin(final_cls_inds, classes)
                mask = np.logical_and(score_mask, class_mask)
            else:
                mask = score_mask
            boxes = final_boxes[mask]
            scores = final_scores[mask]
            cls_inds = final_cls_inds[mask]
        else:
            boxes = np.empty((0, 4))
            scores = np.empty((0, self.n_classes))
            cls_inds = np.empty((0, 1))
        return boxes, scores, cls_inds

    @staticmethod
    def postprocess(predictions, ratio, iou_thr, conf_thr) -> np.ndarray:
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        # dets = multiclass_nms(boxes_xyxy, scores, iou_thr=0.45, conf_thr=0.25)
        dets = TensorRTEngine.multiclass_nms(boxes_xyxy, scores, iou_thr, conf_thr)
        return dets

    @staticmethod
    def nms(boxes, scores, iou_thr) -> list:
        """Single class NMS implemented in NumPy"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]  # get boxes with more conf first

        keep = []
        while order.size > 0:
            i = order[0]  # pick maximum conf box
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)  # maximum width
            h = np.maximum(0, yy2 - yy1 + 1)  # maximum height
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_thr)[0]
            order = order[inds + 1]

        return keep

    @staticmethod
    def multiclass_nms(boxes, scores, iou_thr, conf_thr) -> np.ndarray:
        """Multiclass NMS implemented in NumPy"""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > conf_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_boxes = boxes[valid_score_mask]
                valid_scores = cls_scores[valid_score_mask]
                keep = TensorRTEngine.nms(valid_boxes, valid_scores, iou_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    @staticmethod
    def preprocess(image, input_size) -> tuple:  # imgsz, size of input image as integer [W, H]
        # padded_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)  # HWC (640x640x3)
        # ratio = min(input_size[0]/image.shape[1], input_size[1]/image.shape[0])  # ratio=min, (padded_img) strech the least or shrink the most
        ratio = max(input_size[0] / image.shape[1], input_size[1] / image.shape[0])  # ratio=max, (cropped_img) strech the most or shrink the least
        resized_img = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR)  # cv2.resize(image, (width, height), interpolation)
        h, w = resized_img.shape[:2]
        min_size = min(h, w)
        cropped_img = resized_img[
            int(h / 2 - min_size / 2):int(h / 2 + min_size / 2),
            int(w / 2 - min_size / 2):int(w / 2 + min_size / 2)
        ]
        cropped_img = cropped_img[:, :, ::-1].transpose(2, 0, 1)  # convert BGR to RGB, permute HWC to CHW (3x640x640)
        cropped_img = np.ascontiguousarray(cropped_img, dtype=np.float32)  # uint8 to float32
        cropped_img /= 255  # convert uint8 0-255 to float32 0.0-1.0
        return cropped_img, ratio
