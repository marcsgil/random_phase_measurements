from ximea import xiapi
import numpy as np
from common.utils import calculate_centroid

def closest_divisible(n, m):
    return round(n / m) * m


class CustomXimea(xiapi.Camera):
    def __init__(self, dev_id=0):
        super().__init__(dev_id)
        self.open_device()
        self.img_instance = xiapi.Image()
        
        self.enable_vertical_flip()

        self.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
        self.set_framerate(60)

        self.start_acquisition()

    def capture(self):
        self.get_image(self.img_instance)
        return self.img_instance.get_image_data_numpy()

    def calibrate(self, N, image=None, threshold=0.5):
        if image is None:
            image = self.capture()
        centroid = calculate_centroid(image, threshold)

        self.set_width(N)
        self.set_height(N)

        self.set_offsetY(closest_divisible(centroid[0] - N // 2, self.get_offsetY_increment()))
        self.set_offsetX(closest_divisible(centroid[1] - N // 2, self.get_offsetX_increment()))

        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_acquisition()
        self.close_device()

if __name__ == "__main__":
    with CustomXimea() as camera:
        pass