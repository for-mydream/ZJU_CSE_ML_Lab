import torch
import numpy as np
 
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt 
from torchvision.transforms import transforms
import cv2
try:
    from MTCNN.detector import FaceDetector
    from MobileNetV2 import MobileNetV2
except:
    from .MTCNN.detector import FaceDetector
    from .MobileNetV2 import MobileNetV2
def plot_image(image, image_title="", is_axis=False):
    plt.imshow(image)
    if not is_axis:
        plt.axis('off')
        plt.title(image_title)
        plt.show()
class Recognition(object):
    classes = ["mask", "no_mask"]
 
    def __init__(self, model_path=None):
        self.detector = FaceDetector()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mobilenet = MobileNetV2(num_class=2)
        if model_path:
            self.mobilenet.load_state_dict(
                torch.load(model_path, map_location=device))
    def face_recognize(self, image):
        drawn_image = self.detector.draw_bboxes(image)
        return drawn_image
    def mask_recognize(self, image):
        b_boxes, landmarks = self.detector.detect(image)
        detect_face_img = self.detector.draw_bboxes(image)
        face_num = len(b_boxes)
        mask_num = 0
        for box in b_boxes:
            face = image.crop(tuple(box[:4]))
            face = np.array(face)
            # reshape size of the image to fit the model input request
            face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
            face = transforms.ToTensor()(face).unsqueeze(0)
            self.mobilenet.eval()
 
            with torch.no_grad():
                predict_label = self.mobilenet(face).cpu().data.numpy()
            current_class = self.classes[np.argmax(predict_label).item()]
            draw = ImageDraw.Draw(detect_face_img)
            if current_class == "mask":
                mask_num += 1
                draw.text((200, 50), u'yes', 'fuchsia')
            else:
                draw.text((200, 50), u'no', 'fuchsia')
        return detect_face_img, face_num, mask_num

