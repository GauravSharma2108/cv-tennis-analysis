import torch
import torchvision.transforms as transforms
import cv2
import torchvision.models as models

class CourtLineDetector:
    """
    Class to detect keypoints on the court line. The model is a ResNet50 model with the last layer replaced with a linear layer with 28 output features (14 keypoints with x and y coordinates each).
    """
    def __init__(self, model_path):
        """Initializes the model and the transformation to be applied to the image before passing it through the model.

        Args:
            model_path (str): Path to the model file
        """
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self,image):
        """Predict keypoints on the image. The image is first converted to RGB, transformed and then passed through the model. The keypoints are then converted to original image size.

        Args:
            image: Image on which to predict keypoints

        Returns:
            keypoints: List of keypoints
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0)   # will convert the image to list of images with only 1 image

        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        keypoints = outputs.squeeze().cpu().numpy()

        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w/224.0
        keypoints[1::2] *= original_h/224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        """Draw keypoints on the image

        Args:
            image: Image on which to draw the keypoints
            keypoints: List of keypoints

        Returns:
            image: Image with keypoints drawn on it
        """
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        """Draw keypoints on the video frames. Calls draw_keypoints on each frame.

        Args:
            video_frames: List of video frames
            keypoints: List of keypoints

        Returns:
            output_frames: List of video frames with keypoints drawn on them
        """
        output_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_frames.append(frame)
        return output_frames