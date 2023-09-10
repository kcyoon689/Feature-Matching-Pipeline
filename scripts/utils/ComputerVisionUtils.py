import kornia as K


class ComputerVisionUtils:
    def convert_cv_image_to_torch_image(cv_image):
        torch_img = K.image_to_tensor(cv_image, False).float() / 255.
        torch_img = K.color.bgr_to_rgb(torch_img)
        return torch_img
