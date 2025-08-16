class CNNForceSolver:
    """
    Placeholder for CNN-based normal/tangential force regression.
    """

    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.model = None  # TODO: load your network

    def infer(self, image_rgb):
        # TODO: return force maps
        raise NotImplementedError("CNNForceSolver is a stub.")
