class ImageList(list):
    @property
    def shape(self):
        if len(self) > 0:
            return self[0].shape
        else:
            return None