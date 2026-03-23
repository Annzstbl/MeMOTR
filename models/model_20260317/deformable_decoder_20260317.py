from ..model_20260310.deformable_decoder_20260310 import DeformableDecoder as DeformableDecoder20260310
from ..model_20260310.deformable_decoder_20260310 import DeformableDecoderLayer as DeformableDecoderLayer20260310


class DeformableDecoder20260317(DeformableDecoder20260310):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DeformableDecoderLayer20260317(DeformableDecoderLayer20260310):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


DeformableDecoder = DeformableDecoder20260317
DeformableDecoderLayer = DeformableDecoderLayer20260317
