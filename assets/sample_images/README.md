# Sample images

Notebook 3 and the figure script do not need any files here: they generate a
synthetic sample image with `tokenverse.vision.image_utils.make_sample_image()`.

To try your own image, drop a small JPEG or PNG in this folder (or upload one
to the Colab runtime) and load it with:

```python
from tokenverse.vision.image_utils import load_image_tensor

x = load_image_tensor("assets/sample_images/your_image.jpg", size=224)  # [3, 224, 224]
```

Keep committed images small and make sure you have the right to redistribute
them.
