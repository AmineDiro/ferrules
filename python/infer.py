import pypdfium2 as pdfium
from PIL import Image
from ultralytics import YOLO  # Or whichever library you're using for the YOLO model

# 1) Open the PDF document
pdf_path = "/Users/amine/Downloads/RAG Corporate 2024 016.pdf"
pdf_document = pdfium.PdfDocument(pdf_path)

# 2) Get the first page (index 0). You can adjust if you want a different page.
page_index = 0
page = pdf_document.get_page(page_index)

# 3) Render that page at a scale that maintains aspect ratio but fits into 1024x1024
page_width, page_height = page.get_size()
scale_w = 1024 / page_width
scale_h = 1024 / page_height
scale = min(scale_w, scale_h)

# 4) Render to a bitmap, convert to a PIL image, and resize to exactly 1024x1024
bitmap = page.render(scale=scale, rotation=0)
pil_image = bitmap.to_pil()
pil_image = pil_image.resize((1024, 1024), Image.Resampling.LANCZOS)

# 6) Load the ONNX model and run inference on the saved image
onnx_model = YOLO("yolov8s-doclaynet.onnx")
onnx_model.overrides["imgsz"] = 1024

result = onnx_model(pil_image)[0]

# 7) Retrieve the original height/width that YOLO sees
height = result.orig_shape[0]
width = result.orig_shape[1]

# 8) Collect bounding boxes and class labels
label_boxes = []
for label, box in zip(result.boxes.cls.tolist(), result.boxes.xyxyn.tolist()):
    # Convert normalized coords back to pixel space
    x1, y1, x2, y2 = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
    class_name = result.names[int(label)]
    label_boxes.append((class_name, [x1, y1, x2, y2]))

# 9) Print the results
print(
    f"Detected objects: {len(label_boxes)}, "
    f"Image size: {width}x{height}, "
    f"Speed: {result.speed}"
)

for box in label_boxes:
    print(box)
