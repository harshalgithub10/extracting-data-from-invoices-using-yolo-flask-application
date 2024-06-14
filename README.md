# Flask YOLOv4 Invoice Detection and OCR Application

This repository contains a Flask web application that uses YOLOv4 for detecting objects in invoice images and Tesseract OCR for extracting text. The extracted text is then processed to extract key information like invoice number, date, total amount, and more.

## Features

- Upload invoice images.
- Detect objects in images using YOLOv4.
- Extract text from detected regions using Tesseract OCR.
- Parse and extract key information from the text.
- Display extracted text and key information in a JSON format.

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Flask
- OpenCV
- pytesseract
- PIL (Pillow)

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. **Create and activate a virtual environment:**

   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install required packages:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Download YOLOv4 weights:**

   ```sh
   wget https://pjreddie.com/media/files/yolov4.weights -O yolo/yolov4-custom.weights
   ```

5. **Ensure the following directory structure:**

   flask3 ```
   .
   ├── app.py
   ├── requirements.txt
   ├── templates
   │   ├── index.html
   │   └── result.html
   ├── static
   │   └── uploads
   └── yolo
       ├── yolov4-custom.cfg
       ├── yolov4-custom.weights
       └── obj.names
   ```

### YOLO Configuration

Ensure your `yolov4-custom.cfg` and `obj.names` files are correctly configured for your use case.

### Running the Application

1. **Start the Flask application:**

   ```sh
   python app.py
   ```

2. **Open your web browser and navigate to:**

   ```
   http://127.0.0.1:5000/
   ```

## Usage

- **Upload an Invoice Image:** Select an image file and click upload.
- **View Detected Objects and Extracted Text:** The application will display detected objects in the image and extract text.
- **View Key Information:** Extracted key information from the text will be displayed in JSON format.

## Code Explanation

### `app.py`

This is the main Flask application file.

- **File Upload:** Handles file uploads and saves them to the `static/uploads` directory.
- **YOLO Detection:** Uses OpenCV's DNN module to detect objects in the uploaded images using YOLOv4.
- **OCR Extraction:** Uses Tesseract to extract text from the detected regions in the images.
- **Information Extraction:** Uses regex patterns to extract key information from the OCR output.

### `templates/index.html`

HTML template for the file upload interface.

### `templates/result.html`

HTML template for displaying the detected objects, extracted text, and key information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLOv4](https://github.com/AlexeyAB/darknet) for the object detection model.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction.

```

Make sure to update the GitHub repository URL (`https://github.com/harshalgithub10/extracting-data-from-invoices-using-yolo-flask-application`) with your actual repository URL. Additionally, ensure all file paths and configurations are correctly set up in your project.
