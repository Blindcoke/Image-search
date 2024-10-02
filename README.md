# Image Search Application

This project is an image search application built using FastAPI and provides features for uploading images and searching them by description. The application utilizes machine learning models for image captioning and semantic similarity comparison.

## Features

- Upload multiple images at once
- View selected and uploaded images
- Search images by their descriptions
- Display search results with image similarity

## Technologies Used

- FastAPI
- HTML/CSS/JavaScript
- OpenCV
- Pillow
- Transformers (Hugging Face)
- Sentence Transformers
- NumPy
- Scikit-learn

## Installation

1. **Clone the repository:**
   ```bash
      git clone https://github.com/your_username/image_search_app.git
      cd image_search_app
2. **Create a virtual environment (optional but recommended):**
   ```bash
     python -m venv venv
     source venv/bin/activate   # On Windows use `venv\Scripts\activate`
3. **Install the required packages:**
   ```bash
      pip install -r requirements.txt

## Usage

1. **Run the application:**
   ```bash
      uvicorn main:app --reload
2. **Access the Application:**

Open your browser and go to [http://localhost:8000](http://localhost:8000).

3.**Uploading Images**

  1. Click on **Choose File** to select images from your computer.
  2. Click the **Upload** button to upload the selected images.
  3. You can view uploaded files and selected files.

4. **Searching Images**

  Type a description in the search box to find similar images.

# License

This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments

- Thanks to Hugging Face for their transformers library.
- Thanks to OpenCV for image processing capabilities.
