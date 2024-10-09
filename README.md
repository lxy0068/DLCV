# DLCV project

This project is a YOLO-based object detection and recognition system designed to perform real-time detection on images, videos, and camera feeds. The project provides a web interface for users to upload media and view detection results. It also supports model training and evaluation using custom datasets.

## Folder Structure

- **Dataset Folder (`datasets/`)**  
  Contains datasets used for training and evaluation, including:
  - `train/`: Training images and annotation files
  - `test/`: Testing images and annotation files
  - `val/`: Validation images and annotation files

  You can add custom datasets to this folder while maintaining the directory structure for proper training.

- **Icon Folder (`icons/`)**  
  Stores the icon files used in the web interface. The background and icon files can be modified as needed.

- **Training Result Folder (`runs/`)**  
  Stores the logs, training results, and evaluation charts generated during model training. Each training run will create a new folder, storing the final `.pt` model file and results.

- **Temporary Folder (`temp/`)**  
  Used for storing temporary files, including CSV files and labeled video results generated during program execution.

- **Test Image and Video Folder (`test_media/`)**  
  Stores media files (images and videos) used for testing model performance.

- **YOLO Package (`yolo/`)**  
  The official code package containing YOLO model-related scripts, modules, configuration files, and tools.

- **Model Folder (`models/`)**  
  Stores the trained model weight files. This includes pre-trained YOLO models (`.pt` files) and models trained on the custom datasets in this project.

## Main Files

- **`__init__.py`**  
  Python package initialization file, making the directory a Python package.

- **`LoggerRes.py`**  
  Handles page result recording and saving, logging detection results in tables, and saving them as CSV or video files.

- **`Recognition_UI.py`**  
  The layout code for the project's main interface. It includes the logic for generating and displaying the web interface.

- **`requirements.txt`**  
  A text file listing project dependencies and their versions for setting up the development environment.

- **`run_main_web.py`**  
  The main script to launch the web-based detection interface. Running this script will start the main detection page.

- **`run_test_camera.py`**  
  A script for testing camera input. It displays the detection results directly from the live camera feed.

- **`run_test_image.py`**  
  A script for testing object detection on images. Results are displayed directly on the screen.

- **`run_test_video.py`**  
  A script for testing video stream recognition. Detection results are shown directly on the screen for the video stream.

- **`run_train_model.py`**  
  The script for starting model training. If the GPU version of PyTorch is installed, the training will automatically run on the GPU; otherwise, it will default to CPU.

- **`style_css.py`**  
  Contains the CSS styles for the web interface, beautifying and organizing the display layout.

- **`utils_web.py`**  
  Project utility functions, including saving uploaded files, displaying detection results, and loading images.

- **`YOLOv8v5Model.py`**  
  YOLO model-related code, including model configuration, loading, and training logic.

- **`Environment configuration.txt`**  
  A text file containing environment configuration instructions to guide setup.

## Usage

### Running the Project

1. **Environment Setup**  
   Ensure you have the required dependencies installed by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launching the Web Interface**  
   Start the detection interface by running:
   ```bash
   python run_main_web.py
   ```
   Open the displayed URL in your web browser to access the interface.

3. **Testing Image/Video Detection**  
   You can test object detection on individual images, videos, or camera streams using the following scripts:
   - Image:  
     ```bash
     python run_test_image.py
     ```
   - Video:  
     ```bash
     python run_test_video.py
     ```
   - Camera:  
     ```bash
     python run_test_camera.py
     ```

4. **Training the Model**  
   To retrain the YOLO model on a custom dataset, use:
   ```bash
   python run_train_model.py
   ```
   The training results, including logs and model weights, will be saved in the `runs/` folder. For machines with limited memory, reduce the batch size in the script to avoid memory errors.

## Additional Information

- **Training Notes**  
  - The batch size can be adjusted in the `run_train_model.py` script to avoid memory issues.
  - Each training run creates a new folder in the `runs/` directory to store the trained model weights and result files.
  - GPU-based training will be automatically enabled if PyTorch with CUDA support is installed.

- **Testing Notes**  
  - Test images and videos are stored in the `test_media/` folder, which you can modify with your own test files.

![image](https://github.com/user-attachments/assets/0c7d0aee-ef54-4c9b-9b3e-1a6bc00a265f)
![image](https://github.com/user-attachments/assets/ce57ca4d-a0e2-4143-877e-ca543dc78faf)
![image](https://github.com/user-attachments/assets/e632842b-8692-40f7-8873-62db2cdc403c)
![image](https://github.com/user-attachments/assets/b3d8cedb-c244-4cc4-bec1-a7a09e62acb4)
https://github.com/user-attachments/assets/d06a5062-ae29-409b-a31d-73be7da85255



![labels_correlogram](https://github.com/user-attachments/assets/33ad64ac-43e5-4437-b350-be17a9392d7b)
![labels](https://github.com/user-attachments/assets/dac66b05-205c-4546-8f74-40de31e386d4)
![train_batch0](https://github.com/user-attachments/assets/809e8a6e-d395-4155-aa54-a559c4eb3399)
![F1_curve](https://github.com/user-attachments/assets/96c6bb42-6b0a-45ab-91a2-cad44d8b174b)
![R_curve](https://github.com/user-attachments/assets/76c39530-5a80-4ec1-8b95-92c7610529aa)
![PR_curve](https://github.com/user-attachments/assets/442442de-9fb7-46ff-a9be-90e7de5eac43)
![P_curve](https://github.com/user-attachments/assets/6c6bfa3b-7fb7-43ef-8bea-936866b1f708)
![results](https://github.com/user-attachments/assets/f663ea40-44a3-4cb6-b158-07e0b4a324fd)
![confusion_matrix_normalized](https://github.com/user-attachments/assets/5bad7744-c27e-4cd3-a22f-1f2a85ce9c27)
![confusion_matrix](https://github.com/user-attachments/assets/6237d5d4-6b7b-4e6c-8140-f8ccfda452bf)
