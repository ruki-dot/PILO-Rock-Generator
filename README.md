# PILO: Physics-Informed Latent Outpainting for 3D Digital Rock Reconstruction

This repository contains the official implementation of **PILO**, a novel generative AI system that reconstructs large-scale (128³ voxels), physically-consistent 3D digital rock structures from small (64³ voxels) input samples. The core of this method is a hierarchical "inside-out" generation strategy, conditioned on key physical properties to ensure realistic and coherent outputs.

![Teaser Image](https://i.imgur.com/your-teaser-image.png) 
*(Please replace this with a real image showing a 64³ input and a 128³ output)*

---

## ✨ Key Features

-   **Physics-Informed Generation**: The diffusion model is conditioned on 9 key physical properties (e.g., porosity, permeability) calculated from the input sample, guiding the generation process towards physically plausible structures.
-   **High-Fidelity Reconstruction**: Transforms a 64x64x64 voxel sample into a coherent 128x128x128 structure.
-   **Automated Batch Processing**: Simply place one or more valid sample folders into the `input/` directory and run a single command. The script automatically finds, validates, and processes them in a loop.
-   **Ready to Use**: A streamlined setup process and a single script to run inference.

---

## 🚀 Getting Started

### 1. Prerequisites

-   Python 3.8+
-   NVIDIA GPU with CUDA support is **highly recommended** for acceptable performance.
-   `git` for cloning the repository.

### 2. Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/YOUR_USERNAME/PILO-Rock-Generator.git
cd PILO-Rock-Generator
```

Next, install the required Python packages. It is strongly recommended to use a virtual environment.
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Pre-Trained Models

The pre-trained models are essential for running the reconstruction. Due to their size, they are not included in this repository.

**Action Required:**
1.  **[Click here to download the models (models.zip)](https://your-download-link.com/models.zip)**. 
    *(You must host the `ldm` and `vae` folders in a zip file on a service like Google Drive, Dropbox, or Hugging Face and replace this link.)*

2.  Unzip the downloaded file. You should now have two folders: `ldm` and `vae`.

3.  Place these two folders inside the `models/` directory. Your project structure should now match the one described in the file structure section.

---

## 💻 Usage

### Running the Reconstruction

1.  **Prepare Your Data**: 
    - Place your rock sample folder(s) into the **`input/`** directory.
    - Each sample folder (e.g., `my_rock_1`, `my_rock_2`) must contain **exactly 64 PNG images**, each 64x64 pixels.
    - To run a quick test, you can copy the provided example:
    ```bash
    # Example: Copy the sample to the input directory for processing
    cp -r sample_input/berea_sandstone_64 input/
    ```

2.  **Run the Script**: 
    Execute the main reconstruction script from the project's root directory.
    ```bash
    python reconstruct.py
    ```

The script will automatically find all valid folders in `input/`, process them one by one, and save the results in the `output/` directory.

### Understanding the Output

For each input sample named `my_rock_sample`, the script will generate two items in the `output/` folder:

-   A NumPy data file: `output/out_my_rock_sample.npy`
-   A folder with images: `output/out_my_rock_sample_images/` (contains 128 PNG slices)

---

## 🤝 Citation & Acknowledgements

### Citing this Work

If you use PILO in your research, please consider citing our work:
```bibtex
@article{PILO,
  title   = {PILO: Physics-Informed Latent Outpainting for 3D Digital Rock Reconstruction},
  author  = {HUANG YIZHUO},
  year    = {2025},
}
```

### Acknowledgements

This work heavily relies on the powerful pre-trained **Variational Autoencoder (VAE)** from the original **Stable Diffusion** model. We are immensely grateful to the authors for their groundbreaking research and for making their models publicly available.

The VAE is used for encoding input images into the latent space and decoding them back into pixel space. The specific model files used in this project originate from repositories like [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/vae) on Hugging Face.

Please cite the original Latent Diffusion Models paper if you build upon this component:
```bibtex
@inproceedings{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={10684--10695},
  year={2022}
}
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.