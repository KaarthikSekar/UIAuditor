# Android UI Understanding using YOLO, Clustering & RAG

An end-to-end AI pipeline for understanding Android user interfaces from screenshots using **computer vision, machine learning, clustering, embeddings, vector search, and an LLM**.

The project starts with Android UI screenshots and progressively transforms them into structured component information, visual patterns, spatial relationships, and finally a question-answering system powered by **RAG + Gemini**.

---

## 🚀 Project Overview

The main objective is to move from:

**Screenshot → UI Components → Attributes → Patterns → Relationships → Knowledge → RAG → LLM**

Instead of only detecting UI elements, the system attempts to build a structured understanding of Android UI layouts and allow users to ask natural-language questions about the extracted knowledge.

---

## 🗂️ Dataset

This project uses the **RICO dataset**, a large-scale dataset of Android application user interfaces.

RICO contains screenshots together with Android UI hierarchies and semantic annotations. The dataset provides visual, textual, structural, and interactive information about Android UIs.

Official dataset:

**RICO — A Mobile App Dataset for Building Data-Driven Design Applications**
https://www.interactionmining.org/archive/rico

The official dataset page describes more than **66,000 unique UI screens** and approximately **3 million UI elements**, with screenshots and corresponding UI hierarchies.

> **Note:** The RICO screenshots may contain copyrighted material. Users should review and follow the dataset's terms and copyright policy when downloading, using, or redistributing the data.

---

# 🧠 Project Pipeline

```text
RICO Dataset
     │
     ▼
Dataset Exploration
     │
     ▼
Semantic Analysis
     │
     ▼
RICO → YOLO Format Conversion
     │
     ▼
Dataset Preparation
     │
     ▼
YOLOv8n Training
     │
     ▼
Model Evaluation
     │
     ▼
YOLO Inference / Testing
     │
     ▼
UI Attribute Extraction
     │
     ▼
Clustering & Pattern Discovery
     │
     ▼
Component Relationships
     │
     ▼
RAG Knowledge Dataset
     │
     ▼
Sentence Embeddings
     │
     ▼
ChromaDB Vector Store
     │
     ▼
Gemini LLM
     │
     ▼
Natural Language UI Understanding
```

---

# 📓 Notebooks

## Notebook 1 — Dataset Exploration

Explores the RICO dataset and its structure.

### Tasks

* Inspect screenshots
* Explore JSON UI hierarchies
* Understand component annotations
* Examine bounding-box information
* Analyze available UI component categories

---

## Notebook 2 — Semantic Analysis

Analyzes the semantic UI annotations and determines the component classes used for the detection task.

### Tasks

* Inspect semantic component labels
* Analyze component distributions
* Identify useful UI component categories
* Prepare the class vocabulary for object detection

---

## Notebook 3 — RICO → YOLO Conversion

Converts RICO bounding-box annotations into the format required by YOLO.

### Conversion

RICO:

```text
x1, y1, x2, y2
```

YOLO:

```text
class_id
center_x
center_y
width
height
```

Coordinates are normalized relative to the screenshot dimensions.

---

## Notebook 4 — Dataset Preparation

Prepares the YOLO dataset for model training.

### Tasks

* Organize images and labels
* Create training/validation/testing splits
* Generate YOLO configuration
* Verify annotations
* Check class mappings

---

# 🤖 Notebook 5 — YOLOv8n Training & Evaluation

Trains a YOLOv8n object-detection model to detect Android UI components.

### Why YOLO?

The problem requires **object detection**, not simple image classification.

The model needs to determine:

```text
What is the component?
        +
Where is the component?
```

YOLO provides:

* Component class
* Bounding box
* Confidence score

### Why YOLOv8n?

The nano model provides a lightweight architecture suitable for experimentation and relatively fast inference while supporting multi-class object detection.

### Evaluation Metrics

The model is evaluated using:

* Precision
* Recall
* mAP@50
* mAP@50–95

### Why mAP?

mAP evaluates both:

* Correct component classification
* Bounding-box localization

IoU is used to determine how well the predicted bounding box overlaps with the ground-truth box.

---

# 🔍 Notebook 6 — YOLO Testing & Inference

Uses the trained `best.pt` model on unseen RICO screenshots.

### Outputs

For each detected UI component:

```text
Component class
Confidence
Bounding box
x1
y1
x2
y2
```

Example:

```text
Button
Confidence: 0.94
Bounding Box: [x1, y1, x2, y2]
```

The predictions are also visualized on screenshots to manually inspect model performance.

---

# 📐 Notebook 7 — Attribute Extraction

Moves beyond detection and extracts additional information from each detected UI component.

### Geometric Attributes

* Width
* Height
* Area
* Aspect ratio
* Center X/Y
* Normalized position
* Normalized width/height

### Visual Attributes

* Brightness
* Contrast
* Minimum intensity
* Maximum intensity

### Text Attributes

OCR is used to determine whether text is present inside detected component regions.

Extracted information includes:

* Detected text
* `has_text`
* Text count

The result is a structured component dataset.

---

# 🔬 Notebook 8 — Clustering & Pattern Discovery

Uses unsupervised learning to discover groups of visually and geometrically similar UI components.

### Features

Examples include:

```text
width
height
area
aspect_ratio
position
normalized position
brightness
contrast
confidence
text presence
```

### Preprocessing

Features are standardized using `StandardScaler`.

This prevents large-scale variables such as area from dominating distance-based clustering.

### Clustering

**K-Means clustering** is used to group components according to their extracted characteristics.

Different values of `K` are tested.

### Cluster Evaluation

The **Silhouette Score** is used to compare different numbers of clusters.

Example:

```text
K=2 → 0.4950
K=3 → 0.5246
K=4 → 0.3179
...
```

The tested configuration with the strongest silhouette score was selected for further analysis.

---

# 📊 PCA Visualization

PCA is used primarily to visualize the high-dimensional component feature space.

The original feature space contains many dimensions:

```text
width
height
area
aspect ratio
brightness
contrast
position
...
```

PCA reduces this representation to two principal components for visualization.

```text
High-dimensional features
          ↓
         PCA
          ↓
      PC1 + PC2
          ↓
       2D plot
```

The clustering itself is performed in the standardized feature space; PCA is used to make the clusters visually interpretable.

---

# 🔗 Notebook 9 — Component Patterns & Relationships

Individual components do not exist independently in a UI.

This notebook analyzes relationships between UI components.

### Co-occurrence

Examples:

```text
Button + EditText
Button + CheckedTextView
EditText + TextView
```

This helps identify which components frequently appear together.

### Spatial Relationships

Relationships such as:

```text
left of
right of
above
below
```

are extracted from component positions.

This moves the project from:

**Component Detection**

toward:

**UI Structure Understanding**

---

# 📚 Notebook 10 — RAG Knowledge Dataset

The extracted analysis is transformed into structured knowledge records suitable for retrieval.

Instead of sending the raw component dataset directly to an LLM, information is converted into meaningful records such as:

```text
Component profiles
Cluster profiles
Co-occurrence patterns
Spatial relationships
Attribute summaries
```

This creates the knowledge layer used by the RAG system.

---

# 🧮 Notebook 11 — Embeddings & ChromaDB

The RAG knowledge records are converted into semantic embeddings using a Sentence Transformer model.

### Why embeddings?

Embeddings allow semantically similar questions and knowledge records to be matched even when they don't use exactly the same words.

```text
User Question
      ↓
Embedding
      ↓
Similarity Search
      ↓
Relevant Knowledge
```

### Why ChromaDB?

ChromaDB provides a lightweight vector database for storing embeddings and performing semantic similarity search.

It works well for this project because it:

* Integrates easily with Python
* Supports vector similarity search
* Supports metadata
* Can run locally
* Provides persistent storage

---

# 🧠 Notebook 12 — Gemini + RAG

The final layer combines:

**User Question → Retrieval → Gemini**

The user asks a question about Android UI patterns.

```text
User Question
      ↓
Question Embedding
      ↓
ChromaDB
      ↓
Relevant RICO Knowledge
      ↓
Gemini
      ↓
Natural Language Answer
```

Gemini is responsible for interpreting the question and generating a natural-language response using the retrieved project-specific context.

---

# 💬 Example Questions

The final system can answer questions such as:

### Components

* What UI components are most common in the analyzed dataset?
* What components commonly occur with Buttons?
* What is the typical aspect ratio of a Button?

### Clustering

* What are the main characteristics of Cluster 0?
* Which components dominate Cluster 1?
* How do the three clusters differ?

### Relationships

* What components commonly appear above an EditText?
* What components are usually found below a Button?
* What spatial patterns exist between Buttons and EditTexts?

### Combined Reasoning

> What components commonly occur with Buttons, and where are those components typically positioned relative to the Button?

> If a Button is detected, what other components might commonly appear nearby based on the analyzed RICO data?

---

# 🛠️ Technologies Used

| Technology            | Purpose                                  |
| --------------------- | ---------------------------------------- |
| Python                | Core development                         |
| RICO                  | Android UI dataset                       |
| YOLOv8n               | UI component detection                   |
| OpenCV                | Image processing                         |
| Pillow                | Image manipulation                       |
| EasyOCR               | Text extraction                          |
| Pandas                | Data processing                          |
| NumPy                 | Numerical operations                     |
| Scikit-learn          | Scaling, clustering, PCA, metrics        |
| K-Means               | Unsupervised clustering                  |
| PCA                   | Dimensionality reduction & visualization |
| Sentence Transformers | Text embeddings                          |
| ChromaDB              | Vector database                          |
| Gemini                | LLM / natural-language generation        |

---

# 📁 Repository Structure

```text
RICO-UI-Understanding/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_semantic_analysis.ipynb
│   ├── 03_rico_to_yolo.ipynb
│   ├── 04_dataset_preparation.ipynb
│   ├── 05_yolo_training_evaluation.ipynb
│   ├── 06_yolo_testing_inference.ipynb
│   ├── 07_attribute_extraction.ipynb
│   ├── 08_clustering.ipynb
│   ├── 09_component_relationships.ipynb
│   ├── 10_rag_dataset.ipynb
│   ├── 11_embeddings_chromadb.ipynb
│   └── 12_llm_rag.ipynb
│
├── data/
│   ├── sample_structured_components.csv
│   └── sample_rag_knowledge.json
│
├── src/
│   └── reusable_scripts/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🔐 Model & Dataset Files

Large/private files are intentionally excluded from the repository.

This includes:

* RICO screenshots
* Full RICO annotations
* YOLO model weights (`best.pt`)
* Generated crops
* ChromaDB database
* API keys and credentials
* Environment files

Users should download the RICO dataset directly from the official source and configure their own model/API credentials.

---

# ⚙️ Installation

Clone the repository and install the dependencies:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RICO-UI-Understanding

pip install -r requirements.txt
```

Create a `.env` file for API credentials when required:

```text
GEMINI_API_KEY=your_api_key_here
```

**Never commit `.env` or API keys to GitHub.**

---

# 🎯 Project Goal

The project explores how computer vision and language models can work together to build a richer understanding of Android user interfaces.

Rather than stopping at:

```text
Screenshot → Detected Button
```

the pipeline attempts to reach:

```text
Screenshot
   ↓
UI Components
   ↓
Attributes
   ↓
Visual Patterns
   ↓
Component Relationships
   ↓
Structured Knowledge
   ↓
Semantic Retrieval
   ↓
LLM
   ↓
Natural Language UI Understanding
```

---

# 🚀 Future Improvements

Potential extensions include:

* Increase the number of RICO screenshots used for training
* Improve detection performance for underrepresented UI classes
* Add more UI attributes
* Improve OCR extraction
* Explore richer component relationships
* Improve RAG retrieval
* Add UI screenshot-based multimodal questioning
* Build an interactive UI-analysis application
* Compare multiple embedding models and LLMs

---

# 📌 Disclaimer

This project is intended for educational and research purposes.

The RICO dataset is maintained by the Interaction Mining research team. Please refer to the official dataset page and its associated terms before downloading or redistributing dataset contents.

### Dataset

https://www.interactionmining.org/archive/rico

### RICO Paper

Deka et al., *Rico: A Mobile App Dataset for Building Data-Driven Design Applications*, UIST 2017.

---

## ⭐ Project Summary

**RICO + YOLOv8n + Attribute Extraction + Clustering + Relationship Analysis + RAG + ChromaDB + Gemini**

A complete pipeline for turning Android UI screenshots into structured, searchable, and queryable UI knowledge.
