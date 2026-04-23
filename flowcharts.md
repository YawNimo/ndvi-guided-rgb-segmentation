# Pipeline Flowcharts

## 1) Preprocessing

```mermaid
flowchart TD
    A([Start preprocessing]) --> F[blur images]
    F --> J{Create pixel-level masks of each image}
    J --> J1[Read green/red/nir bands]
    J1 --> J2[Compute NDVI and NDWI]
    J2 --> D1{NDWI >= 0.20?}
    D1 -- Yes --> J4["Set class water (0)"]
    D1 -- No --> D2{NDVI >= 0.5?}
    D2 -- Yes --> J6["Set class dense vegetation (3)"]
    D2 -- No --> D3{NDVI >= 0.2?}
    D3 -- Yes --> J5["Set class sparse vegetation (2)"]
    D3 -- No --> J7["Set class impervious (1)"]
    J4 --> W[write mask]
    J5 --> W
    J6 --> W
    J7 --> W
    W --> Z([End])
```

## 2) Training

```mermaid
flowchart TD
    A([Start training/main.py]) --> D[create_data_split from input/images and input/masks]
    D --> J{Epoch loop 1..N}

    J --> K[run_one_epoch train mode]
    K --> L{Validation epoch?}
    L -- No --> P{N reached?}
    L -- Yes --> M[run_one_epoch eval mode]
    M --> N[Compute F1, IoU, accuracy and optional full metrics]
    N --> O{Best val macro F1 improved?}
    O -- Yes --> O1[Save best checkpoint in checkpoints/]
    O1 --> O2[Reset no-improvement counter]
    O -- No --> O3[Increment no-improvement counter]
    O2 --> P
    O3 --> P
    P -- Yes --> R[Finalize summaries and optional plots]
    P -- No --> Q{Early stop patience reached?}
    Q -- Yes --> R
    Q -- No --> J
    R --> S([End])
```

## 3) Runmodel (Inference)

```mermaid
flowchart TD
    A([Start runmodel/main.py]) --> F[Load checkpoint and set eval mode]
    F --> G{For each tile}
    G --> H[Load RGB tile]
    H --> I[Forward pass and argmax prediction]
    I --> J[Save predicted mask for visualization]
    J --> G
    G -->|Done| L([End])
```
- Runmodel uses trained checkpoints and tile images for prediction; ground-truth masks are only needed for optional visualization comparisons.

## 4) High-Level End-to-End Flow

```mermaid
flowchart LR
    A[Raw geotiff URLs CSV] --> B[Preprocessing: download and unzip]
    B --> C[Preprocessing: tile]
    C --> D[Preprocessing: blur]
    D --> E[Preprocessing: create masks]

    C --> C1[input/images tiles]
    E --> E1[input/masks labels]

    C1 --> F[Training main.py]
    E1 --> F
    F --> G[checkpoints best model .pt]

    G --> I[Runmodel main.py inference]
    C1 --> I
    I --> J[results model pred_masks]
```

## 5) Landcover Verification

```mermaid
flowchart TD
    A([Start run_landcover_verification.sh]) --> D[Step 1: convert_landcover_dataset.py to remapped_landcover_masks]
    D --> F[Step 2: runmodel/main.py on remapped_landcover_masks]
    F --> G[Write predictions to datasets/pred_masks]
    G --> J[Step 3: score_landcover_metrics.py pred_masks vs remapped_landcover_masks]
    J --> K[Write landcover_metrics_scores.csv]
    K --> M([Verification complete])
```
