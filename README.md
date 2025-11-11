# Case Study

---

## Kurulum

### Docker İmajını Oluşturma

```bash
docker build -f Dockerfile -t case_study .
```

### Konteyneri Başlatma

```bash
docker run -it --name case_study_container case_study /bin/bash
docker run -it --gpus all --name case_study_container case_study /bin/bash
```

### Veri Setlerini İndirme ve Ön İşleme

```bash
python src/dataset_downloader.py
```


## Örnek Komutlar

### Model Eğitimi
Not: Eğitilmiş modeller outputs klasörüne kaydedilir.
```bash
python src/task1/train.py
python src/task2/train.py
```

### Test

```bash
python src/task1/predict.py "/app/dataset/processed/task1/test/fake/fake_360.jpg"
python src/task2/predict.pt "/app/dataset/processed/task2/images/140_F_NA1.png"
```

### Eğitim Sonrası Performans Analizi
Not: Eğitilmiş modeller için karışıklık matrisi, Precision/Recall/F1 ve ROC-AUC grafikleri outputs klasörüne kaydedilir.

```bash
python src/task1/evulate.py
python src/task2/evulate.py
```

## GPU

Model eğitimi için kullanılan GPU ve sürücü bilgileri

|           |                   |
|-----------------|-----------------------|
| GPU              | NVIDIA RTX A5000      |
| Driver Version   | 550.90.07             |
| CUDA Version     | 12.4                  |

## Dizin Yapısı

```bash
.
├── dataset/
│   ├── processed/
│   │   ├── task1/
│   │   │   ├── test/
│   │   │   │   ├── fake/
│   │   │   │   │   ├── fake_11.jpg
│   │   │   │   │   ├── fake_116.jpg
│   │   │   │   │   ├── ...
│   │   │   │   ├── real/
│   │   │   │   │   ├── real_113.jpg
│   │   │   │   │   ├── real_126.jpg
│   │   │   │   │   ├── ...
│   │   │   ├── train/
│   │   │   │   ├── fake/
│   │   │   │   │   ├── fake_1.jpg
│   │   │   │   │   ├── fake_10.jpg
│   │   │   │   │   ├── ...
│   │   │   │   ├── real/
│   │   │   │   │   ├── real_1.jpg
│   │   │   │   │   ├── real_10.jpg
│   │   │   │   │   ├── ...
│   │   │   ├── val/
│   │   │   │   ├── fake/
│   │   │   │   │   ├── fake_102.jpg
│   │   │   │   │   ├── fake_112.jpg
│   │   │   │   │   ├── ...
│   │   │   │   ├── real/
│   │   │   │   │   ├── real_103.jpg
│   │   │   │   │   ├── real_105.jpg
│   │   │   │   │   ├── ...
│   │   ├── task2/
│   │   │   ├── images/
│   │   │   │   ├── 001_F_BC1.png
│   │   │   │   ├── 001_F_BC2.png
│   │   │   │   ├── ...
│   │   │   ├── masks/
│   │   │   │   ├── 001_M.png
│   │   │   │   ├── 002_M.png
│   │   │   │   ├── ...
│   ├── raw/
│   │   ├── CoMoFoD_small_v2/
│   │   │   ├── 001_B.png
│   │   │   ├── 001_F.png
│   │   │   ├── ...
│   │   ├── hardfakevsrealfaces/
│   │   │   ├── fake/
│   │   │   │   ├── fake_1.jpg
│   │   │   │   ├── fake_10.jpg
│   │   │   │   ├── ...
│   │   │   ├── real/
│   │   │   │   ├── real_1.jpg
│   │   │   │   ├── real_10.jpg
│   │   │   │   ├── ...
│   │   │   ├── data.csv
│   │   ├── CoMoFoD_small_v2.zip
│   │   ├── hardfakevsrealfaces.zip
├── models/
│   ├── final_task1_resnet18_model.pth
│   ├── final_task2_unet_model.pth
├── outputs/
│   ├── task1/
│   │   ├── best_resnet18_model.pth
│   │   ├── confusion_matrix.png
│   │   ├── metrics.png
│   │   ├── roc_auc.png
│   ├── task2/
│   │   ├── best_unet_model.pth
│   │   ├── confusion_matrix.png
│   │   ├── metrics.png
│   │   ├── pred_image.png
│   │   ├── pred_mask.png
│   │   ├── roc_auc.png
├── src/
│   ├── task1/
│   │   ├── evulate.py
│   │   ├── predict.py
│   │   ├── train.py
│   ├── task2/
│   │   ├── evulate.py
│   │   ├── predict.py
│   │   ├── train.py
│   ├── dataset_downloader.py
├── Dockerfile
├── requirements.txt
