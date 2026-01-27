# 디렉토리 구성
```
project/
├─ artifacts/
│  ├─ features.json (researcher2 공유정보-feature)
|  ├─ metrics.json (researcher2 공유정보-target)
|  ├─ mission_test_add.csv(researcher2 전처리후 데이터)
|  └─ mission_test.csv(researcher2 공유 데이터)
│
├─ Data/ (Researcher2 적업 데이터)
│  ├─ mission15_test.csv
|  ├─ mission_train_add.csv(researcher1 전처리후 데이터)
|  └─ mission_traincsv(researcher2 훈련 데이터)
│
├─ researcher1/ (Train)
│  └─notebooks/
│     ├─ eda_preprocessing.ipynb (EDA 및 전처리)
│     ├─ eda_preprocessing.ipynb (EDA 및 전처리)
│     └─ modeling.ipynb(Train수행-노트북 버전)
│  ├─ Dockerfile 
│  ├─ requirements.txt
│  └─ train.py (훈련 python버전)
│
├─ researcher2/ (Inference)
│  ├─ Dockerfile
│  ├─ requirements.txt
│  ├─ inference.ipynb (추론(예측) 노트북 버전)
│  ├─ inference.py (추론(예측) 파이선 버전)
│  └─ preprocess.py (테스트 데이터 전처리 - 추가 feature 생성, inference.py에서 호출)
└─ docker-compose.yml

```

# 실행순서
```
# 1. 이미지 재빌드 (연구자 1)
cd researcher1
docker build -t mungmung1970/perf-trainer:1.0 .

# 2. Docker Hub 재업로드 (https://hub.docker.com/repository/docker/mungmung1970/perf-trainer/general)
docker push mungmung1970/perf-trainer:1.0

# 3. 컨테이너 실행
cd ..
docker compose up trainer

# 4. 파일 생성 확인(docker내부 관리, named volumn으로 컨테이너를 삭제해도 됨)
docker run --rm -v docker_dev_artifacts:/artifacts busybox ls -l /artifacts
✔ /artifacts에 다음 생성됨
model.onnx
features.json
metrics.json
test.csv

# 5. Jupyter Notebook 컨테이너 실행 (연구자 2)
docker compose up notebook


#6 브라우저 접속
콜솔에 아래와 같이 나옴.  이에 접속
http://127.0.0.1:8888/lab?token=토큰

#7 inference.ipynb 실행
/artifacts/result.csv 생성
```
