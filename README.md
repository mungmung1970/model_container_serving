# 디렉토리 구성
```
project/
├─ researcher1/ (Train)
│  ├─ Dockerfile
│  ├─ requirements.txt
│  ├─ train.py
│  └─ data/
│     ├─ train.csv
│     └─ test.csv
│
├─ researcher2/ (Inference)
│  ├─ Dockerfile
│  ├─ requirements.txt
│  └─ inference.ipynb
│
└─ docker-compose.yml

```

# 실행순서
```
# 1. 이미지 재빌드 (연구자 1)
cd researcher1
docker build -t mungmung1970/perf-trainer:1.0 .

# 2. Docker Hub 재업로드
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


# 기타사항 - 추가로 테스트 할 사항
```
- RandomForest → ONNX 가능 모델 추천
- Dev Container로 하는 방식
- FastAPI 서빙 추가
- MLflow로 확장한 MLOps 구조

```