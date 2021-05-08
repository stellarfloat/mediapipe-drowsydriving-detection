# mediapipe-drowsydriving-detection

2021-1 차량지능기초 전반부 최종 프로젝트의 레포지토리입니다.

운전자의 졸음운전 상태를 하품 횟수 측정을 통해 알려줍니다.

## 구조

```text
mediapipe-drowsydriving-detection/
├── Pipfile
├── Pipfile.lock
├── README.md
├── face_img_index.png
├── infer.py
├── joseph-gonzalez-iFgRcqHznqg-unsplash.jpg
├── landmark_coors_test.py
└── main.py
```

## 설치하기

* Prerequisite:
`Python 3.9+`

로컬 디렉토리에 이 저장소를 클론합니다.

    git clone https://github.com/stellarfloat/mediapipe-drowsydriving-detection.git

클론한 디렉토리 안에서 다음의 명령을 실행하여 환경을 구성합니다.

    pipenv install



## 사용법

### 졸음운전 감지

_PC에 연결된 웹캠이 있어야 합니다._

    python3 main.py

실행과 동시에 분석을 시작하며, 처음 30프레임은 평균 필터 초기화를 위해 졸음운전 판정이 이루어지지 않습니다. 이후 초기화가 완료되면, 운전자가 하품을 할 때마다 콘솔창에 `Warning: driver is drowsy | t = {}`와 같은 메시지가 출력됩니다. t 값은 프로그램이 시작된 이후 경과한 시간입니다.


### Face landmark 인덱스 시각화

`landmark_coors_test.py` 에서 하단의 `mark_index_image()` 함수를 호출합니다. 첫번째 파라미터는 얼굴이 나타난 사진의 경로이고, 두번째 파라미터는 결과물이 저장될 경로입니다.

`mark_index_webcam()`을 호출할 경우, 웹캠의 비디오 스트림에서 동일한 작업을 수행합니다.

```python
if __name__ == '__main__':
    #mark_index_webcam()
    mark_index_image('./joseph-gonzalez-iFgRcqHznqg-unsplash.jpg', './face_img_index.png')
```



## 크레딧
`소프트웨어학부_20203155_추헌준`

사용된 라이브러리:
- mediapipe
- opencv
