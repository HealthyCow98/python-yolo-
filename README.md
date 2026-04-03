# Vision AI - Needle Defect Detection

## 📌 프로젝트 개요

주사바늘 이미지에서 불량을 검출하기 위한 Vision AI 프로젝트

- OpenCV 기반 Detection + Rule 기반 처리
- YOLO 기반 Detection + Classification 개선

---

## ⚙️ 시스템 구조

### 1. 기존 방식 (OpenCV)

- Threshold 기반 이진화
- Contour 기반 바늘 검출
- Bounding Box 추출 후 Crop

  문제

- 조명 / 해상도에 따라 detection 결과 불안정
- 동일 객체라도 crop 영역이 달라짐
- classification 성능 저하 발생

---

### 2. 개선 방식 (YOLO)

- YOLO Detection 모델 도입
- 안정적인 Bounding Box 생성
- Crop → Classification 파이프라인 구성

개선 효과

- Detection 안정성 확보
- 데이터 일관성 향상
- 전체 파이프라인 성능 개선

---

## 핵심 기술

### Bounding Box 정규화

- 바늘 구멍 위치 기준 중심 재설정
- 크기 고정 및 확장
- 이미지 경계 보정

-> 데이터 품질 개선

---

## 핵심 문제 해결

> Detection의 불안정성이 전체 AI 성능의 병목이 되는 문제를 확인하고  
> Rule 기반에서 학습 기반(YOLO)으로 전환하여 해결

---

## 결과

- Detection 정확도 향상
- Classification 안정성 확보
- 실사용 가능한 구조 설계

---

---

## 기술 스택

- Python
- OpenCV
- YOLO (Ultralytics)
