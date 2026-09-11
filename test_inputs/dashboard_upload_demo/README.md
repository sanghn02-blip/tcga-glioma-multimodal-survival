# Dashboard Upload Demo Inputs

이 폴더는 대시보드 `위험 예측 실행` 화면 테스트용 입력 파일입니다.

## 올릴 파일

- RNA 표 파일: `demo_rna_high_risk.tsv`
- WSI 병리 이미지: `demo_wsi_pathology_image.jpg`

## 임상정보 입력 셀

- 환자 ID: `TCGA-06-5412`
- 프로젝트: `TCGA-GBM`
- 성별: `Unknown`
- 진단 시 나이: `78.746`
- 종양 등급: `Unknown`
- 진단명: `Glioblastoma`
- 분자아형: `GBM_IDHwt`
- 상세 암종: `Glioblastoma Multiforme`
- 종양 유형: `Glioblastoma Multiforme (GBM), Untreated`

## 참고

`demo_clinical_cells.json`은 같은 값을 JSON으로 저장한 참고 파일입니다. 현재 대시보드 화면에서는 임상정보를 JSON 파일로 올리는 대신 입력 셀에 직접 넣으면 됩니다.
