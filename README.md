# AI-practice

`www.geun.my/posts`의 각 주제에 대한 예시 코드를 직접 실행해 볼 수 있도록 모아둔 실습용 코드 저장소입니다.  
AI의 기본 개념부터 대표적인 알고리즘, 실제 응용 예제까지 단계적으로 학습할 수 있도록 구성했습니다.

## 들어가기 앞서

이 저장소는 `www.geun.my/posts`의 AI 관련 글과 함께 학습할 수 있도록 구성되어 있습니다.  
작성자가 공부하며 기록하기 위한 장소로, **수시로 내용이 추가·변경·제거될 수 있습니다.**  
**대부분의 자료는 AI를 활용하여 작성**되었습니다.  


### 기초편

기초편은 개념 이해를 위한 이론 중심 내용이고, **기본편부터 실제 실행 가능한 코드**가 포함됩니다.

| 기초편 | 제목 |
| :-- | :-- |
| [기초 1편](https://duckport.pages.dev/posts/AI_intro) | AI, 머신러닝, 딥러닝 차이점 개념 정리 |
| [기초 2편](https://duckport.pages.dev/posts/AI_ML) | 머신러닝 알고리즘 종류 정리 |
| [기초 3편](https://duckport.pages.dev/posts/AI_DL) | 딥러닝 신경망 구조 정리 (CNN, RNN, Transformer) |
| [기초 4편](https://duckport.pages.dev/posts/AI_generative) | 생성형 AI란? GPT, DALL·E, Stable Diffusion 비교 |
| [기초 5편](https://duckport.pages.dev/posts/AI_finetuning) | 전이 학습이란? 실전 파인튜닝 가이드 |

## 0. Python 가상환경 실행

- 기본적으로 Python은 가상환경에서 실행하는 것을 권장합니다.
- `envSetting.ipynb` 파일은 가상환경 생성부터 인터프리터 설정까지 한 번에 진행할 수 있도록 구성되어 있습니다.
- 파일을 실행한 뒤 `Run All`을 실행하면 됩니다.

## 1. Basic Folder

AI에 활용되는 기본 라이브러리 실습 코드를 모아둔 폴더입니다.  

### 기본편

| 기본편 | 제목 | 역할 |
| :-- | :-- | :-- |
| [기본 1편](https://duckport.pages.dev/posts/NumPy) | NumPy 완벽 정리 — 숫자 배열 다루기 | 데이터 기초 |
| [기본 2편](https://duckport.pages.dev/posts/Pandas) | Pandas 완벽 정리 — 데이터 불러오고 정리하기 | 데이터 전처리 |
| [기본 3편](https://duckport.pages.dev/posts/Matplotlib_Seaborn) | Matplotlib / Seaborn — 데이터 시각화 | 시각화 |
| [기본 4편](https://duckport.pages.dev/posts/Scikit_learn) | Scikit-learn — 머신러닝 실습 입문 | ML 실습 |
| [기본 5편](https://duckport.pages.dev/posts/Keras) | TensorFlow / Keras — 딥러닝 모델 만들기 | DL 실습 |
| [기본 6편](https://duckport.pages.dev/posts/PyTorch) | PyTorch 입문 — Keras와 무엇이 다를까? | DL 심화 |

## 2. Deepen Folder

AI에서 대표적으로 활용되는 알고리즘 실습 코드를 모아둔 폴더입니다.  
`www.geun.my/posts`의 `AI_deep_dive` 카테고리에 해당하며, **심화편 5개만** 이 폴더에 포함됩니다.

### 심화편

| 심화편 | 제목 | 난이도 | 역할 |
| :-- | :-- | :-- | :-- |
| [심화 1편](https://duckport.pages.dev/posts/CNN_Deep) | CNN — 이미지 분류 완벽 정리 | ⭐⭐ | 이미지 처리 |
| [심화 2편](https://duckport.pages.dev/posts/RNN) | RNN / LSTM — 시계열·텍스트 처리 | ⭐⭐⭐ | 순서 데이터 |
| [심화 3편](https://duckport.pages.dev/posts/Transformer) | Transformer 구조 완벽 정리 | ⭐⭐⭐⭐ | 현대 AI 핵심 |
| [심화 4편](https://duckport.pages.dev/posts/BERT) | BERT / GPT 원리와 활용 | ⭐⭐⭐⭐ | 언어 모델 |
| [심화 5편](https://duckport.pages.dev/posts/Finetuning_Deep) | HuggingFace로 LLM Fine-tuning | ⭐⭐⭐⭐⭐ | LLM 실전 |

## 3. Application Folder **준비중**

정해진 목표를 AI를 활용하여 달성하는 응용 실습 코드를 모아둔 폴더입니다.  
`www.geun.my/posts`의 `AI_deep_dive` 카테고리에 포함되며, **Deepen 외의 응용 예제**를 이 폴더에 넣습니다.

### 응용편 

| 응용편 | 제목 | 난이도 | 핵심 기술 |
| :-- | :-- | :-- | :-- |
| [응용 1](https://duckport.pages.dev/posts/AiCharacterImageGenerationWithLoRA) | 사진 10장으로 AI 캐릭터·프로필 이미지 만들기 | ⭐⭐⭐ | FLUX.1 dev · LoRA · ComfyUI |
| [응용 2](https://duckport.pages.dev/posts/AiVoiceCloningWithF5TtsAndRVC) | 내 목소리 AI 클론 — 유튜브 내레이터 자동화 | ⭐⭐⭐ | F5-TTS · Kokoro · Sesame CSM-1B |
| [응용 3](https://duckport.pages.dev/posts/ProductImageToVideoWithWan2) | 상품 이미지 1장 → 15초 광고 영상 자동 생성 | ⭐⭐⭐⭐ | Wan2.2 · HunyuanVideo 1.5 · LTX-2 |
| [응용 4](https://duckport.pages.dev/posts/FactoryDefectDetectionWithYOLOv12) | 공장 불량 자동 검사 — NG/OK 탐지 + 로봇 좌표 추출 | ⭐⭐~⭐⭐⭐⭐⭐ | YOLOv12 · OpenCV · RealSense |
| [응용 5](https://duckport.pages.dev/posts/AutoBgmGenerationWithMusicGen) | 영상 분위기 분석 → BGM 자동 생성 & 싱크 | ⭐⭐⭐ | MusicGen · AudioCraft · Stable Audio Open |
| [응용 6](https://duckport.pages.dev/posts/YoutubeContentAutomationWithLangGraph) | 주제 한 줄 입력으로 유튜브 영상 완성 — AI 콘텐츠 자동화 파이프라인 | ⭐⭐⭐⭐ | LangGraph · CrewAI · AutoGen 0.4 |
| [응용 7](https://duckport.pages.dev/posts/VisionLlmProductDescriptionAutomation) | 사진 보고 글 쓰는 AI — Vision LLM 상세페이지 자동 작성 | ⭐⭐⭐⭐ | Qwen2.5-VL · InternVL3 · LLaVA-Next |
| [응용 8](https://duckport.pages.dev/posts/RagChatbotWithLlamaIndexAndChromaDB) | 내 PDF 문서를 AI가 읽는다 — 사내 지식 RAG 챗봇 구축 | ⭐⭐⭐ | LlamaIndex · ChromaDB · Qdrant |

**사이트 포스트의 `AI_deep_dive` 카테고리에는 `Deepen`과 `Application`이 함께 포함됩니다.**