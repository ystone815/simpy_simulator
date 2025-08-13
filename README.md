# 🎯 SimPy GPU Simulator - 모듈화된 NVIDIA GPU 시뮬레이션 시스템

A comprehensive, modular discrete-event simulation system for NVIDIA H100 and B200 GPUs, built with SimPy. This simulator models GPU architectures from Thread/Warp level up to complete GPU systems, enabling detailed performance analysis of AI workloads with advanced storage optimization patterns.

## 🚀 핵심 기능

### 🏗️ 모듈화된 아키텍처 
- **체계적인 프로젝트 구조**: config/, src/, tests/, results/, scripts/로 분리
- **YAML 기반 설정 시스템**: 외부 설정 파일로 완전 파라미터화  
- **Factory 패턴 객체 생성**: 설정 기반 자동 객체 생성
- **CLI 벤치마크 실행기**: 명령줄 인터페이스로 간편한 실행

### 🔬 GPU 아키텍처 모델링
- **Thread-Level 시뮬레이션**: 개별 GPU 스레드의 레지스터 컨텍스트 및 분기 추적
- **Warp-Level 실행**: 32-thread SIMT 실행과 분기 발산 처리
- **Thread 0 리더십 패턴**: **96.9% 스토리지 효율성** 달성 with warp leader 최적화
- **SM 아키텍처**: 144개 Streaming Multiprocessor, 각각 4개 warp scheduler
- **메모리 계층**: L1/L2 캐시, shared memory, register file with 실제 지연 시간

### 🖥️ H100 Hopper 아키텍처
- **4세대 Tensor Core**: FP8/FP16 혼합 정밀도 지원
- **Transformer Engine**: AI 성능 최적화를 위한 동적 정밀도 스위칭
- **80GB HBM3 메모리**: 2TB/s 대역폭 시뮬레이션
- **Thread Block Cluster**: 최대 16블록 클러스터 지원

### 🆕 B200 Blackwell 아키텍처
- **듀얼 칩렛 설계**: 2×72 SM with 10TB/s inter-chiplet interconnect
- **고급 Tensor Core**: FP4 정밀도 with 2:4 sparsity 최적화
- **SER 2.0**: 향상된 warp 스케줄링을 위한 Shader Execution Reordering
- **192GB HBM3E 메모리**: 8TB/s 대역폭 시뮬레이션

### 🧠 AI 특화 스토리지 시스템 (Thread 0 최적화)
- **KV Cache**: LLM 추론 캐시 with 압축, adaptive retention, thread 0 접근 패턴
- **Vector Database**: RAG 워크로드용 HNSW/FAISS 인덱싱 with warp-level broadcast
- **GNN Storage**: 그래프 샘플링과 이웃 쿼리 with 조정된 접근 패턴

### 🎯 GNN 워크로드 최적화 (cuGraph 통합)
- **Edge-centric Pattern**: cuGraph 스타일 완벽한 로드 밸런스
- **Adaptive Pattern Selection**: 그래프 특성 기반 자동 패턴 스위칭
- **Hybrid Access Patterns**: 스토리지 효율성과 컴퓨팅 성능 균형
- **SQ Doorbell Management**: NVMe 경합 분석 및 최적화

## 📁 프로젝트 구조

```
simpy_simulator/
├── config/                           # 설정 파일들
│   ├── default.yaml                  # 기본 설정
│   ├── gnn_benchmark.yaml            # GNN 벤치마크 설정
│   └── b200_comparison.yaml          # GPU 아키텍처 비교 설정
├── src/                              # 메인 소스 코드
│   ├── base/                         # 기본 추상 클래스들
│   ├── components/                   # 하드웨어 컴포넌트들
│   │   ├── gpu/                      # GPU 관련 (H100, B200, SM, Warp 등)
│   │   └── storage/                  # NVMe 스토리지 컴포넌트들
│   ├── workloads/                    # 워크로드 모듈들
│   │   ├── gnn/                      # GNN 특화 모듈들
│   │   └── ai_storage/               # AI 스토리지 워크로드들
│   └── utils/                        # Factory, 설정 로더, 성능 도구들
├── tests/                            # 테스트 시스템
│   ├── unit/                         # 단위 테스트
│   ├── integration/                  # 통합 테스트
│   └── benchmarks/                   # 벤치마크 테스트
├── scripts/                          # 실행 스크립트들
│   ├── run_benchmark.py              # CLI 벤치마크 실행기
│   └── test_new_structure.py         # 구조 검증 테스트
├── results/                          # 테스트 결과들
└── examples/                         # 예제 코드들
```

## 🚀 빠른 시작

### 1. 개발 환경 설정
```bash
python scripts/setup_dev_environment.py
```

### 2. 구조 검증 테스트
```bash
python scripts/test_new_structure.py
```

### 3. GNN 벤치마크 실행
```bash
# 기본 GNN 벤치마크
python scripts/run_benchmark.py --benchmark gnn

# Dry run으로 미리 확인
python scripts/run_benchmark.py --benchmark gnn --dry-run --verbose

# 특정 접근 패턴만 테스트
python scripts/run_benchmark.py --benchmark gnn --access-patterns thread_0_leader,multi_thread_parallel
```

### 4. 설정 관리
```bash
# 설정 파일 목록 확인
python scripts/run_benchmark.py --list-configs

# 설정 파일 검증
python scripts/run_benchmark.py --validate-configs

# 새 설정 템플릿 생성
python scripts/run_benchmark.py --create-template gnn_benchmark
```

### 5. GPU 아키텍처 비교
```bash
python scripts/run_benchmark.py --benchmark comparison --gpu-types H100,B200
```

## 📊 성능 결과

### Thread 0 스토리지 최적화
- **96.9% 스토리지 효율성**: 32개 스레드 중 1개만 스토리지 접근
- **21.9x 평균 접근 속도**: 기존 대비 대폭 향상된 스토리지 성능
- **0% SQ 경합률**: 32-1024 SQ 환경에서 경합 없는 접근

### cuGraph 통합 결과
- **Edge-centric 패턴**: 완벽한 로드 밸런스로 균등한 작업 분산
- **Adaptive 의사결정**: 그래프 특성(degree, sparsity, hub ratio)에 따른 자동 패턴 선택
- **Hybrid 접근법**: 스토리지 효율성과 컴퓨팅 성능의 최적 균형

### 벤치마크 검증
- **46/46 테스트 통과**: 100% 성공률로 모든 기능 검증
- **멀티 워크로드 지원**: 5-40 warp 스케일링 테스트 완료  
- **실시간 성능 분석**: JSON/CSV 리포트 자동 생성

## 🔧 CLI 사용법

### 기본 명령어
```bash
# 도움말
python scripts/run_benchmark.py --help

# GNN 벤치마크 (기본)
python scripts/run_benchmark.py --benchmark gnn

# 사용자 정의 설정으로 실행
python scripts/run_benchmark.py --config config/my_config.yaml --benchmark gnn

# 결과를 특정 디렉토리에 저장
python scripts/run_benchmark.py --benchmark gnn --output-dir my_results/
```

### 고급 옵션
```bash
# 특정 워크로드 크기로 테스트
python scripts/run_benchmark.py --benchmark gnn --workload-sizes 10,20,40

# Verbose 모드로 자세한 출력
python scripts/run_benchmark.py --benchmark gnn --verbose

# 여러 벤치마크 동시 실행
python scripts/run_benchmark.py --benchmark all
```

### 설정 관리
```bash
# 모든 설정 파일 나열
python scripts/run_benchmark.py --list-configs

# 설정 파일 유효성 검사
python scripts/run_benchmark.py --validate-configs

# 새 설정 템플릿 생성
python scripts/run_benchmark.py --create-template default
python scripts/run_benchmark.py --create-template gnn_benchmark
python scripts/run_benchmark.py --create-template gpu_comparison
```

## ⚙️ 설정 시스템

### YAML 설정 구조
```yaml
simulation:
  simulation_time: 50000
  random_seed: 42
  log_level: "INFO"

gpu:
  name: "H100"  # H100 또는 B200
  num_sms: 144
  memory_bandwidth_gbps: 2000.0

gnn:
  execution_mode: "adaptive_hybrid"  # storage_optimized, compute_optimized, adaptive_hybrid
  access_pattern:
    pattern: "hybrid_degree_based"   # thread_0_leader, multi_thread_parallel, edge_centric_cugraph
    degree_threshold: 32
    enable_thread0_optimization: true
  
benchmark:
  access_patterns: ["thread_0_leader", "multi_thread_parallel", "hybrid_degree_based"]
  workload_sizes: [5, 10, 20, 40]
  save_results: true
```

### Factory 패턴 사용
```python
from src.utils import GNNEngineFactory, SimulationFactory

# 환경 생성
env = SimulationFactory.create_environment("config/default.yaml")

# 최적화된 엔진 생성
storage_engine = GNNEngineFactory.create_storage_optimized_engine(env)
compute_engine = GNNEngineFactory.create_compute_optimized_engine(env)
adaptive_engine = GNNEngineFactory.create_adaptive_engine(env)
```

## 🧪 테스트 시스템

### 자동화된 테스트
```bash
# 전체 구조 테스트
python scripts/test_new_structure.py

# 포괄적 GNN 벤치마크
python tests/benchmarks/test_gnn_benchmark.py
```

### 테스트 범위
- **단위 테스트**: 개별 컴포넌트 기능 검증
- **통합 테스트**: 모듈 간 상호작용 테스트
- **벤치마크 테스트**: 성능 측정 및 회귀 검사

## 📈 성능 분석

### 자동 리포트 생성
- **JSON 리포트**: 상세한 성능 메트릭 및 구성 정보
- **CSV 데이터**: 스프레드시트 분석용 원시 데이터
- **콘솔 요약**: 실시간 성능 요약 및 권장사항

### 핵심 메트릭
- **스토리지 효율성**: 메모리 대역폭 사용률 최적화
- **처리량**: 초당 연산 처리 성능
- **지연 시간**: 평균 응답 시간 측정
- **SQ 경합률**: NVMe doorbell lock 경합 분석

## 🛠️ 개발 및 확장

### 새로운 워크로드 추가
1. `src/workloads/` 하위에 새 모듈 생성
2. Factory 패턴으로 객체 생성 로직 추가
3. 설정 클래스에 새 워크로드 매개변수 추가
4. 벤치마크 테스트 구현

### 새로운 GPU 아키텍처 지원
1. `src/components/gpu/`에 새 GPU 클래스 추가
2. `config/gpu_config.py`에 설정 클래스 추가
3. Factory에서 새 아키텍처 지원 추가

## 📚 기술 세부사항

### 아키텍처 정확도
- **사이클 레벨 타이밍**: 정밀한 실행 사이클 모델링
- **리소스 경합**: 하드웨어 병목 현상 정확한 모델링
- **메모리 지연시간**: 현실적인 캐시 미스 페널티 및 DRAM 접근 시간

### 확장성
- **스레드 시뮬레이션**: 최대 294,912개 동시 스레드 (144 SM × 2048 threads)
- **메모리 모델링**: 설정 가능한 크기의 다층 캐시 계층
- **워크로드 지원**: 단순한 elementwise부터 복잡한 transformer attention까지

## 🔮 향후 개발 계획

### 단기 계획
- **더 많은 워크로드 지원**: Computer Vision, NLP transformer 모델
- **시각화 도구**: 성능 분석 결과 실시간 차트 및 그래프
- **분산 시뮬레이션**: 다중 GPU 시스템 및 NVLink 패브릭 모델링

### 장기 계획  
- **전력 모델링**: 다양한 정밀도에서의 에너지 소비 분석
- **온도 시뮬레이션**: 온도 인식 성능 스로틀링 모델
- **컴파일러 통합**: CUDA 코드를 시뮬레이션 커널로 자동 변환

## 📄 라이선스 및 기여

This project demonstrates advanced GPU simulation capabilities and provides a valuable tool for GPU performance analysis and architectural research.

## 🙋‍♂️ 지원 및 문의

- **이슈 리포트**: GitHub Issues를 통한 버그 리포트 및 기능 요청
- **기술 문의**: 시뮬레이션 정확도 및 성능 최적화 관련 질문
- **기여 방법**: Pull Request를 통한 새 기능 및 개선사항 기여

---

**🎯 SimPy GPU Simulator는 현대 GPU 아키텍처의 심층적인 이해를 바탕으로 구축된 고도의 시뮬레이션 도구입니다. 모듈화된 구조와 Thread 0 스토리지 최적화를 통해 GPU 성능 분석 및 아키텍처 연구에 귀중한 도구를 제공합니다.**