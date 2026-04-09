# 변경 이력 (Changelog)

이 프로젝트의 모든 주요 변경 사항을 기록한다.
형식: [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)
버전 관리: [Semantic Versioning](https://semver.org/lang/ko/)

---

## [1.1.0] - 2026-04-09

### Added

#### 성능 (Performance)
- **Numba JIT 가속** — RNEA 37.8×, CRBA 47.9×, FK 55.0× 속도 향상
  - 순수 역학 연산 속도: ~600 Hz → **7408 Hz**
  - 전체 시뮬레이션 속도: ~300 Hz → **~3840 Hz** (12.8× 실시간)
  - 구현 파일: `thor/dynamics/rnea_jit.py`, `thor/dynamics/crba_jit.py`, `thor/dynamics/aba_jit.py`, `thor/model/kinematics_jit.py`

#### 아키텍처 (Architecture)
- **DynamicsFacade** — 통합 동역학 API (Facade 패턴): JIT/Python 디스패치, 버퍼 관리, Cholesky 캐시를 단일 인터페이스로 통합 (`thor/dynamics/facade.py`)
- **DynamicsBuffers** — 사전 할당 작업 버퍼: 반복적인 `np.zeros` 메모리 할당 제거 (`thor/dynamics/buffers.py`)
- **CholeskyCache** — Cholesky 조건부 캐시: 양발 지지 시 질량 행렬 변화 감지 후 조건부 재분해 (DynamicsFacade 내장)
- **Strategy 패턴 (Integrators)** — 교체 가능한 수치 적분기 (`SemiImplicitEuler`, `RungeKutta4`) (`thor/dynamics/integrators.py`)
- **Observer 패턴 (Observers)** — 시뮬레이션 데이터 수집기 (`TrajectoryRecorder`, `CoMRecorder`, `PrintObserver`) (`thor/simulation/observers.py`)
- **Mediator 패턴 (SimulationMediator)** — 시뮬레이션 루프 조율자: 동역학 엔진, 제어기, 적분기, 옵저버를 조합 가능한 형태로 통합 (`thor/simulation/mediator.py`)
- **ThorConfig YAML 설정 관리자** — 7개 섹션 (simulation, gait, control, ci_mpc, contact, visualization, output) 불변 frozen 데이터클래스, 부분 설정 파일 지원 (`thor/config/config_manager.py`)
- **RobotState Named Indexing** — `QIndex`/`VIndex` 명명 슬라이스로 가독성 높은 상태 벡터 접근 (`thor/model/state.py`)
- **OutputManager** — 출력 디렉터리 자동 생성, NPZ 궤적 저장, JSON 메타데이터 관리 (`thor/pipeline/output_manager.py`)

#### 테스트 (Tests)
- `test_energy_momentum_conservation.py` — 에너지/운동량 보존 검증 (25 tests)
- `test_constraint_residual.py` — 제약 조건 잔차 검증 (15 tests)
- `test_analytical_solutions.py` — 해석적 솔루션 비교 (15 tests)
- `test_numerical_stability.py` — 수치 안정성/엣지 케이스 검증 (10 tests)
- `test_corner_cases.py` — 경계 사례 검증 (10 tests)

#### 시각화 (Visualization)
- `thor/visualization/publication_plots.py` — 300 DPI 출판 품질 플롯 10종 생성기
- `thor/visualization/walking_animation.py` — 향상된 보행 GIF (접촉력 벡터, CoM 궤적 오버레이 포함)
- `scripts/generate_all_plots.py` — 마스터 플롯 생성 스크립트 (모든 10종 플롯 일괄 생성)
- `output/plots/` 디렉터리: 01~10번 출판 플롯 (`01_standing_analysis.png` ~ `10_controller_comparison.png`)

#### 문서 (Documentation)
- `docs/THEORY.md` — 수학적 이론 참조 문서 (한국어, 증명 생략 없음): 공간 벡터 대수, Featherstone 알고리즘, LCP 유도, 제어 이론 완전 유도
- `CHANGELOG.md` — 본 변경 이력 문서
- `README.md` 한국어 번역/재구성 — 최신 성능 수치, 새 아키텍처, 248 테스트 반영

### Changed

#### 성능 (Performance)
- `contact_implicit_step`에서 Python FK → JIT FK 사용: 0.41ms → 0.007ms (58×)
- Cholesky 조건부 캐시 도입: 양발 지지 구간에서 재분해 ~90% 절약
- 정규화 행렬 사전 할당: `np.eye` 반복 동적 할당 제거
- `rnea_jit`에 `gravity_z` 파라미터 추가: 무중력 환경 테스트 및 부분 중력 시뮬레이션 지원

#### 테스트 (Tests)
- 총 테스트 수: **137 → 248** (80% 증가)
- 5개 신규 검증 카테고리 추가 (에너지 보존, 제약 잔차, 해석해 비교, 수치 안정성, 경계 사례)

#### 의존성 (Dependencies)
- `numba` 버전 상한 완화: `<0.61` → `<0.65` (최신 Numba 버전 호환성 확장)

### Fixed

- `contact_implicit_step`에서 불필요한 `np.copy` 3개 제거 (메모리 절약 및 미세 성능 개선)

---

## [1.0.0] - 2026-04-08

### Added

#### 핵심 동역학 (Core Dynamics)
- **Featherstone O(N) 알고리즘 처음부터 구현** — Pinocchio/Drake/MuJoCo 없이 순수 Python/NumPy
  - RNEA (Recursive Newton-Euler Algorithm): O(N) 역동역학
  - CRBA (Composite Rigid Body Algorithm): O(Nd) 질량 행렬
  - ABA (Articulated Body Algorithm): O(N) 전진 동역학
- **Featherstone 공간 벡터 대수** — rotation, transform, inertia, cross_product, motion_subspace 5모듈
- **THOR 34-DOF 운동학적 모델** — 35체, 40-DOF 부유 기저, 총 67.2 kg
- **중심 운동량 행렬 (Centroidal Momentum Matrix)** (Orin et al. 2013)

#### 접촉 역학 (Contact Dynamics)
- **LCP 기반 접촉-내재적 시간 스테핑** (Stewart-Trinkle 1996)
  - 8면 다각형 Coulomb 마찰 원뿔 근사 (오차 < 4%)
  - Signorini 상보성 조건 자동 처리
- **Fischer-Burmeister NCP 솔버** — 역추적 선탐색을 가진 감쇠 Newton 반복법
- **내부점법 (Interior-Point) LCP 솔버** — 대안 솔버
- **스프링-댐퍼 접촉 모델** (Marhefka & Orin 1999)

#### 제어 (Control)
- **접촉-내재적 MPC (CI-MPC)** (Le Cleac'h et al. 2024) — 정적 자세 유지
- **계산 토크 제어 (Computed Torque Control)** — 6보 연속 보행
- **Schur 보완 기저 소거** — 40×40 → 34×34 관절 시스템, 기저 결합 불안정성 해결
- **Winter (1991) 생체역학적 보행 궤적 생성기**
  - 고관절 피치: 정현파 굴곡/신전 프로파일
  - 무릎 피치: $\sin^{0.8}$ 비대칭 벨 (스윙 ~40% 피크)
  - 발목 피치: 배측굴곡 (발 지면 여유)
  - 코사인 블렌드 이중 지지 전환 ($C^0$ 연속)
- **중심 LQR (LIPM 기반 CoM 조절)** (Kajita et al. 2003)
- **전신 QP (Whole-Body QP)** 역동역학 (Escande et al. 2014)
- **관절 PD + 중력 보상**

#### 검증 (Verification)
- **CRBA-RNEA 교차 검증**: $2.27 \times 10^{-13}$ N·m (기계 정밀도)
- **137 tests, 16개 모듈** — 공간 대수, 동역학, 접촉, 보행, 성능 검증
- 12패널 시각적 증거 대시보드 (`test_evidence.png`)

#### 시각화 (Visualization)
- 막대 그림(stick figure) 2D 렌더러 + GIF 애니메이션 생성기
- CI-MPC 직립 분석 6패널 플롯
- 보행 동역학 4패널 분석 플롯
- 질량 행렬 구조 분석 (히트맵, 고유값, 대각 원소)
- 에너지 보존, 지면 반력, 위상 공간 그림, CoM 궤적 플롯
- 성능 벤치마크 타이밍 플롯

#### 결과 (Results)
- CI-MPC 직립 자세: CoM 표준편차 **1.57 mm** (3초, dt=2ms)
- 보행: 6보 연속, 5.1초, 0.95 m 전진, 0.19 m/s
- 고관절 피치 범위: −5° ~ +20.5° (Winter 1991 일치)
- 무릎 스윙 굴곡: +36.5° (생체역학적 범위)
- 시뮬레이션 속도: ~300 Hz (시각화 포함), ~2000 Hz (순수 역학)

---

[1.1.0]: https://github.com/lsh330/THOR_34_DOF_Humanoid_Optimization_Based_Whole_Body_Control_Simulation/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/lsh330/THOR_34_DOF_Humanoid_Optimization_Based_Whole_Body_Control_Simulation/releases/tag/v1.0.0
