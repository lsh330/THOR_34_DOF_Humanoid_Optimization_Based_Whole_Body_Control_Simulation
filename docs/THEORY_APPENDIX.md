# THOR 34-DOF 휴머노이드 — 부록 (Appendices B–G)

> 이 문서는 [THEORY.md](THEORY.md)의 부록 B–G를 포함합니다. 본문의 수식 번호(예: 식 2.14, 식 9.3)를 참조합니다.
>
> **기호 일람**은 [THEORY.md 부록 A](THEORY.md#부록-a-기호-일람-notation-summary)를 참조하십시오.

---
## 부록 B. 스큐 대칭 행렬 (Skew-Symmetric Matrix) 성질

### B.1 정의와 기본 성질

벡터 $p = [p\_{x}, p\_{y}, p\_{z}]^T$ 에 대한 스큐 대칭 행렬:

**(B.1)**

```math
[p]_{\times} = \begin{bmatrix} 0 & -p_z & p_y \\ p_z & 0 & -p_x \\ -p_y & p_x & 0 \end{bmatrix}
```

**성질 1** (반대칭): $[p]\_{\times}^T = -[p]\_{\times}$

**증명**: $(i,j)$ 원소와 $(j,i)$ 원소의 부호가 반대이므로 직접 확인 가능. $\quad \checkmark$

**성질 2** (크로스곱 표현): $p \times q = [p]\_{\times} q$

**증명**:

**(B.2)**

```math
[p]_{\times} q = \begin{bmatrix} -p_z q_y + p_y q_z \\ p_z q_x - p_x q_z \\ -p_y q_x + p_x q_y \end{bmatrix} = p \times q \quad \checkmark
```

**성질 3**: $[p]\_{\times}^2 = p p^T - \lVert p\rVert^2 I\_{3}$

**증명**: $( p \times q) \times r = (q r^T - r q^T)p$ 에서 행렬 형태로 변환하면 확인 가능. $\quad \checkmark$

**성질 4**: $[p]\_{\times} [p]\_{\times}^T = \lVert p\rVert^2 I\_{3} - p p^T$ (음의 준정치)

이로부터 공간 관성 식 (2.14)에서 $I\_{cm} + m[c]\_{\times} [c]\_{\times}^T$ 가 양정치임을 알 수 있다.

### B.2 회전 변환 하에서의 스큐 행렬

**(B.3)**

```math
[Rp]_{\times} = R[p]_{\times} R^T
```

**증명**: 임의의 $q$ 에 대해 $(Rp) \times q = R(p \times R^T q)$ 에서:

```math
[Rp]_{\times} q = R[p]_{\times} R^T q \quad \forall q \implies [Rp]_{\times} = R[p]_{\times} R^T \quad \checkmark
```

이 성질은 공간 관성의 좌표 변환 (식 2.15)에서 핵심적으로 사용된다.

---

## 부록 C. 회전 행렬의 미분 (Rotation Matrix Derivatives)

### C.1 SO(3) 위의 미분

$R(t) \in SO(3)$ 의 시간 미분:

**(C.1)**

```math
\dot{R} = [\omega]_{\times} R
```

**유도**: $R R^T = I$ 를 시간 미분하면:

```math
\dot{R} R^T + R \dot{R}^T = 0
```

```math
\dot{R} R^T = -(\dot{R} R^T)^T
```

따라서 $\dot{R} R^T$ 는 반대칭이므로 어떤 벡터 $\omega$ 에 대해 $\dot{R} R^T = [\omega]\_{\times}$ 가 성립한다. 이로부터:

**(C.2)**

```math
\dot{R} = [\omega]_{\times} R \quad \checkmark
```

### C.2 쿼터니언 미분 유도 (상세)

식 (9.3)의 완전 유도. $R(\mathbf{q}(t))$ 를 시간 미분하면 식 (C.1)이 성립해야 하므로:

**(C.3)**

```math
\frac{dR}{dt} = \frac{\partial R}{\partial \mathbf{q}} \dot{\mathbf{q}} = [\omega]_{\times} R
```

식 (1.7)의 $R$ 을 $\mathbf{q} = [w, x, y, z]^T$ 에 대해 편미분하고, $[\omega]\_{\times} R$ 와 등치 조건을 비교하면:

**(C.4a)**

```math
\dot{w} = \frac{1}{2}(-\omega_x x - \omega_y y - \omega_z z)
```

**(C.4b)**

```math
\dot{x} = \frac{1}{2}(\omega_x w + \omega_z y - \omega_y z)
```

**(C.4c)**

```math
\dot{y} = \frac{1}{2}(\omega_y w - \omega_z x + \omega_x z)
```

**(C.4d)**

```math
\dot{z} = \frac{1}{2}(\omega_z w + \omega_x y - \omega_y x)
```

이를 행렬로 표현하면:

**(C.5)**

```math
\dot{\mathbf{q}} = \frac{1}{2} \Omega(\omega) \mathbf{q}, \quad \Omega(\omega) = \begin{bmatrix} 0 & -\omega_x & -\omega_y & -\omega_z \\ \omega_x & 0 & \omega_z & -\omega_y \\ \omega_y & -\omega_z & 0 & \omega_x \\ \omega_z & \omega_y & -\omega_x & 0 \end{bmatrix}
```

이것이 식 (9.3)의 행렬 형태다. $\quad \blacksquare$

---

## 부록 D. 공간 벡터의 좌표 변환 심층 분석

### D.1 프레임 A에서 프레임 B로의 변환

프레임 B가 프레임 A에 대해 회전 $R\_{AB}$ 와 이동 $p\_{AB}$ 로 정의될 때, 프레임 A의 공간 속도 $\mathbf{v}\_{A}$ 를 프레임 B에서 표현하면:

**(D.1)**

```math
\mathbf{v}_B = X_{AB} \mathbf{v}_A
```

**(D.2)**

```math
X_{AB} = \begin{bmatrix} R_{AB} & 0 \\ -R_{AB}[p_{AB}]_{\times} & R_{AB} \end{bmatrix}
```

### D.2 변환 합성 (Transform Composition)

프레임 A → B → C 의 연속 변환:

**(D.3)**

```math
X_{AC} = X_{BC} \cdot X_{AB}
```

**증명**:

```math
\mathbf{v}_{C} = X_{BC} \mathbf{v}_{B} = X_{BC}(X_{AB} \mathbf{v}_{A}) = (X_{BC} X_{AB}) \mathbf{v}_{A}
```

$\quad \checkmark$

이는 순기구학 식 (3.4)에서 `X_world[i] = X_parent[i] @ X_world[parent]` 로 구현된다.

### D.3 힘 변환과 운동 변환의 쌍대성

운동 변환 $X$ 에 대해 힘 변환은 $X^{-T}$ 이다. 두 변환 행렬의 관계:

**(D.4)**

```math
X^{-T} = \begin{bmatrix} R & [p]_{\times} R \\ 0 & R \end{bmatrix}
```

**유도**: 식 (2.9)에서 $X^{-1}$ 을 구한 후 전치:

```math
X^{-1} = \begin{bmatrix} R^T & 0 \\ [p]_{\times} R^T & R^T \end{bmatrix}
```

전치하면:

**(D.5)**

```math
X^{-T} = (X^{-1})^T = \begin{bmatrix} R & R[p]_{\times}^T \\ 0 & R \end{bmatrix} = \begin{bmatrix} R & -R[p]_{\times} \\ 0 & R \end{bmatrix}
```

$[p]\_{\times}^T = -[p]\_{\times}$ 를 사용하면 식 (D.4)를 얻는다.

RNEA의 역전파에서 `X_up[i].T @ f[i]` 는 정확히 $X\_{up,i}^T = X\_{up,i}^{-T}$ (힘 변환의 역)를 사용함을 주목하라 — 즉 `X_up[i]`가 운동 변환이면 그 전치가 힘을 "역방향"으로 변환한다.

---

## 부록 E. 수치 구현 세부 사항

### E.1 Cholesky 분해와 선형 시스템 풀기

$M\_{jj}$ 는 대칭 양정치 행렬이므로 Cholesky 분해를 사용한다:

**(E.1)**

```math
M_{jj} = L L^T
```

여기서 $L$ 은 하삼각 행렬(Lower Triangular Matrix)이다.

시스템 $M\_{jj} x = b$ 풀기:
1. $L y = b$ 전방 대입(Forward Substitution)
2. $L^T x = y$ 후방 대입(Backward Substitution)

복잡도: $O(n^2)$ (LU 분해 $O(n^3/3)$ 대비 절반)

```python
# thor/dynamics/contact_implicit.py
from scipy.linalg import cho_factor, cho_solve
cho_jj = cho_factor(M_jj + reg_jj)  # L L^T 분해
ddq_j  = cho_solve(cho_jj, rhs_j)    # 전방/후방 대입
```

### E.2 Cholesky 캐시 전략

질량 행렬 $M\_{jj}$ 는 설정 $q$ 에 의존하지만, 느리게 변화한다. 매 스텝마다 재계산하는 비용을 줄이기 위해 조건부 캐시를 사용한다:

```python
# thor/dynamics/contact_implicit.py
def _get_cached_cho(buf, M_jj, threshold=1e-3):
    if buf["cho_valid"]:
        rel = ||M_jj - M_jj_ref|| / ||M_jj_ref||
        if rel < threshold:   # 변화 작으면 캐시 재사용
            return buf["cho_cache"]
    cho = cho_factor(M_jj + reg)
    # 캐시 갱신
    buf["M_jj_ref"] = M_jj
    buf["cho_cache"] = cho
```

임계값 `threshold=1e-3` 은 0.1% 이상 변화 시 재계산함을 의미한다.

### E.3 JIT 컴파일 전략 (Numba)

병목 함수에 Numba JIT를 적용하되, 실패 시 순수 Python으로 폴백한다:

```python
# thor/dynamics/rnea.py
_USE_JIT = True

def rnea(model, q, v, a, f_ext=None):
    if _USE_JIT and f_ext is None:
        try:
            from .rnea_jit import rnea_jit
            return rnea_jit(...)      # Numba 경로
        except Exception:
            pass
    return _rnea_python(model, q, v, a, f_ext)  # Python 폴백
```

동일 패턴이 `crba.py`, `kinematics.py` 에도 적용된다.

### E.4 접촉 감지 임계값

```python
# thor/dynamics/contact_implicit.py
if p_foot[2] < 0.05:   # 5 cm 이내 → 접촉 판정
    contact_feet.append(fid)
```

스프링-댐퍼 모델에서는 `phi >= 0` 이면 힘이 0이므로, 5 cm 임계값은 접촉 제어 로직 진입을 위한 별도 판정이다.

### E.5 토크 한계 (Torque Limits)

각 관절의 최대 토크는 `link.tau_max` 에 저장되며 다음과 같이 클리핑한다:

```python
# thor/control/walking_controller.py
lim = self._model.links[i+1].tau_max
tau[6+i] = np.clip(tau[6+i], -lim, lim)
```

대표 토크 한계:
- 엉덩이/무릎 (SEA): 289 N·m
- 발목: 150 N·m
- 팔: 20–60 N·m
- 손목/손가락: 10–20 N·m

---

## 부록 F. 알고리즘 수렴성 및 안정성 분석

### F.1 RNEA/CRBA/ABA 수치 안정성

세 알고리즘 모두 순수 행렬 곱셈과 덧셈으로 구성되므로 수치적으로 안정적이다. 단, 질량 행렬 $M$ 의 조건수(Condition Number)가 클 경우 선형 시스템 풀기가 불안정해질 수 있다.

THOR에서 조건 악화 원인:
- 질량 비 최대: $m\_{pelvis}/m\_{finger} \approx 10.6/0.3 \approx 35$
- 링크 길이 비: 상체(0.3m) vs 허벅지(0.35m)

정규화 항 추가로 완화: `M_jj += 1e-10 * np.eye(n_j)`

### F.2 LCP 수렴 조건

Fischer-Burmeister Newton 방법은 M이 **P-행렬(P-matrix)**일 때 전역 수렴이 보장된다.

**P-행렬 정의**: 모든 주부분행렬(Principal Submatrix)의 행렬식이 양수.

Delassus 행렬 $M\_{LCP} = J\_{n} M^{-1} J\_{n}^T$ 의 경우:
- $M \succ 0$ 이고 $J\_{n}$ 이 행 독립 → $M\_{LCP} \succ 0$ → P-행렬 조건 만족
- 단, 접촉점이 기하학적으로 종속이면 $J\_{n}$ 이 행 종속 → 정규화 필요

### F.3 보행 제어의 안정성

계산 토크 제어 (8.2절)에서 오차 동역학 (식 8.7)의 안정성:

특성 다항식 $s^2 + K\_{d} s + K\_{p} = 0$ 의 근:

**(F.1)**

```math
s_{1,2} = \frac{-K_d \pm \sqrt{K_d^2 - 4K_p}}{2}
```

THOR 다리 관절 ($K\_{p}=600$, $K\_{d}=60$):

**(F.2)**

```math
K_d^2 - 4K_p = 3600 - 2400 = 1200 > 0
```

두 근 모두 실수이고 음수이므로 **과감쇠(Overdamped)** 수렴. 진동 없이 지수적 감소.

**(F.3)**

```math
s_{1,2} = \frac{-60 \pm \sqrt{1200}}{2} \approx \frac{-60 \pm 34.6}{2} = \{-12.7,\; -47.3\}
```

시정수: $\tau\_{1} = 1/12.7 \approx 79$ ms, $\tau\_{2} = 1/47.3 \approx 21$ ms.

---

## 부록 G. 물리 파라미터 전체 목록

### G.1 로봇 전체 파라미터

| 파라미터 | 값 | 단위 |
|:---|:---:|:---:|
| 전체 질량 $M$ | 65.0 | kg |
| 키 | 1.78 | m |
| DOF | 40 | — |
| 몸체 수 | 35 | — |
| 회전 관절 수 | 34 | — |
| 최대 다리 토크 | 289 | N·m |

### G.2 시뮬레이션 파라미터

| 파라미터 | 값 | 단위 | 의미 |
|:---|:---:|:---:|:---|
| `DEFAULT_DT` | 0.001 | s | 역학 루프 스텝 (1 kHz) |
| `MPC_DT` | 0.02 | s | MPC 스텝 (50 Hz) |
| `CONTACT_STIFFNESS` | 3000 | N/m | 접촉 법선 강성 |
| `CONTACT_DAMPING` | 300 | N·s/m | 접촉 법선 감쇠 |
| `MU_DEFAULT` | 0.7 | — | 마찰 계수 |
| `STICTION_VEL` | 0.01 | m/s | 정지 임계 속도 |
| `MAX_CONTACT_FORCE` | 2000 | N | 접촉력 상한 |

### G.3 접촉 기하학 파라미터

| 파라미터 | 값 | 단위 |
|:---|:---:|:---:|
| 발 길이 | 0.22 | m |
| 발 너비 | 0.10 | m |
| 접촉점 수 (발당) | 4 | — |
| 총 접촉점 수 | 8 | — |
| 접촉 감지 임계값 | 0.05 | m |

### G.4 보행 제어 파라미터

| 파라미터 | 값 | 단위 |
|:---|:---:|:---:|
| 스윙 지속 시간 $T\_{swing}$ | 0.55 | s |
| 이중 지지 시간 $T\_{DS}$ | 0.25 | s |
| 보행 사이클 $T\_{cycle}$ | 0.80 | s |
| 엉덩이 신전각 $\theta\_{ext}$ | -5 | ° |
| 엉덩이 굴곡각 $\theta\_{flex}$ | 20 | ° |
| 무릎 스윙각 $\theta\_{knee,swing}$ | 45 | ° |
| 발목 스윙각 $\theta\_{ankle,swing}$ | 5 | ° |
| 다리 관절 $K\_{p}$ | 600 | N·m/rad |
| 다리 관절 $K\_{d}$ | 60 | N·m·s/rad |

---
