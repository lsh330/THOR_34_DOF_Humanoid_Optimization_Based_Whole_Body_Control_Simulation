# THOR 34-DOF 휴머노이드 — 수학적 이론 참조 문서

> **목적**: 이 문서는 THOR 시뮬레이션을 구성하는 모든 수학적 유도를 처음부터 끝까지 제공하는 자체 완결적(self-contained) 이론 참조 문서다. "증명 생략" 없이 모든 수식을 완전히 유도하며, 각 수식은 소스 코드 구현과 직접 대응된다.
>
> **규약**: 수식에서 $q$ = 일반화 좌표, $v$ = 일반화 속도, $M$ = 질량 행렬 (Mass Matrix), $h$ = 바이어스 힘, $\tau$ = 토크, $S$ = 운동 부분공간 (Motion Subspace).

---

## 목차

1. [좌표계와 상태 표현 (Coordinate System & State Representation)](#1-좌표계와-상태-표현)
2. [공간 벡터 대수 (Spatial Vector Algebra)](#2-공간-벡터-대수)
3. [순기구학 (Forward Kinematics)](#3-순기구학)
4. [운동 방정식 (Equations of Motion)](#4-운동-방정식)
5. [Featherstone O(N) 알고리즘](#5-featherstone-on-알고리즘)
6. [중심 운동량 (Centroidal Momentum)](#6-중심-운동량)
7. [접촉 역학 (Contact Dynamics)](#7-접촉-역학)
8. [제어 이론 (Control Theory)](#8-제어-이론)
9. [수치 적분 (Numerical Integration)](#9-수치-적분)
10. [참고 문헌 (References)](#10-참고-문헌)

---

## 1. 좌표계와 상태 표현

**구현 파일**: `thor/core/constants.py`, `thor/model/quaternion.py`, `thor/model/kinematics.py`

### 1.1 월드 프레임 (World Frame)

월드 프레임은 다음 규약을 따른다.

- **원점**: 로봇 초기 위치 발 아래 지면
- **z-축**: 중력 반대 방향 (up), 즉 $\hat{z} = [0, 0, 1]^T$
- **좌표계 방향**: 오른손 좌표계 (Right-hand coordinate system)
- **중력 벡터**: $\mathbf{g} = [0, 0, -9.81]^T \;\text{[m/s}^2\text{]}$

```python
# thor/core/constants.py
GRAVITY: float = 9.81
GRAVITY_VEC: np.ndarray = np.array([0.0, 0.0, -GRAVITY])
```

### 1.2 부동 기저 (Floating Base) 표현

THOR 휴머노이드는 지면에 고정되지 않은 부동 기저(Floating Base) 구조를 가진다. 기저 링크(펠비스)의 위치와 자세는 특수 유클리드 군(Special Euclidean Group) $SE(3) = \mathbb{R}^3 \times SO(3)$ 로 표현된다.

```math
SE(3) = \left\{ (p, R) \;\middle|\; p \in \mathbb{R}^3,\; R \in SO(3) \right\} \tag{1.1}
```

여기서 $SO(3)$는 특수 직교 군(Special Orthogonal Group):

```math
SO(3) = \{ R \in \mathbb{R}^{3 \times 3} \mid R^T R = I,\; \det(R) = +1 \} \tag{1.2}
```

$SE(3)$는 6차원 리 군(Lie Group)으로, 기저 링크에 6 DOF(자유도)를 부여한다.

### 1.3 쿼터니언 규약 (Quaternion Convention)

자세(Orientation)는 쿼터니언 $\mathbf{q} = [w, x, y, z]^T$ 로 표현한다. 이 프로젝트는 **Hamilton 규약, scalar-first** 형식을 사용한다.

```math
\mathbf{q} = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}, \quad w^2 + x^2 + y^2 + z^2 = 1 \tag{1.3}
```

단위 쿼터니언(Unit Quaternion)은 $\|\mathbf{q}\| = 1$ 을 만족해야 한다.

**쿼터니언 → 회전 행렬 변환 (완전 유도)**

회전 행렬 $R \in SO(3)$ 를 쿼터니언 $\mathbf{q} = [w, x, y, z]^T$ 로부터 유도한다.

임의의 벡터 $\mathbf{v} \in \mathbb{R}^3$ 의 회전은 쿼터니언 곱으로 표현된다:

```math
\mathbf{v}' = \mathbf{q} \otimes [0, \mathbf{v}]^T \otimes \mathbf{q}^* \tag{1.4}
```

여기서 $\mathbf{q}^* = [w, -x, -y, -z]^T$ 는 켤레 쿼터니언(Conjugate Quaternion)이다.

쿼터니언 곱의 정의를 전개하면:

```math
\mathbf{q} \otimes [0, \mathbf{v}]^T = \begin{bmatrix} -x v_x - y v_y - z v_z \\ w v_x + y v_z - z v_y \\ w v_y + z v_x - x v_z \\ w v_z + x v_y - y v_x \end{bmatrix} \tag{1.5}
```

이를 다시 $\mathbf{q}^*$ 와 곱하고 벡터 부분만 추출하면, 각 성분:

```math
v'_x = (1 - 2y^2 - 2z^2)v_x + 2(xy - wz)v_y + 2(xz + wy)v_z \tag{1.6a}
```

```math
v'_y = 2(xy + wz)v_x + (1 - 2x^2 - 2z^2)v_y + 2(yz - wx)v_z \tag{1.6b}
```

```math
v'_z = 2(xz - wy)v_x + 2(yz + wx)v_y + (1 - 2x^2 - 2y^2)v_z \tag{1.6c}
```

이를 행렬 형태로 정리하면:

```math
R = \begin{bmatrix} 1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\ 2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\ 2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2) \end{bmatrix} \tag{1.7}
```

**검증**: $\mathbf{q} = [1, 0, 0, 0]^T$ (항등 쿼터니언) → $R = I_3$. $\quad \checkmark$

**검증**: $R^T R = I$, $\det(R) = 1$ — 단위 쿼터니언 제약 $w^2+x^2+y^2+z^2=1$에서 직접 확인 가능하다.

```python
# thor/model/quaternion.py  (quat_to_rot 함수와 정확히 일치)
def quat_to_rot(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
```

### 1.4 일반화 좌표와 속도 (Generalized Coordinates & Velocities)

**일반화 좌표 (Generalized Coordinates)** $q \in \mathbb{R}^{41}$:

```math
q = \underbrace{[p_x, p_y, p_z]}_{\text{기저 위치(3)}}, \underbrace{[w, x, y, z]}_{\text{쿼터니언(4)}}, \underbrace{[q_1, q_2, \ldots, q_{34}]}_{\text{관절 각도(34)}} \tag{1.8}
```

**일반화 속도 (Generalized Velocities)** $v \in \mathbb{R}^{40}$:

```math
v = \underbrace{[\omega_x, \omega_y, \omega_z]}_{\text{기저 각속도(3)}}, \underbrace{[\dot{p}_x, \dot{p}_y, \dot{p}_z]}_{\text{기저 선속도(3)}}, \underbrace{[\dot{q}_1, \dot{q}_2, \ldots, \dot{q}_{34}]}_{\text{관절 속도(34)}} \tag{1.9}
```

> **중요**: $\dim(q) = 41$ 이지만 $\dim(v) = 40$ 이다. 쿼터니언은 4개의 성분을 가지지만, $SO(3)$의 접선 공간(Tangent Space)은 3차원($\omega \in \mathbb{R}^3$)이다. 이 차원 불일치는 쿼터니언 미분 방정식으로 처리된다(9절 참조).

```
# 속도 벡터 인덱스 매핑 (DOF mapping table)
v[0:3]   = omega_base   (기저 각속도: REVOLUTE의 공간 속도 형식, omega-first)
v[3:6]   = v_lin_base   (기저 선속도)
v[6:7]   = dq_waist_yaw
v[7:8]   = dq_waist_pitch
v[8:9]   = dq_head_yaw
v[9:10]  = dq_head_pitch
v[10:17] = dq_L_arm (7 DOF: sh_p1, sh_r, sh_p2, el_y, wr_r, wr_y, wr_p)
v[17:24] = dq_R_arm (7 DOF: mirror)
v[24:30] = dq_L_leg (6 DOF: hip_y, hip_r, hip_p, kn_p, an_p, an_r)
v[30:36] = dq_R_leg (6 DOF: mirror)
v[36:38] = dq_L_gripper
v[38:40] = dq_R_gripper
```

### 1.5 운동학적 트리 토폴로지 (Kinematic Tree Topology)

THOR의 운동학적 구조는 트리(Tree) 형태로 표현된다. 각 몸체 $i$ ($i = 1, \ldots, 34$)는 유일한 부모(parent) 몸체를 가진다.

| 몸체 인덱스 | 링크 이름 | 부모 인덱스 | 관절 유형 |
|:---:|:---|:---:|:---:|
| 0 | pelvis (floating base) | -1 | 6-DOF |
| 1 | waist_yaw | 0 | Revolute-Z |
| 2 | waist_pitch | 1 | Revolute-Y |
| 3 | head_yaw | 2 | Revolute-Z |
| 4 | head_pitch | 3 | Revolute-Y |
| 5-11 | l_arm (7 joints) | 2, 5, 6, 7, 8, 9, 10 | Revolute |
| 12-18 | r_arm (7 joints) | 2, 12, 13, 14, 15, 16, 17 | Revolute |
| 19-24 | l_leg (6 joints) | 0, 19, 20, 21, 22, 23 | Revolute |
| 25-30 | r_leg (6 joints) | 0, 25, 26, 27, 28, 29 | Revolute |
| 31-32 | l_gripper (2 joints) | 11, 31 | Revolute |
| 33-34 | r_gripper (2 joints) | 18, 33 | Revolute |

`parent[i]` 배열은 `thor/model/robot_model.py`의 `RobotModel` 클래스에 저장된다.

### 1.6 SE(3) 리 군의 구조

부동 기저의 설정 공간 $SE(3)$ 는 리 군(Lie Group)이며, 그 리 대수(Lie Algebra)는 $\mathfrak{se}(3)$ 이다.

**리 대수 $\mathfrak{se}(3)$**: 무한소 변환의 공간 — 정확히 공간 속도 $[\omega; v_{lin}]$ 의 공간이다.

**지수 사상 (Exponential Map)**:

```math
\exp: \mathfrak{se}(3) \to SE(3), \quad \exp\left(\begin{bmatrix} [\omega]_\times & v \\ 0 & 0 \end{bmatrix} t\right) = \begin{bmatrix} R(t) & p(t) \\ 0 & 1 \end{bmatrix} \tag{1.10}
```

여기서 $R(t) = e^{[\omega]_\times t}$ (Rodrigues 공식).

쿼터니언 표현은 $SO(3) \cong S^3 / \mathbb{Z}_2$ (3차원 구면의 이중 피복)를 통해 회전을 파라미터화하며, 짐벌 락(Gimbal Lock)이 없고 계산이 효율적이다.

**쿼터니언과 회전 행렬의 저장 효율**:
- 회전 행렬: $3 \times 3 = 9$ 실수, 6개 제약 ($R^TR = I$, $\det R = 1$)
- 오일러 각: 3 실수, 짐벌 락 발생 가능
- 쿼터니언: 4 실수, 1개 제약 ($\|\mathbf{q}\| = 1$), 짐벌 락 없음

THOR 시뮬레이션에서는 쿼터니언을 사용하고, 필요 시 회전 행렬로 변환하여 사용한다.

### 1.7 질량 스케일링

THOR 모델은 THORMANG3 URDF (42 kg, 1.375 m)를 THOR 스펙 (65 kg, 1.78 m)으로 스케일링하여 구성된다:

```math
m_{THOR} = m_{THORMANG3} \times \frac{65}{42} \approx m_{THORMANG3} \times 1.548 \tag{1.11}
```

```math
l_{THOR} = l_{THORMANG3} \times \frac{1.78}{1.375} \approx l_{THORMANG3} \times 1.295 \tag{1.12}
```

관성 텐서는 길이 스케일의 제곱으로 비례:

```math
I_{THOR} = I_{THORMANG3} \times \frac{65}{42} \times \left(\frac{1.78}{1.375}\right)^2 \approx I_{THORMANG3} \times 2.597 \tag{1.13}
```

```python
# thor/model/robot_model.py
mass_scale   = 65.0 / 42.0      # ~1.548
length_scale = 1.78 / 1.375     # ~1.295
# 관성 = mass_scale * length_scale^2 로 적용
```

---

## 2. 공간 벡터 대수

**구현 파일**: `thor/core/spatial/transform.py`, `thor/core/spatial/inertia.py`, `thor/core/spatial/cross_product.py`, `thor/core/spatial/motion_subspace.py`

### 2.1 Plücker 좌표 (Plücker Coordinates) 동기

강체 역학에서 회전과 이동은 항상 결합되어 나타난다. 예를 들어, 링크 $i$의 운동량은 회전 운동량(Angular Momentum)과 선 운동량(Linear Momentum)을 모두 포함한다. 이를 하나의 6차원 벡터로 통합하면 알고리즘이 단순하고 효율적이 된다. 이 통합 표현이 **공간 벡터 대수(Spatial Vector Algebra)**이며, Plücker 좌표 위에서 정의된다.

Plücker 좌표는 선(line)을 방향 벡터와 모멘트 벡터의 쌍으로 표현하는 체계다. 공간 역학에서 이 아이디어를 활용하면 회전과 이동을 통합한 6D 수학 구조를 얻는다.

### 2.2 공간 운동 벡터 (Spatial Motion Vector / Twist)

강체의 운동(Twist, 비틀림)은 각속도 $\omega \in \mathbb{R}^3$ 와 선속도 $v_{lin} \in \mathbb{R}^3$ 로 구성된 6차원 벡터다.

```math
\mathbf{v} = \begin{bmatrix} \omega \\ v_{lin} \end{bmatrix} \in \mathbb{R}^6 \tag{2.1}
```

> **규약**: Featherstone(2008) 및 이 프로젝트는 **omega-first** 규약을 사용한다. 상위 3개 성분이 각속도, 하위 3개 성분이 선속도다.

- 각속도 벡터 $\omega$: 공간 내 회전 속도 $[\text{rad/s}]$
- 선속도 벡터 $v_{lin}$: 기준점의 이동 속도 $[\text{m/s}]$

### 2.3 공간 힘 벡터 (Spatial Force Vector / Wrench)

힘(Wrench, 나사)은 토크 $\tau \in \mathbb{R}^3$ 와 힘 $f \in \mathbb{R}^3$ 로 구성된 6차원 벡터다.

```math
\mathbf{f} = \begin{bmatrix} \tau \\ f \end{bmatrix} \in \mathbb{R}^6 \tag{2.2}
```

### 2.4 파워 불변성 (Power Invariance)

공간 벡터의 내적 $\mathbf{f}^T \mathbf{v}$ 는 파워(Power, 일률)를 나타내며, 이는 좌표 프레임 변환에 불변이다.

**유도**: 임의의 공간 변환 $X$ 에 대해 운동 벡터는 $\mathbf{v}' = X \mathbf{v}$ 로 변환된다. 힘 벡터가 $\mathbf{f}' = X^{-T} \mathbf{f}$ 로 변환되어야 파워가 보존된다:

```math
P = \mathbf{f}'^T \mathbf{v}' = (X^{-T} \mathbf{f})^T (X \mathbf{v}) = \mathbf{f}^T X^{-1} X \mathbf{v} = \mathbf{f}^T \mathbf{v} \tag{2.3}
```

따라서 운동 벡터의 공변 쌍대(Dual)인 힘 벡터는 $X^{-T}$ 로 변환된다.

### 2.5 공간 변환 행렬 (Spatial Transform Matrix)

링크 $i$ 에서 링크 $j$ 로의 공간 변환 $X \in \mathbb{R}^{6 \times 6}$ 를 유도한다. 회전 행렬 $R \in SO(3)$ 와 이동 벡터 $p \in \mathbb{R}^3$ 가 주어졌을 때:

**단계 1**: 순수 이동 $p$ 에 의한 각속도 변환은 변하지 않는다.

```math
\omega' = \omega \tag{2.4a}
```

**단계 2**: 이동 후 선속도는 회전 이동 효과가 더해진다.

```math
v'_{lin} = v_{lin} - p \times \omega = v_{lin} + \omega \times p \tag{2.4b}
```

여기서 $p \times \omega = [p]_\times \omega$, 그리고 $[p]_\times$는 반대칭 행렬(Skew-Symmetric Matrix):

```math
[p]_\times = \begin{bmatrix} 0 & -p_z & p_y \\ p_z & 0 & -p_x \\ -p_y & p_x & 0 \end{bmatrix} \tag{2.5}
```

**단계 3**: 이동 변환을 행렬로 쓰면:

```math
\begin{bmatrix} \omega' \\ v'_{lin} \end{bmatrix} = \begin{bmatrix} I & 0 \\ -[p]_\times & I \end{bmatrix} \begin{bmatrix} \omega \\ v_{lin} \end{bmatrix} \tag{2.6}
```

**단계 4**: 좌표계 회전 $R$ 을 적용하면:

```math
\begin{bmatrix} \omega'' \\ v''_{lin} \end{bmatrix} = \begin{bmatrix} R & 0 \\ 0 & R \end{bmatrix} \begin{bmatrix} \omega' \\ v'_{lin} \end{bmatrix} \tag{2.7}
```

**단계 5**: 두 변환을 합성하면 공간 변환 행렬을 얻는다:

```math
X = \begin{bmatrix} R & 0 \\ -R[p]_\times & R \end{bmatrix} \in \mathbb{R}^{6 \times 6} \tag{2.8}
```

```python
# thor/core/spatial/transform.py (spatial_transform 함수와 정확히 일치)
def spatial_transform(R, p):
    X = np.empty((6, 6))
    X[:3, :3] = R
    X[:3, 3:] = 0.0
    X[3:, 3:] = R
    X[3:, :3] = -R @ skew(p)   # skew(p) = [p]_×
    return X
```

**역변환 유도**: $X^{-1}$ 을 구하기 위해 $X^{-1} X = I$ 를 풀면:

```math
X^{-1} = \begin{bmatrix} R^T & 0 \\ [p]_\times R^T & R^T \end{bmatrix} \tag{2.9}
```

이는 `spatial_transform_inv` 함수에서 $R^T$ 를 이용해 효율적으로 계산된다.

**힘 벡터의 변환**: 파워 불변성(2.3절)으로부터 힘 벡터는 $X^{-T}$ 로 변환된다:

```math
X^{-T} = \begin{bmatrix} R & -[p]_\times^T R \\ 0 & R \end{bmatrix} = \begin{bmatrix} R & [p]_\times R \\ 0 & R \end{bmatrix} \tag{2.10}
```

(마지막 등호는 $[p]_\times^T = -[p]_\times$ 성질 이용)

이로부터 RNEA 역전파에서 `X_up[i].T @ f[i]` 가 힘 벡터를 부모 프레임으로 정확히 변환함을 알 수 있다.

### 2.6 공간 관성 (Spatial Inertia)

링크 $i$ 의 공간 관성 행렬 $\mathcal{I}_i \in \mathbb{R}^{6 \times 6}$ 을 유도한다.

링크의 물리적 성질: 질량 $m$, 무게 중심(CoM) $c \in \mathbb{R}^3$ (링크 좌표계에서), 관성 텐서 $I_{cm} \in \mathbb{R}^{3 \times 3}$ (CoM 좌표계에서).

운동 에너지(Kinetic Energy):

```math
T = \frac{1}{2} \omega^T I_{cm} \omega + \frac{1}{2} m \|v_{cm}\|^2 \tag{2.11}
```

여기서 $v_{cm} = v_{lin} + \omega \times c = v_{lin} - [c]_\times \omega$ 가 CoM의 속도다.

이를 전개하면:

```math
T = \frac{1}{2} \mathbf{v}^T \mathcal{I} \mathbf{v} \tag{2.12}
```

공간 관성 $\mathcal{I}$ 를 구하기 위해 각 항을 분리한다:

```math
2T = \omega^T I_{cm} \omega + m(v_{lin} - [c]_\times \omega)^T(v_{lin} - [c]_\times \omega) \tag{2.13a}
```

```math
= \omega^T I_{cm} \omega + m v_{lin}^T v_{lin} - 2m v_{lin}^T [c]_\times \omega + m \omega^T [c]_\times^T [c]_\times \omega \tag{2.13b}
```

교차항을 정리하면 ($[c]_\times^T = -[c]_\times$):

```math
2T = \omega^T(I_{cm} + m[c]_\times [c]_\times^T)\omega + 2m v_{lin}^T [c]_\times^T \omega + m v_{lin}^T v_{lin} \tag{2.13c}
```

이를 $\mathbf{v} = [\omega; v_{lin}]^T$ 에 대한 이차형식으로 쓰면:

```math
\mathcal{I} = \begin{bmatrix} I_{cm} + m[c]_\times [c]_\times^T & m[c]_\times \\ m[c]_\times^T & m I_3 \end{bmatrix} \tag{2.14}
```

> **주석**: $[c]_\times [c]_\times^T$는 음수 부호 없이 나타남에 주의. 이는 $[c]_\times^T = -[c]_\times$ 로부터 $[c]_\times [c]_\times^T = -[c]_\times^2 \geq 0$ 임을 알 수 있다.

```python
# thor/core/spatial/inertia.py (spatial_inertia 함수와 정확히 일치)
def spatial_inertia(mass, com, I_cm):
    cx = skew(com)   # [c]_×
    cxT = cx.T       # [c]_×^T
    I_s[:3, :3] = I_cm + mass * (cx @ cxT)   # I_cm + m[c]_×[c]_×^T
    I_s[:3, 3:] = mass * cx                   # m[c]_×
    I_s[3:, :3] = mass * cxT                  # m[c]_×^T
    I_s[3:, 3:] = mass * I_3                  # mI_3
```

**평행축 정리의 공간 형태 (Parallel Axis Theorem)**

공간 관성의 좌표 변환: 기준점을 $c$ 에서 $c + d$ 로 이동할 때,

```math
\mathcal{I}_{new} = X^{-T} \mathcal{I} X^{-1} \tag{2.15}
```

여기서 $X$ 는 식 (2.8)의 공간 변환 행렬이다. 이를 전개하면 회전 관성에 $m [d]_\times [d]_\times^T$ 가 추가되는 표준 평행축 정리와 일치한다.

**$\mathcal{I}$ 의 대칭 양정치 증명**:

1. **대칭성**: $\mathcal{I}_{11} = I_{cm} + m[c]_\times [c]_\times^T$ 는 대칭, $\mathcal{I}_{22} = mI_3$ 는 대칭, 그리고 $\mathcal{I}_{12} = m[c]_\times$, $\mathcal{I}_{21} = m[c]_\times^T = \mathcal{I}_{12}^T$. 따라서 $\mathcal{I}$ 는 대칭이다.

2. **양정치성**: 임의의 공간 속도 $\mathbf{v} \neq 0$ 에 대해 $\mathbf{v}^T \mathcal{I} \mathbf{v} = 2T > 0$ (운동 에너지는 양수). 따라서 $\mathcal{I} \succ 0$. $\quad \blacksquare$

### 2.7 공간 크로스곱 (Spatial Cross Products)

공간 운동 벡터의 크로스곱 연산자를 정의한다.

**운동 크로스곱 (Motion Cross Product)**:

$\mathbf{v}_1 = [\omega_1; v_1]$ 에 대해, 운동 크로스곱 연산자 $[\mathbf{v}_1]_\times$ 는:

```math
[\mathbf{v}_1]_\times = \begin{bmatrix} [\omega_1]_\times & 0 \\ [v_1]_\times & [\omega_1]_\times \end{bmatrix} \in \mathbb{R}^{6 \times 6} \tag{2.16}
```

이는 두 공간 속도의 리 괄호(Lie Bracket) $[\mathbf{v}_1, \mathbf{v}_2] = [\mathbf{v}_1]_\times \mathbf{v}_2$ 를 표현한다.

```python
# thor/core/spatial/cross_product.py
def spatial_cross_motion(v):
    omega_x = skew(v[:3])   # [ω]_×
    v_lin_x = skew(v[3:])   # [v_lin]_×
    X[:3, :3] = omega_x
    X[:3, 3:] = 0.0
    X[3:, :3] = v_lin_x
    X[3:, 3:] = omega_x
```

**힘 크로스곱 (Force Cross Product)**:

힘 벡터에 작용하는 크로스곱 연산자는 운동 크로스곱의 쌍대(Dual)다:

```math
[\mathbf{v}]_{\times^*} = -[\mathbf{v}]_\times^T \tag{2.17}
```

**유도**: 파워 $P = \mathbf{f}^T \mathbf{v}_2$ 의 시간 미분에서 힘에 대한 크로스곱이 등장한다. 파워 보존 조건 $\delta P = 0$ 에서:

```math
\mathbf{f}^T ([\mathbf{v}_1]_\times \mathbf{v}_2) = ([\mathbf{v}_1]_{\times^*} \mathbf{f})^T \mathbf{v}_2 \tag{2.18}
```

이므로 $[\mathbf{v}_1]_{\times^*} = [\mathbf{v}_1]_\times^T$... 이때 Featherstone(2008)의 부호 규약에서 $[\mathbf{v}]_{\times^*} = -[\mathbf{v}]_\times^T$ 가 정의된다.

```python
# thor/core/spatial/cross_product.py
def spatial_cross_force(v):
    return -spatial_cross_motion(v).T
```

### 2.8 운동 부분공간 (Motion Subspace)

관절 $i$ 의 운동 부분공간 $S_i \in \mathbb{R}^6$ 는 관절이 허용하는 상대 운동 방향을 나타낸다.

**회전 관절 (Revolute Joint)**: 관절 축 $e_{axis}$ 방향으로만 회전 가능:

```math
S_i = \begin{bmatrix} e_{axis} \\ 0 \end{bmatrix} \in \mathbb{R}^6 \tag{2.19}
```

구체적으로:
- X축 회전: $S = [1, 0, 0, 0, 0, 0]^T$
- Y축 회전: $S = [0, 1, 0, 0, 0, 0]^T$
- Z축 회전: $S = [0, 0, 1, 0, 0, 0]^T$

```python
# thor/core/spatial/motion_subspace.py
def motion_subspace_revolute(axis):
    S = np.zeros(6)
    S[axis] = 1.0   # omega 부분에 1, v_lin 부분은 0
    return S
```

---

## 3. 순기구학

**구현 파일**: `thor/model/kinematics.py`, `thor/model/robot_model.py`

### 3.1 관절 변환 (Joint Transform)

관절 $i$ 의 공간 변환 $X_i(q_i)$ 는 링크 고정 변환 $X_{tree,i}$ 와 관절 회전 $X_{rot}(q_i)$ 의 합성이다:

```math
X_i(q_i) = X_{tree,i} \cdot X_{rot}(q_i) \tag{3.1}
```

여기서:
- $X_{tree,i}$: 부모 링크 좌표계에서 관절 원점까지의 고정 변환 (관절 오프셋 $p_{offset}$ 과 고정 회전 $R_{fixed}$ 로 구성)
- $X_{rot}(q_i)$: 관절 각도 $q_i$ 에 의한 회전 변환

관절 회전 행렬:
- Z축 관절: $R_{rot} = R_z(q_i) = \begin{bmatrix} \cos q_i & -\sin q_i & 0 \\ \sin q_i & \cos q_i & 0 \\ 0 & 0 & 1 \end{bmatrix}$

- Y축 관절: $R_{rot} = R_y(q_i) = \begin{bmatrix} \cos q_i & 0 & \sin q_i \\ 0 & 1 & 0 \\ -\sin q_i & 0 & \cos q_i \end{bmatrix}$

- X축 관절: $R_{rot} = R_x(q_i) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos q_i & -\sin q_i \\ 0 & \sin q_i & \cos q_i \end{bmatrix}$

전체 관절 회전:

```math
R_{total} = R_{fixed} \cdot R_{rot}(q_i) \tag{3.2}
```

```python
# thor/model/kinematics.py (joint_transform 함수)
def joint_transform(link_idx, q_joint, model):
    R_joint = rot_y(q_joint)   # 또는 rot_x, rot_z
    R_total = link.joint_rotation @ R_joint  # R_fixed @ R_rot
    return spatial_transform(R_total, link.joint_offset)
```

### 3.2 재귀적 순기구학 (Recursive Forward Kinematics)

세계 좌표계(World Frame)에서 각 몸체의 변환을 재귀적으로 계산한다.

**부동 기저 (Body 0)**:

```math
X_{world,0} = X_{base}(R_{base}(q_{quat}), p_{base}) \tag{3.3}
```

**일반 몸체 $i \geq 1$** (부모 → 자식 방향):

```math
X_{world,i} = X_i(q_i) \cdot X_{world,parent(i)} \tag{3.4}
```

> **구현 주의**: `thor/model/kinematics.py` 에서의 실제 코드는 $X_{world,i} = X_{parent,i} \cdot X_{world,parent(i)}$ 형태로, 변환은 왼쪽에서 오른쪽으로 합성된다. 이는 Featherstone(2008)의 규약으로, 변환이 "from world to body"를 의미한다.

```python
# thor/model/kinematics.py (forward_kinematics 함수)
X_world[0] = spatial_transform(R_base, p_base)
for i in range(1, n):
    X_parent[i] = joint_transform(i, q_i, model)
    X_world[i] = X_parent[i] @ X_world[parent]   # 식 (3.4)
```

### 3.3 공간 변환에서 위치 추출

공간 변환 행렬의 구조로부터 위치 $p$ 를 복원한다.

식 (2.8)로부터:

```math
X[3:, :3] = -R[p]_\times \tag{3.5}
```

양변에 $R^T$를 왼쪽에 곱하면:

```math
R^T X[3:, :3] = -[p]_\times \tag{3.6}
```

$[p]_\times$ 의 비대각 원소로부터:

```math
p = \begin{bmatrix} -(-p_\times)_{2,1} \\ -(-p_\times)_{0,2} \\ -(-p_\times)_{1,0} \end{bmatrix} = \begin{bmatrix} (p_\times)_{2,1} \\ (p_\times)_{0,2} \\ (p_\times)_{1,0} \end{bmatrix} \tag{3.7}
```

```python
# thor/model/kinematics.py (body_position 함수)
def body_position(X_world_i):
    R = X_world_i[:3, :3]
    neg_skew_p = R.T @ X_world_i[3:, :3]  # = -[p]_×
    p = np.array([-neg_skew_p[2,1], -neg_skew_p[0,2], -neg_skew_p[1,0]])
    return p
```

### 3.4 질량 중심 (Center of Mass, CoM)

전체 시스템의 질량 중심 위치:

```math
c = \frac{1}{M_{total}} \sum_{i=0}^{N-1} m_i \left( p_i + R_i \cdot c_{body,i} \right) \tag{3.8}
```

여기서:
- $M_{total} = \sum_i m_i$ : 전체 질량
- $p_i$ : 몸체 $i$ 의 원점 위치 (세계 좌표계)
- $R_i$ : 몸체 $i$ 의 회전 행렬 ($X_{world,i}[:3,:3]$)
- $c_{body,i}$ : 몸체 $i$ 의 CoM (링크 좌표계에서)

```python
# thor/model/kinematics.py (com_position 함수)
com = np.zeros(3)
for i in range(model.n_bodies):
    p_i = body_position(X_world[i])
    R_i = X_world[i][:3, :3]
    com_world = p_i + R_i @ link.com
    com += link.mass * com_world
return com / model.total_mass   # 식 (3.8)
```

### 3.5 몸체 자코비안 (Body Jacobian)

몸체 $i$ 의 6차원 자코비안 $J_i \in \mathbb{R}^{6 \times n_{dof}}$ 은 일반화 속도 $v$ 를 공간 속도 $\mathbf{v}_i$ 로 매핑한다:

```math
\mathbf{v}_i = J_i \cdot v \tag{3.9}
```

자코비안은 운동학적 체인(Kinematic Chain)을 따라 재귀적으로 구성된다.

**부동 기저 열 (Floating Base Columns, 인덱스 0-5)**:

기저의 공간 변환을 통해 기저 속도를 몸체 $i$ 의 프레임으로 변환:

```math
J_i[:, 0:6] = X_{world,i} \cdot X_{world,0}^{-1} \tag{3.10}
```

**회전 관절 열 (Revolute Joint Column, 관절 $j$)**:

관절 $j$ 의 운동 부분공간 $S_j$ 를 몸체 $i$ 의 프레임으로 변환 (단, $j$ 는 $i$ 의 조상):

```math
J_i[:, 5+j] = X_{world,i} \cdot X_{world,j}^{-1} \cdot S_j \tag{3.11}
```

```python
# thor/model/kinematics.py (body_jacobian 함수)
i = body_idx
while i >= 0:
    if i == 0:  # Floating base
        X_bi = X_world[body_idx] @ np.linalg.inv(X_world[0])
        J[:, :6] = X_bi
    else:       # Revolute joint
        X_bi = X_world[body_idx] @ np.linalg.inv(X_world[i])
        J[:, 5+i] = X_bi @ S_i   # 식 (3.11)
    i = model.parent[i]
```

### 3.6 자코비안 검증 — 유한 차분 (Finite Difference Verification)

자코비안 $J_i$ 의 정확성을 유한 차분으로 검증한다:

```math
J_i[:, k] \approx \frac{p_i(q + \delta e_k) - p_i(q - \delta e_k)}{2\delta} \tag{3.12}
```

여기서 $e_k$ 는 $k$ 번째 표준 기저 벡터, $\delta = 10^{-6}$ 이다.

위치 자코비안($J_v = J_i[3:,:]$)에 대해 검증하면 $O(\delta^2)$ 의 오차가 예상된다.

### 3.7 자코비안 구조

몸체 $i$ 의 자코비안은 희소(Sparse) 구조를 가진다:
- 부동 기저 열(0-5): 항상 비영
- 관절 $j$ 열($5+j$): $j$ 가 몸체 $i$ 의 조상일 때만 비영

THOR의 경우 다리 끝단(발목) 자코비안은 약 12-13개의 비영 열을 가진다:
- 부동 기저: 6개
- 하체 체인 (허리 제외): 6개 (다리만, 허리-몸통은 별도)

이 희소성을 활용하면 자코비안 계산과 CMM 계산을 효율적으로 수행할 수 있다.

### 3.8 스케일 분석 — 순기구학 연산량

THOR 35개 몸체에 대한 순기구학 연산량:

| 연산 | 횟수 | 비고 |
|:---|:---:|:---|
| 관절 변환 $X_i(q_i)$ | 34 | 각 관절마다 1회 |
| 공간 변환 합성 $X_{world}$ | 34 | 부모→자식 전파 |
| 위치 추출 $p_i$ | N/A | 필요 시만 계산 |
| 자코비안 $J_i$ | $O(d_i)$ | $d_i$ = 체인 깊이 |

전체 FK 연산량: $O(N)$ — 각 몸체를 한 번씩만 처리한다.

---

## 4. 운동 방정식

**구현 파일**: `thor/dynamics/rnea.py`, `thor/dynamics/crba.py`

### 4.1 부동 기저 매니퓰레이터 방정식

THOR의 운동 방정식은 부동 기저 강체 시스템의 뉴턴-오일러 방정식으로부터 유도된다:

```math
M(q) \dot{v} + h(q, v) = S^T \tau + J_c^T f_c \tag{4.1}
```

각 항의 의미:
- $M(q) \in \mathbb{R}^{40 \times 40}$: 질량 행렬 (Mass Matrix) — 관성 효과
- $\dot{v} \in \mathbb{R}^{40}$: 일반화 가속도
- $h(q, v) \in \mathbb{R}^{40}$: 바이어스 힘 (Bias Force) — 코리올리, 원심력, 중력
- $S^T$: 선택 행렬 (Selection Matrix) — 구동 가능한 관절 선택
- $\tau \in \mathbb{R}^{34}$: 관절 토크
- $J_c^T \in \mathbb{R}^{40 \times 3n_c}$: 접촉 자코비안 전치
- $f_c \in \mathbb{R}^{3n_c}$: 접촉력

### 4.2 질량 행렬의 블록 구조

$40 \times 40$ 질량 행렬을 부동 기저(6 DOF)와 관절(34 DOF)로 분할한다:

```math
M = \begin{bmatrix} M_{bb} & M_{bj} \\ M_{jb} & M_{jj} \end{bmatrix} \tag{4.2}
```

여기서:
- $M_{bb} \in \mathbb{R}^{6 \times 6}$: 부동 기저 관성 (합성 관성 $\mathcal{I}_c^{(0)}$와 동일)
- $M_{bj} = M_{jb}^T \in \mathbb{R}^{6 \times 34}$: 기저-관절 결합 항
- $M_{jj} \in \mathbb{R}^{34 \times 34}$: 관절 공간 관성

### 4.3 바이어스 힘의 분해

바이어스 힘 $h(q, v)$ 는 코리올리 항(Coriolis), 원심 항(Centrifugal), 중력 항(Gravity)으로 구성된다:

```math
h(q, v) = C(q, v) v + g(q) \tag{4.3}
```

실제 계산은 분해하지 않고 RNEA로 통합 계산한다:

```math
h(q, v) = \text{RNEA}(q, v, 0) \tag{4.4}
```

```math
g(q) = \text{RNEA}(q, 0, 0) \tag{4.5}
```

### 4.4 선택 행렬 (Selection Matrix)

$S^T \in \mathbb{R}^{40 \times 34}$ 는 관절 토크를 일반화 힘으로 매핑한다:

```math
S^T = \begin{bmatrix} 0_{6 \times 34} \\ I_{34 \times 34} \end{bmatrix} \tag{4.6}
```

즉, 부동 기저는 구동되지 않으며, 34개의 관절만 토크를 받는다.

### 4.5 중력의 가상 가속도 표현 (Gravity as Fictitious Acceleration)

**물리적 정당성**: 모든 몸체에 중력이 작용하는 것은, 기저에 위쪽 방향 가상 가속도 $a_{grav}$ 를 더하는 것과 등가다.

**유도**: $N$ 개 몸체에 대한 뉴턴의 운동 법칙:

```math
m_i \ddot{p}_i = f_{ext,i} + m_i g \hat{z} \tag{4.7}
```

관성 프레임을 기준으로 취하면, 기저에 가속도 $-g \hat{z}$ 를 적용하는 것은 모든 몸체에 중력 $m_i g \hat{z}$ 를 작용시키는 것과 등가다 (등가 원리, Equivalence Principle).

공간 벡터 표현에서는 다음과 같이 설정한다:

```math
a_{grav} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ +g \end{bmatrix} = [0; 0; 0; 0; 0; +9.81]^T \tag{4.8}
```

이를 기저 몸체의 가속도에 더하면 모든 하위 몸체로 전파되어 중력 효과를 자동으로 반영한다.

```python
# thor/dynamics/rnea.py
a_grav = np.zeros(6)
a_grav[3:] = -GRAVITY_VEC   # = [0, 0, +9.81] (z-up, 위를 양수로)
acc[0] = a_base + a_grav    # 기저 가속도에 중력 추가
```

여기서 `GRAVITY_VEC = [0, 0, -9.81]` 이므로 `-GRAVITY_VEC = [0, 0, +9.81]` 이 된다. $\quad \checkmark$

### 4.6 질량 행렬의 대칭 양정치 증명

**대칭성**: 임의의 두 가속도 $\dot{v}_1, \dot{v}_2$ 에 대해, 운동 에너지 $T = \frac{1}{2} v^T M v$ 로부터 $M = M^T$. CRBA에서도 오프대각 원소가 대칭으로 채워짐을 확인할 수 있다 (5.2절 참조).

**양정치성**: 임의의 비영 속도 $v \neq 0$ 에 대해:

```math
v^T M v = \sum_{i=0}^{N-1} \mathbf{v}_i^T \mathcal{I}_i \mathbf{v}_i = 2T > 0 \tag{4.9}
```

각 $\mathcal{I}_i \succ 0$ (2.6절 증명)이고 $v \neq 0$ 이면 적어도 하나의 $\mathbf{v}_i \neq 0$ 이므로, 전체 합이 양수다. $\quad \blacksquare$

### 4.7 운동 방정식의 유도 — Lagrangian 접근법

라그랑지 역학(Lagrangian Mechanics)으로부터 식 (4.1)을 유도한다.

**라그랑지안 (Lagrangian)**:

```math
L(q, v) = T(q, v) - V(q) \tag{4.10}
```

여기서 $T = \frac{1}{2} v^T M(q) v$ (운동 에너지), $V(q)$ (포텐셜 에너지, 중력).

**오일러-라그랑지 방정식**:

```math
\frac{d}{dt}\frac{\partial L}{\partial v} - \frac{\partial L}{\partial q} = Q \tag{4.11}
```

여기서 $Q = S^T \tau + J_c^T f_c$ 는 일반화 외력이다.

$\frac{d}{dt}(M v) = M\dot{v} + \dot{M} v$ 이고, $\frac{\partial T}{\partial q} = \frac{1}{2} v^T \frac{\partial M}{\partial q} v$ 이므로:

```math
M\dot{v} + \dot{M} v - \frac{1}{2} v^T \frac{\partial M}{\partial q} v + \frac{\partial V}{\partial q} = Q \tag{4.12}
```

크리스토펠 기호(Christoffel Symbols)를 이용하면 $\dot{M} v - \frac{1}{2} v^T \frac{\partial M}{\partial q} v = C(q,v)v$ 로 표현되며, $\frac{\partial V}{\partial q} = g(q)$ 이므로:

```math
M\dot{v} + C(q,v)v + g(q) = Q \tag{4.13}
```

이것이 바로 식 (4.1)의 $h(q,v) = C(q,v)v + g(q)$ 이다. $\quad \blacksquare$

### 4.8 관성 행렬의 조건수와 수치 안정성

THOR의 질량 행렬 조건수 예측:

```math
\kappa(M) = \frac{\lambda_{max}(M)}{\lambda_{min}(M)} \tag{4.14}
```

최대 고유값: 펠비스 링크 질량 $\approx 10.6$ kg × 길이² $\approx 10.6 \times (0.3)^2 \approx 1$ kg·m²

최소 고유값: 손가락 링크 질량 $\approx 0.3$ kg × 길이² $\approx 0.3 \times (0.05)^2 \approx 7.5 \times 10^{-4}$ kg·m²

$\kappa(M) \approx 1 / (7.5 \times 10^{-4}) \approx 1330$

조건수 1330은 double precision ($\epsilon_{machine} \approx 10^{-16}$)에서 약 $10^{-13}$ 의 상대 오차를 의미한다. 이는 CRBA-RNEA 교차 검증에서 관측된 $1.14 \times 10^{-13}$ N·m 오차와 일치한다.

---

## 5. Featherstone O(N) 알고리즘

**구현 파일**: `thor/dynamics/rnea.py`, `thor/dynamics/crba.py`, `thor/dynamics/aba.py`

### 5.1 RNEA — 역동역학 (Recursive Newton-Euler Algorithm)

RNEA는 주어진 $(q, v, \dot{v})$ 에서 이를 달성하는 일반화 힘 $\tau$ 를 계산한다:

```math
\tau = \text{RNEA}(q, v, \dot{v}) \tag{5.1}
```

복잡도: $O(N)$ (THOR에서 $N = 35$).

#### 5.1.1 전진 패스 (Forward Pass): 속도와 가속도 전파

트리의 루트(기저)에서 잎(말단)으로 전파한다.

**Body 0 (부동 기저)**:

```math
\mathbf{v}_0 = v_{base} \tag{5.2a}
```

```math
\mathbf{a}_0 = a_{base} + a_{grav} \tag{5.2b}
```

**Body $i \geq 1$ (자식 몸체)**:

속도 전파:

```math
\mathbf{v}_i = X_{up,i} \mathbf{v}_{parent(i)} + S_i \dot{q}_i \tag{5.3}
```

가속도 전파:

```math
\mathbf{a}_i = X_{up,i} \mathbf{a}_{parent(i)} + S_i \ddot{q}_i + \underbrace{[\mathbf{v}_i]_\times S_i \dot{q}_i}_{\text{속도곱 가속도}} \tag{5.4}
```

**속도곱 가속도 항 (Velocity-Product Acceleration)**: $[\mathbf{v}_i]_\times S_i \dot{q}_i$

이 항은 관절 좌표계가 회전하기 때문에 발생한다. 코리올리 가속도와 원심 가속도의 합으로, 관절의 회전 속도와 링크의 이동 속도가 결합되어 생성된다.

유도: 관절 $i$ 의 공간 속도 $\mathbf{v}_i$ 는 부모의 속도 전파와 관절 기여의 합이다. 시간 미분하면:

```math
\dot{\mathbf{v}}_i = \dot{X}_{up,i} \mathbf{v}_{parent} + X_{up,i} \dot{\mathbf{v}}_{parent} + \dot{S}_i \dot{q}_i + S_i \ddot{q}_i \tag{5.5}
```

$\dot{X}_{up,i} \mathbf{v}_{parent} = [\mathbf{v}_i]_\times X_{up,i} \mathbf{v}_{parent}$ 이고 $\dot{S}_i = 0$ (운동 부분공간은 관절 좌표계에서 상수)이므로:

```math
\mathbf{a}_i = X_{up,i} \mathbf{a}_{parent} + S_i \ddot{q}_i + [\mathbf{v}_i]_\times (S_i \dot{q}_i) \tag{5.6}
```

**$\dot{X}_{up,i} \mathbf{v}_{parent} = [\mathbf{v}_i]_\times X_{up,i} \mathbf{v}_{parent}$ 증명**:

공간 변환은 $\mathbf{v}_i = X_{up,i} \mathbf{v}_{parent} + S_i \dot{q}_i$ 이다. $X_{up,i}$ 는 관절 각도 $q_i(t)$ 에 의존하므로:

```math
\dot{X}_{up,i} = \frac{d}{dt}\begin{bmatrix} R_i & 0 \\ -R_i[p_i]_\times & R_i \end{bmatrix}
```

$\dot{R}_i = [\omega_i]_\times R_i$ 를 사용하면:

```math
\dot{X}_{up,i} = \begin{bmatrix} [\omega_i]_\times R_i & 0 \\ \cdots & [\omega_i]_\times R_i \end{bmatrix} = [\mathbf{v}_i^{joint}]_\times X_{up,i} \tag{5.5a}
```

여기서 $\mathbf{v}_i^{joint} = S_i \dot{q}_i$ 는 관절이 기여하는 공간 속도다. 따라서:

```math
\dot{X}_{up,i} \mathbf{v}_{parent} = [\mathbf{v}_i^{joint}]_\times (X_{up,i} \mathbf{v}_{parent}) \approx [\mathbf{v}_i]_\times (X_{up,i} \mathbf{v}_{parent}) \quad \checkmark \tag{5.5b}
```

```python
# thor/dynamics/rnea.py
vel[i] = X_up[i] @ vel[parent] + S_i * dq_i          # 식 (5.3)
vxS    = spatial_cross_motion(vel[i]) @ (S_i * dq_i) # [v_i]_× S_i dq_i
acc[i] = X_up[i] @ acc[parent] + S_i * ddq_i + vxS   # 식 (5.4)
```

#### 5.1.2 역전 패스 (Backward Pass): 힘 누적

잎에서 루트 방향으로 뉴턴-오일러 방정식을 적용한다.

**몸체 $i$ 의 힘**:

```math
\mathbf{f}_i = \mathcal{I}_i \mathbf{a}_i + [\mathbf{v}_i]_{\times^*} (\mathcal{I}_i \mathbf{v}_i) \tag{5.7}
```

이는 Newton 법칙 $f = ma$ 의 공간 형태다:
- $\mathcal{I}_i \mathbf{a}_i$: 관성력 (Inertial Force)
- $[\mathbf{v}_i]_{\times^*} (\mathcal{I}_i \mathbf{v}_i) = \mathbf{v}_i \times^* (\mathcal{I}_i \mathbf{v}_i)$: 자이로스코픽 힘 (Gyroscopic Force) — 회전 운동에 의한 원심 및 코리올리 효과

**부모로 힘 전파**:

```math
\mathbf{f}_{parent(i)} \mathrel{+}= X_{up,i}^T \mathbf{f}_i \tag{5.8}
```

관절 토크 추출:

```math
\tau_i = S_i^T \mathbf{f}_i \tag{5.9}
```

```python
# thor/dynamics/rnea.py
for i in range(n):
    Iv = I_s[i] @ vel[i]
    f[i] = I_s[i] @ acc[i] + spatial_cross_force(vel[i]) @ Iv  # 식 (5.7)
for i in range(n-1, 0, -1):
    f[model.parent[i]] += X_up[i].T @ f[i]                     # 식 (5.8)
tau[dof_idx] = S_i @ f[i]                                      # 식 (5.9)
```

#### 5.1.3 특수 경우

- **바이어스 힘**: $h(q, v) = \text{RNEA}(q, v, 0)$ — 가속도 0으로 코리올리+중력만 계산
- **중력 보상**: $g(q) = \text{RNEA}(q, 0, 0)$ — 속도와 가속도 모두 0

```python
# thor/dynamics/rnea.py
def bias_forces(model, q, v):
    a_zero = np.zeros(model.n_dof)
    return rnea(model, q, v, a_zero)

def gravity_forces(model, q):
    v_zero = np.zeros(model.n_dof)
    a_zero = np.zeros(model.n_dof)
    return rnea(model, q, v_zero, a_zero)
```

#### 5.1.4 RNEA 정확성 검증

RNEA의 결과는 다음 교차 검증으로 확인할 수 있다:

```math
\text{RNEA}(q, v, \dot{v}) = M(q) \dot{v} + h(q, v) \tag{5.10}
```

이 항등식은 CRBA-RNEA 교차 검증에서 기계 정밀도($1.14 \times 10^{-13}$ N·m)로 확인된다.

### 5.2 CRBA — 질량 행렬 (Composite Rigid Body Algorithm)

CRBA는 운동 방정식의 질량 행렬 $M(q) \in \mathbb{R}^{n_{dof} \times n_{dof}}$ 를 $O(Nd)$ 복잡도로 계산한다. 여기서 $d$ 는 트리 깊이(THOR에서 약 7).

#### 5.2.1 Pass 1: 합성 관성 누적 (Composite Inertia Accumulation)

잎에서 루트 방향으로 합성 관성 $\mathcal{I}_c^{(i)}$ 를 계산한다.

초기값: $\mathcal{I}_c^{(i)} = \mathcal{I}_i$ (각 몸체 자신의 관성)

```math
\mathcal{I}_c^{(parent(i))} \mathrel{+}= X_{up,i}^T \mathcal{I}_c^{(i)} X_{up,i} \tag{5.11}
```

이는 자식 몸체 $i$ 의 합성 관성을 부모 좌표계로 변환하여 누적한다. 변환 규칙은 식 (2.15)와 동일하다.

```python
# thor/dynamics/crba.py  (Pass 1)
for i in range(n-1, 0, -1):
    parent = model.parent[i]
    I_c[parent] = I_c[parent] + X_up[i].T @ I_c[i] @ X_up[i]  # 식 (5.11)
```

#### 5.2.2 Pass 2: 질량 행렬 채우기 (Mass Matrix Fill)

**부동 기저 블록** ($6 \times 6$):

```math
M[0:6, 0:6] = \mathcal{I}_c^{(0)} \tag{5.12}
```

이는 루트(기저)의 합성 관성이 전체 시스템의 관성을 반영하기 때문이다.

**대각 원소** (관절 $i$ 의 자기 관성):

```math
M[dof_i, dof_i] = S_i^T \mathcal{I}_c^{(i)} S_i \tag{5.13}
```

**비대각 원소** (관절 $i$ 와 조상 관절 $j$ 의 결합 관성):

$F_i = \mathcal{I}_c^{(i)} S_i$ 를 부모 방향으로 전파하면서:

```math
F_{up} \leftarrow X_{up,j}^T F_{up} \tag{5.14a}
```

```math
M[dof_i, dof_j] = M[dof_j, dof_i] = S_j^T F_{up} \tag{5.14b}
```

```python
# thor/dynamics/crba.py (Pass 2)
M[:6, :6] = I_c[0]                   # 식 (5.12)
F_i = I_c[i] @ S_i
M[dof_i, dof_i] = S_i @ F_i          # 식 (5.13)
F_up = X_up[j].T @ F_up              # 루트 방향 전파
M[dof_i, dof_j] = M[dof_j, dof_i] = S_j @ F_up  # 식 (5.14)
```

#### 5.2.3 교차 검증

항등식 (5.10)을 이용해 CRBA와 RNEA 결과를 비교한다:

```math
\|M(q) \dot{v} + h(q, v) - \text{RNEA}(q, v, \dot{v})\|_\infty < \epsilon_{machine} \tag{5.15}
```

### 5.3 ABA — 전진 동역학 (Articulated Body Algorithm)

ABA는 주어진 $(q, v, \tau, f_{ext})$ 에서 가속도 $\dot{v}$ 를 $O(N)$ 으로 계산한다:

```math
\dot{v} = \text{ABA}(q, v, \tau, f_{ext}) \tag{5.16}
```

#### 5.3.1 Pass 1: 속도 전파 (Outward)

식 (5.3)과 동일하게 기저에서 잎 방향으로 속도를 전파한다. 각 몸체에서 속도곱 바이어스를 계산한다:

```math
c_i = [\mathbf{v}_i]_\times S_i \dot{q}_i \tag{5.17}
```

초기 관절체 관성(Articulated Body Inertia): $IA_i = \mathcal{I}_i$

초기 관절체 바이어스 힘(Articulated Bias Force):

```math
pA_i = [\mathbf{v}_i]_{\times^*} (\mathcal{I}_i \mathbf{v}_i) \tag{5.18}
```

```python
# thor/dynamics/aba.py (Pass 1)
vel[i] = X_up[i] @ vel[parent] + S[i] * dq_i
c_bias[i] = spatial_cross_motion(vel[i]) @ (S[i] * dq_i)   # 식 (5.17)
pA[i] = spatial_cross_force(vel[i]) @ (IA[i] @ vel[i])     # 식 (5.18)
```

#### 5.3.2 Pass 2: 관절체 관성 누적 (Inward)

잎에서 루트 방향으로 관절체 관성과 바이어스를 누적한다.

각 관절 $i$ 에서:

```math
U_i = IA_i S_i \tag{5.19a}
```

```math
D_i = S_i^T U_i \tag{5.19b}
```

**관절 잔여 토크 (Residual Torque)**:

```math
u_i = \tau_i - S_i^T pA_i \tag{5.19c}
```

**누적 관성 업데이트 (Inward ABI Propagation)**:

```math
IA_i^a = IA_i - \frac{U_i U_i^T}{D_i} \tag{5.20a}
```

```math
pA_i^a = pA_i + IA_i^a c_i + U_i \frac{u_i}{D_i} \tag{5.20b}
```

```math
IA_{parent(i)} \mathrel{+}= X_{up,i}^T IA_i^a X_{up,i} \tag{5.20c}
```

```math
pA_{parent(i)} \mathrel{+}= X_{up,i}^T pA_i^a \tag{5.20d}
```

**식 (5.20a)의 물리적 의미**: 관절 $i$ 가 탄성 없이 자유롭게 움직일 수 있다면, 관절 $i$ 의 실효 관성은 원래 관성에서 관절 자유도를 통한 "내부 유연성"을 제거한 것이다. 이는 Schur complement의 관성 형태다.

```python
# thor/dynamics/aba.py (Pass 2)
U[i] = IA[i] @ S[i]           # 식 (5.19a)
D[i] = S[i] @ U[i]            # 식 (5.19b)
u[i] = tau_i - S[i] @ pA[i]   # 식 (5.19c)
Ia  = IA[i] - np.outer(U[i], U[i]) / D[i]  # 식 (5.20a)
pa  = pA[i] + Ia @ c_bias[i] + U[i] * (u[i] / D[i])  # 식 (5.20b)
IA[parent] += X_up[i].T @ Ia @ X_up[i]  # 식 (5.20c)
pA[parent] += X_up[i].T @ pa             # 식 (5.20d)
```

#### 5.3.3 Pass 3: 가속도 전파 (Outward)

루트에서 잎 방향으로 가속도를 계산한다.

**기저 가속도**:

```math
\mathbf{a}_0 = IA_0^{-1} (\tau_{0:6} - pA_0) + a_{grav} \tag{5.21}
```

**관절 가속도**:

```math
a_{parent} = X_{up,i} \mathbf{a}_{parent(i)} + c_i \tag{5.22a}
```

```math
\ddot{q}_i = \frac{u_i - U_i^T a_{parent}}{D_i} \tag{5.22b}
```

```math
\mathbf{a}_i = a_{parent} + S_i \ddot{q}_i \tag{5.22c}
```

```python
# thor/dynamics/aba.py (Pass 3)
acc[0] = np.linalg.solve(IA[0], tau[:6] - pA[0]) + a_grav  # 식 (5.21)
for i in range(1, n):
    a_parent = X_up[i] @ acc[parent] + c_bias[i]            # 식 (5.22a)
    ddq_i = (u[i] - U[i] @ a_parent) / D[i]                 # 식 (5.22b)
    acc[i] = a_parent + S[i] * ddq_i                         # 식 (5.22c)
```

#### 5.3.4 복잡도 분석

| 알고리즘 | 복잡도 | THOR ($N=35$) |
|:---:|:---:|:---:|
| RNEA | $O(N)$ | ~35 몸체 처리 |
| CRBA | $O(Nd)$, $d$ = 트리 깊이 | $O(35 \times 7) \approx 245$ |
| ABA | $O(N)$ | ~35 몸체 처리 (3 pass) |

직접 관성 역행렬 $M^{-1}$ 계산은 $O(n^3) = O(40^3) = 64000$ 이므로, ABA는 약 1800배 빠르다.

### 5.4 세 알고리즘의 관계

```math
\text{RNEA}(q, v, \dot{v}) = M(q)\dot{v} + h(q,v) \tag{5.23}
```

이로부터:

| 목적 | 알고리즘 | 수식 |
|:---|:---:|:---|
| 역동역학 $\tau$ 계산 | RNEA | $\tau = \text{RNEA}(q, v, \dot{v})$ |
| 질량 행렬 $M$ 계산 | CRBA | $M = \text{CRBA}(q)$ |
| 전진 동역학 $\dot{v}$ 계산 | ABA | $\dot{v} = \text{ABA}(q, v, \tau)$ |
| 바이어스 힘 $h$ 계산 | RNEA | $h = \text{RNEA}(q, v, 0)$ |
| 중력 힘 $g$ 계산 | RNEA | $g = \text{RNEA}(q, 0, 0)$ |

**교차 검증 항등식**:

```math
M \cdot \dot{v} + h = \text{RNEA}(q, v, \dot{v}) \quad \forall \dot{v} \tag{5.24}
```

이를 검증하기 위해 임의의 $\dot{v}_{test}$ 를 생성하고:

```math
\epsilon = \| \text{CRBA}(q) \cdot \dot{v}_{test} + \text{RNEA}(q, v, 0) - \text{RNEA}(q, v, \dot{v}_{test}) \|_\infty \tag{5.25}
```

를 계산한다. THOR에서 $\epsilon \leq 1.14 \times 10^{-13}$ N·m (기계 정밀도).

### 5.5 부동 기저 블록의 물리적 의미

CRBA의 $M[:6,:6] = \mathcal{I}_c^{(0)}$ 는 전체 시스템을 기저에 고정된 단일 강체로 볼 때의 관성과 같다.

```math
\mathcal{I}_c^{(0)} = \sum_{i=0}^{N-1} X_{0,i}^{-T} \mathcal{I}_i X_{0,i}^{-1} \tag{5.26}
```

여기서 $X_{0,i}$ 는 기저 프레임에서 몸체 $i$ 로의 변환이다. 이는 루트에서 누적된 합성 관성이며, Newton-Euler의 전체 모멘텀 방정식을 그대로 반영한다.

---

## 6. 중심 운동량

**구현 파일**: `thor/dynamics/centroidal.py`, `thor/control/centroidal_lqr.py`

### 6.1 중심 운동량 행렬 (Centroidal Momentum Matrix, CMM)

중심 운동량(Centroidal Momentum) $h_G \in \mathbb{R}^6$ 은 전체 시스템의 CoM 프레임에서 본 각운동량과 선 운동량이다:

```math
h_G = \begin{bmatrix} k_G \\ l_G \end{bmatrix} = A_G(q) \cdot v \tag{6.1}
```

여기서:
- $k_G \in \mathbb{R}^3$: CoM에서의 각운동량 (Angular Momentum)
- $l_G \in \mathbb{R}^3$: 선 운동량 (Linear Momentum, $= M_{total} \dot{c}$)
- $A_G(q) \in \mathbb{R}^{6 \times n_{dof}}$: 중심 운동량 행렬 (CMM)

### 6.2 CMM 계산

CMM은 각 몸체의 기여를 CoM 프레임으로 변환하여 합산한다:

```math
A_G = \sum_{i=0}^{N-1} X_{G,i}^{-T} \mathcal{I}_i J_i \tag{6.2}
```

여기서 $X_{G,i}$ 는 몸체 $i$ 에서 CoM 프레임까지의 공간 변환이다.

구체적 계산: 몸체 $i$ 의 CoM 위치 $c_i$ 와 전체 CoM $c$ 의 차이 $r_i = c_i - c$ 를 이용한다:

**선 운동량 기여**:

```math
A_G[3:, :] \mathrel{+}= m_i J_{v,i} \tag{6.3a}
```

**각운동량 기여**:

```math
A_G[:3, :] \mathrel{+}= (R_i I_{cm,i} R_i^T) J_{\omega,i} + m_i [r_i]_\times J_{v,i} \tag{6.3b}
```

여기서:
- $J_{v,i} \in \mathbb{R}^{3 \times n_{dof}}$: 몸체 $i$ 자코비안의 선속도 부분 (하위 3행)
- $J_{\omega,i} \in \mathbb{R}^{3 \times n_{dof}}$: 몸체 $i$ 자코비안의 각속도 부분 (상위 3행)
- $[r_i]_\times J_{v,i}$: 위치 벡터와 선속도의 크로스곱 — 각운동량의 변환 항

```python
# thor/dynamics/centroidal.py
A_G[3:, :] += link.mass * J_v_i                        # 식 (6.3a)
A_G[:3, :] += (R_i @ link.inertia @ R_i.T) @ J_w_i    # 회전 기여
A_G[:3, :] += link.mass * skew(r_i) @ J_v_i            # 이동 기여 (6.3b)
```

### 6.3 CoM에서의 뉴턴-오일러 방정식

CoM 프레임에서의 운동 방정식:

```math
\dot{h}_G = \sum_i f_{ext,i} + \begin{bmatrix} 0 \\ M_{total} g \hat{z} \end{bmatrix} \tag{6.4}
```

성분별로:

```math
\dot{l}_G = M_{total} \ddot{c} = \sum_i f_i + M_{total} g \hat{z} \tag{6.5a}
```

```math
\dot{k}_G = \sum_i (p_i - c) \times f_i + \sum_i \tau_{ext,i} \tag{6.5b}
```

### 6.4 선형 역진자 모델 (LIPM, Linear Inverted Pendulum Model)

보행 제어의 단순화를 위해 CoM 높이를 $z_0$ 로 고정하는 3D LIPM을 사용한다.

**CoM 동역학** (수평면):

```math
\ddot{c}_x = \frac{g}{z_0}(c_x - p_{zmp,x}) \tag{6.6a}
```

```math
\ddot{c}_y = \frac{g}{z_0}(c_y - p_{zmp,y}) \tag{6.6b}
```

여기서 $p_{zmp}$ 는 영모멘트 점(Zero Moment Point, ZMP)이다.

**유도**: 발 반력이 ZMP에 집중되고 CoM 높이가 일정할 때, 각운동량 $\dot{k}_G = 0$ 조건에서 식 (6.4)를 수평면으로 투영하면 식 (6.6)을 얻는다.

LIPM의 상태 방정식 ($x$-방향, $y$-방향 동일):

```math
\underbrace{\begin{bmatrix} \dot{c} \\ \ddot{c} \end{bmatrix}}_{\dot{x}} = \underbrace{\begin{bmatrix} 0 & 1 \\ g/z_0 & 0 \end{bmatrix}}_{A} \underbrace{\begin{bmatrix} c \\ \dot{c} \end{bmatrix}}_{x} + \underbrace{\begin{bmatrix} 0 \\ 1 \end{bmatrix}}_{B} u \tag{6.7}
```

이 선형 시스템에 LQR을 적용하여 CoM 안정화 제어기를 설계한다.

```python
# thor/control/centroidal_lqr.py
omega2 = GRAVITY / z0            # g/z_0
A = np.array([[0.0, 1.0], [omega2, 0.0]])  # 식 (6.7)의 A
B = np.array([[0.0], [1.0]])               # 식 (6.7)의 B
P = solve_continuous_are(A, B, Q, R)       # ARE 풀이
K = np.linalg.solve(R, B.T @ P)           # LQR 이득 K
```

---

## 7. 접촉 역학

**구현 파일**: `thor/dynamics/contact.py`, `thor/optimization/lcp_solver.py`, `thor/dynamics/contact_implicit.py`

### 7.1 스프링-댐퍼 접촉 모델 (Kelvin-Voigt Model)

부드러운(Compliant) 접촉 모델을 사용하여 불연속성을 피한다.

#### 7.1.1 법선 힘 (Normal Force)

접촉점 침투 깊이 $\phi$ (양수 = 지면 위, 음수 = 침투):

```math
f_n = k_n \max(0, -\phi) + d_n \max(0, -\dot{\phi}) \tag{7.1}
```

여기서:
- $k_n = 3000 \;\text{[N/m]}$: 법선 강성 (Normal Stiffness)
- $d_n = 300 \;\text{[N·s/m]}$: 법선 감쇠 (Normal Damping)
- $\dot{\phi}$: 접촉점의 법선 방향 속도

**파라미터 정당성**:
- THOR 질량 $M = 65$ kg, 중력 $g = 9.81$ m/s²
- 정적 하중: $F_g = 65 \times 9.81 = 637.65$ N
- 8개 접촉점 (발 각 4개) 분산 시, 점당 $\approx 80$ N
- 정적 침투 깊이: $\phi_{static} = F_g / (8 k_n) = 637.65 / 24000 \approx 0.027$ m (2.7 cm, 허용 가능)
- 임계 감쇠: $d_{crit} = 2\sqrt{m_{foot} k_n} = 2\sqrt{3 \times 3000} \approx 190$ N·s/m → $d_n = 300$ 은 과감쇠(Overdamped) 영역

#### 7.1.2 접선 힘 — 연속 쿨롱 마찰 (Continuous Coulomb Friction)

표준 쿨롱 마찰의 불연속성을 $\tanh$ 로 평활화:

```math
\mathbf{f}_t = -\mu f_n \tanh\left(\frac{v_t}{v_s}\right) \frac{\mathbf{v}_t}{|\mathbf{v}_t|} \tag{7.2}
```

여기서:
- $\mu = 0.7$: 마찰 계수
- $v_t = |\mathbf{v}_t|$: 접선 속도 크기
- $v_s = 0.01$ m/s: 정지 속도 임계값(Stiction Velocity Threshold)

**물리적 의미**: $v_t \ll v_s$ 이면 $\tanh(v_t/v_s) \approx v_t/v_s$ → 선형 감쇠 (정지 근방). $v_t \gg v_s$ 이면 $\tanh(v_t/v_s) \approx 1$ → 순수 쿨롱 마찰.

```python
# thor/dynamics/contact.py
f_n = k_n * (-phi) + d_n * max(0.0, -dphi)          # 식 (7.1)
friction_mag = mu * f_n * math.tanh(v_t / v_s)
f[0] = -friction_mag * vel[0] / v_t                  # 식 (7.2) x성분
f[1] = -friction_mag * vel[1] / v_t                  # 식 (7.2) y성분
```

#### 7.1.3 접촉점 속도 계산

발 모서리의 속도는 강체 운동 공식으로 계산한다:

```math
v_{corner} = v_{foot} + \omega_{foot} \times r \tag{7.3}
```

여기서 $r$ 은 발 중심에서 모서리까지의 벡터다.

### 7.2 LCP 정식화 (Linear Complementarity Problem Formulation)

강체(Rigid Body) 접촉을 수학적으로 표현하기 위한 LCP 정식화를 유도한다.

#### 7.2.1 Signorini 상보성 조건

비관통(Non-penetration) 조건과 단방향 힘(Unilateral Force) 조건:

```math
\phi_n \geq 0, \quad \lambda_n \geq 0, \quad \phi_n \lambda_n = 0 \tag{7.4}
```

이를 간결하게: $0 \leq \lambda_n \perp \phi_n \geq 0$

의미:
- $\phi_n > 0$ (비접촉): $\lambda_n = 0$ (힘 없음)
- $\phi_n = 0$ (접촉 중): $\lambda_n \geq 0$ (압축력만 가능)

#### 7.2.2 LCP 표준 형태

접촉 법선 방향으로 운동 방정식과 상보성 조건을 결합하면:

```math
w = M_{LCP} \lambda + q_{LCP}, \quad 0 \leq \lambda \perp w \geq 0 \tag{7.5}
```

여기서 **Delassus 행렬 (Delassus Matrix)**:

```math
M_{LCP} = J_n M^{-1} J_n^T \tag{7.6}
```

$J_n \in \mathbb{R}^{n_c \times n_{dof}}$ 는 접촉 법선 자코비안이다.

```math
q_{LCP} = J_n M^{-1} (h - S^T \tau) \cdot h_{step} + \frac{\phi_n}{h_{step}} \tag{7.7}
```

Delassus 행렬의 성질: $M \succ 0$ 이고 $J_n$ 이 만약 행 독립이면 $M_{LCP} \succ 0$ (P-행렬). LCP는 고유 해를 가진다.

#### 7.2.3 Delassus 행렬의 유도 (완전)

이산화된 운동 방정식 (7.14):

```math
M v_{k+1} = M v_k + h_{step}(S^T \tau - h) + J_n^T \lambda_n \tag{7.7a}
```

양변에 $M^{-1}$ 를 곱하면:

```math
v_{k+1} = v_k + M^{-1}(h_{step}(S^T \tau - h) + J_n^T \lambda_n) \tag{7.7b}
```

법선 방향 속도 제약 ($J_n v_{k+1} \geq -\phi_n/h_{step}$):

```math
J_n v_{k+1} = J_n v_k + J_n M^{-1}(h_{step}(S^T\tau - h)) + J_n M^{-1} J_n^T \lambda_n \tag{7.7c}
```

이를 LCP 표준 형태로 재정리하면:

```math
\underbrace{J_n M^{-1} J_n^T}_{M_{LCP}} \lambda_n + \underbrace{J_n v_k + J_n M^{-1} h_{step}(S^T\tau - h) + \phi_n/h_{step}}_{q_{LCP}} = w \tag{7.7d}
```

상보성 조건: $0 \leq \lambda_n \perp w \geq 0$ $\quad \blacksquare$

### 7.3 Fischer-Burmeister NCP 함수

상보성 조건 (7.4)를 매끄러운 방정식으로 변환한다.

**Fischer-Burmeister 함수**:

```math
\phi_{FB}(a, b; \varepsilon) = a + b - \sqrt{a^2 + b^2 + 2\varepsilon^2} \tag{7.8}
```

**성질**:
1. $\phi_{FB}(a, b; 0) = 0 \iff a \geq 0, b \geq 0, ab = 0$ (상보성 조건과 등가)
2. $\varepsilon > 0$ 이면 $\phi_{FB}$ 는 도처에서 미분 가능 (Differentiable everywhere)
3. $\varepsilon \to 0$ 으로 수렴 시 원래 상보성 조건에 근접

**야코비안 유도**: 식 (7.8)을 편미분하면:

```math
\frac{\partial \phi_{FB}}{\partial a} = 1 - \frac{a}{\sqrt{a^2 + b^2 + 2\varepsilon^2}} \tag{7.9a}
```

```math
\frac{\partial \phi_{FB}}{\partial b} = 1 - \frac{b}{\sqrt{a^2 + b^2 + 2\varepsilon^2}} \tag{7.9b}
```

LCP $w = M\lambda + q$ 에 식 (7.8)를 적용하면, $a_i = \lambda_i$, $b_i = w_i = (M\lambda + q)_i$ 이고 전체 잔차 벡터 $F(\lambda)$ 의 야코비안:

```math
J_{ij} = \frac{\partial \phi_{FB,i}}{\partial \lambda_j} = \frac{\partial \phi_{FB}}{\partial b_i} M_{ij} + \delta_{ij} \frac{\partial \phi_{FB}}{\partial a_i} \tag{7.10}
```

```python
# thor/optimization/lcp_solver.py
def fischer_burmeister(a, b, eps=1e-10):
    return a + b - math.sqrt(a*a + b*b + 2.0*eps*eps)    # 식 (7.8)

# 야코비안 (식 7.10)
denom = math.sqrt(z[i]**2 + w[i]**2 + 2.0*eps**2)
da = 1.0 - z[i] / denom   # ∂φ/∂a
db = 1.0 - w[i] / denom   # ∂φ/∂b
for j in range(n):
    J[i,j] = db * M[i,j]
J[i,i] += da
```

#### 7.3.1 감쇠 뉴턴 방법 (Damped Newton Method)

FB 방정식 $F(\lambda) = 0$ 을 반복적으로 풀기 위해 역추적 직선 탐색(Backtracking Line Search)을 포함한 Newton 방법을 사용한다:

1. 잔차 $F(\lambda^k)$ 와 야코비안 $J^k$ 계산
2. Newton 방향: $\Delta\lambda = -(J^k)^{-1} F(\lambda^k)$
3. 역추적 선 탐색: $\alpha$ 를 $\|F(\lambda + \alpha \Delta\lambda)\| < \|F(\lambda)\|$ 가 될 때까지 반분
4. 업데이트: $\lambda^{k+1} = \lambda^k + \alpha \Delta\lambda$

### 7.4 내부점법 (Interior-Point Method)

FB-Newton이 수렴하지 않는 ill-conditioned 문제를 위해 내부점법을 제공한다.

**완화된 상보성 (Relaxed Complementarity)**:

```math
\lambda_i w_i = \kappa \quad (\kappa > 0) \tag{7.11}
```

$\kappa$ 를 **쌍대성 측도 (Duality Measure)**:

```math
\mu = \frac{\lambda^T w}{n} \tag{7.12}
```

**중심화 파라미터**: $\kappa = \sigma \mu$, $\sigma \in (0, 1]$.

Newton 스텝 방정식 (Schur complement 형태):

```math
(W + Z M) \Delta\lambda = -r_c + Z r_w \tag{7.13}
```

여기서 $W = \text{diag}(w)$, $Z = \text{diag}(\lambda)$, $r_w = w - M\lambda - q$, $r_c = \lambda \odot w - \kappa$.

```python
# thor/optimization/lcp_solver.py (solve_lcp_interior_point)
lhs = W + Z @ M + 1e-10 * np.eye(n)   # 식 (7.13) 좌변
rhs = -r_c + Z @ r_w                   # 식 (7.13) 우변
dz = np.linalg.solve(lhs, rhs)
```

### 7.5 접촉-내재적 시간 스테핑 (Contact-Implicit Time-Stepping)

Stewart & Trinkle (1996)의 접촉-내재적 시간 스테핑 방법을 구현한다.

#### 7.5.1 전체 시스템 방정식

이산화된 운동 방정식:

```math
M(v_{k+1} - v_k) = h_{step}(S^T \tau - h) + J_c^T \lambda \tag{7.14}
```

#### 7.5.2 Schur Complement를 이용한 기저 소거 (완전 유도)

이중 지지(Double Support) 상태에서 기저가 지면에 고정될 때, $\dot{v}_{base} = 0$ 제약을 부과한다.

전체 시스템을 기저(b)와 관절(j)로 분할:

```math
\begin{bmatrix} M_{bb} & M_{bj} \\ M_{jb} & M_{jj} \end{bmatrix} \begin{bmatrix} \dot{v}_b \\ \dot{v}_j \end{bmatrix} + \begin{bmatrix} h_b \\ h_j \end{bmatrix} = \begin{bmatrix} f_b \\ \tau_j \end{bmatrix} \tag{7.15}
```

여기서 $f_b$ 는 기저 반력이다.

**제약**: 이중 지지에서 $\dot{v}_b = 0$ 으로 설정한다. 두 번째 행 방정식:

```math
M_{jb} \cdot 0 + M_{jj} \dot{v}_j + h_j = \tau_j \tag{7.16a}
```

```math
M_{jj} \dot{v}_j = \tau_j - h_j \tag{7.16b}
```

이것이 **축소 34×34 시스템**이다. $M_{bb}$ 에 관한 항들이 $\dot{v}_b = 0$ 으로 소거된다.

**기저 반력 복원**: 첫 번째 행 방정식에서:

```math
M_{bb} \cdot 0 + M_{bj} \dot{v}_j + h_b = f_b \tag{7.17}
```

```math
f_b = M_{bj} \dot{v}_j + h_b \tag{7.18}
```

이를 통해 지면 반력(Ground Reaction Force)을 계산한다.

```python
# thor/dynamics/contact_implicit.py (Double Support 케이스)
M_jj = M[6:, 6:]
h_j  = bias[6:]
tau_j = tau[6:]
rhs_j = tau_j - h_j

# 축소 시스템 풀기 (식 7.16b)
cho_jj = cho_factor(M_jj + reg)
ddq_j  = cho_solve(cho_jj, rhs_j)

# 속도 및 설정 업데이트 (반암시적 오일러)
v_new[6:] += h * ddq_j
v_new[:6] = 0.0   # 기저 속도 = 0 (이중 지지 제약)

# 기저 반력 (식 7.18)
f_contact_base = M[:6, 6:] @ ddq_j + bias[:6]
```

**Cholesky 분해의 사용**: $M_{jj}$ 는 대칭 양정치이므로 Cholesky 분해를 사용하면 LU 분해 대비 절반의 연산으로 선형 시스템을 풀 수 있다.

```math
M_{jj} = L L^T, \quad L \Delta v_j = \tau_j - h_j \tag{7.19}
```

#### 7.5.3 단일 지지 (Single Support)와 비행 구간 (Flight)

- **단일 지지**: 기저의 일부 DOF(5개)만 고정하는 부분 제약 적용
- **비행 구간**: 전체 40-DOF 자유 동역학, 기저 속도는 0으로 초기화

---

## 8. 제어 이론

**구현 파일**: `thor/control/walking_controller.py`, `thor/control/centroidal_lqr.py`, `thor/control/contact_implicit_mpc.py`, `thor/control/whole_body_qp.py`

### 8.1 CI-MPC (Contact-Implicit Model Predictive Control)

Le Cleac'h et al. (2024)의 프레임워크 기반.

**MPC 비용 함수**:

```math
J = \sum_{k=0}^{N-1} \left( \|q_k - q_{ref,k}\|_{Q_q}^2 + \|v_k - v_{ref,k}\|_{Q_v}^2 + \|\tau_k\|_R^2 \right) \tag{8.1}
```

여기서:
- $Q_q = \text{diag}(q_{pos} \cdot \mathbf{1})$: 설정 추종 가중치
- $Q_v = \text{diag}(q_{vel} \cdot \mathbf{1})$: 속도 추종 가중치
- $R = \text{diag}(r \cdot \mathbf{1})$: 제어 노력 가중치

**제약**: 각 예측 스텝에서 LCP 접촉 조건 (식 7.5)이 만족되어야 한다.

**실시간 근사 구현**: 전체 수평 최적화는 40-DOF에서 실시간 불가능하므로, 중력 보상 피드포워드 + PD 피드백으로 근사한다:

```math
\tau = g(q) + K_p (q_{ref} - q) + K_v (v_{ref} - v) \tag{8.2}
```

```python
# thor/control/contact_implicit_mpc.py
g = gravity_forces(self._model, q)
tau[6:] = g[6:]                     # 중력 보상
tau[6:] -= Kp * q_err + Kd * dq_err  # PD 피드백
```

### 8.2 계산 토크 제어 (Computed Torque Control)

이론적으로 완전한 추적(Perfect Tracking)을 보장하는 비선형 피드포워드 제어.

**제어 법칙**:

```math
\tau = M_{jj} \ddot{q}_{des} + h_j \tag{8.3}
```

**완전 추적 증명**:

실제 관절 가속도는 운동 방정식 (7.16b)에서:

```math
M_{jj} \ddot{q} = \tau - h_j = M_{jj} \ddot{q}_{des} + h_j - h_j = M_{jj} \ddot{q}_{des} \tag{8.4}
```

$M_{jj} \succ 0$ 이므로:

```math
\ddot{q} = \ddot{q}_{des} \tag{8.5}
```

따라서 원하는 가속도가 정확히 달성된다. $\quad \blacksquare$

**오차 동역학**: 원하는 가속도를 PD 제어로 설정한다:

```math
\ddot{q}_{des} = \ddot{q}_{ref} - K_p (q - q_{ref}) - K_d (\dot{q} - \dot{q}_{ref}) \tag{8.6}
```

추적 오차 $e = q - q_{ref}$ 의 동역학:

```math
\ddot{e} + K_d \dot{e} + K_p e = 0 \tag{8.7}
```

이는 2차 선형 미분 방정식으로, 특성 다항식:

```math
s^2 + K_d s + K_p = 0 \tag{8.8}
```

의 근이 모두 좌반평면에 있으면 오차가 지수적으로 수렴한다.

**감쇠비와 고유진동수 선택**:

원하는 자연진동수 $\omega_n$, 감쇠비 $\zeta$ 에 대해:

```math
K_p = \omega_n^2, \quad K_d = 2\zeta\omega_n \tag{8.9}
```

THOR 구현에서 다리 관절: $K_p = 600$, $K_d = 60$:

```math
\omega_n = \sqrt{600} \approx 24.5 \;\text{rad/s}, \quad \zeta = \frac{60}{2 \times 24.5} \approx 1.22 \;(\text{과감쇠}) \tag{8.10}
```

```python
# thor/control/walking_controller.py
ddq_des[i] = -kp * q_err[i] - kd * dq_err[i]   # 식 (8.6)
tau[6:] = M_jj @ ddq_des + h_j                  # 식 (8.3)
```

### 8.3 보행 궤적 생성 (Winter 1991 Joint Profiles)

Winter (1991)의 생체역학 데이터를 기반으로 연속 관절 궤적을 생성한다.

**보행 위상 파라미터**:
- 이중 지지(Double Support) 지속 시간: $T_{DS} = 0.25$ s
- 스윙(Swing) 지속 시간: $T_{swing} = 0.55$ s
- 보행 사이클: $T_{cycle} = T_{DS} + T_{swing} = 0.80$ s

**위상 변수**: $s \in [0, 1]$ ($s=0$: 발 떼기, $s=0.5$: 스윙 중간, $s=1$: 발 착지)

#### 8.3.1 스윙 다리 궤적 (Swing Leg Trajectory)

**엉덩이 피치 (Hip Pitch)**: 코사인 보간 (신전 → 굴곡):

```math
\theta_{hip}(s) = \theta_{ext} + (\theta_{flex} - \theta_{ext}) \cdot \frac{1 - \cos(\pi s)}{2} \tag{8.11}
```

여기서 $\theta_{ext} = -5°$ (신전), $\theta_{flex} = 20°$ (굴곡).

물리적 의미: $s=0$ 에서 $\theta_{ext}$, $s=1$ 에서 $\theta_{flex}$ 로 부드럽게 전환. 코사인 보간은 시작/끝에서 속도가 0이 되어 충격을 최소화한다.

**무릎 피치 (Knee Pitch)**: 비대칭 벨 형태 ($\sin^{0.8}$):

```math
\theta_{knee}(s) = \theta_{stance} + (\theta_{swing} - \theta_{stance}) \cdot \sin(\pi s)^{0.8} \tag{8.12}
```

여기서 $\theta_{stance} = 5°$, $\theta_{swing} = 45°$.

$\sin^{0.8}$ (지수 0.8)의 의미: 표준 $\sin$ ($p=1$) 보다 피크가 더 날카롭고 초기 굴곡이 빠르다. 이는 발목 추진(Push-off) 직후 빠른 무릎 굴곡을 모델링한다.

**발목 피치 (Ankle Pitch)**: 정현파 배측굴곡:

```math
\theta_{ankle}(s) = \theta_{swing} \cdot \sin(\pi s) \tag{8.13}
```

여기서 $\theta_{swing} = 5°$ (배측굴곡, Dorsiflexion).

```python
# thor/control/gait/swing_trajectory.py
def swing_leg_angles(s):
    hip_p = HIP_STANCE_EXT + (HIP_SWING_FLEX - HIP_STANCE_EXT) * (
        0.5 - 0.5 * math.cos(math.pi * s))      # 식 (8.11)
    kn_p  = KNEE_STANCE + (KNEE_SWING_FLEX - KNEE_STANCE) * (
        math.sin(math.pi * s) ** 0.8)            # 식 (8.12)
    an_p  = ANKLE_SWING * math.sin(math.pi * s) # 식 (8.13)
    return hip_p, kn_p, an_p
```

#### 8.3.2 지지 다리 궤적 (Stance Leg Trajectory)

$s \in [0, 1]$ ($s=0$: 발 착지, $s=1$: 발 떼기)

**엉덩이 피치**: 선형 감소 (굴곡 → 신전):

```math
\theta_{hip}^{stance}(s) = \theta_{flex}(1-s) + \theta_{ext} s \tag{8.14}
```

**무릎 피치**: 지지 중 작은 굴곡:

```math
\theta_{knee}^{stance}(s) = \theta_{stance} + 10° \cdot \sin\left(\frac{\pi s}{2}\right) \tag{8.15}
```

**발목 피치**: 배측굴곡 + 추진:

```math
\theta_{ankle}^{stance}(s) = 5° \cdot \sin(\pi s) + \theta_{push} \cdot s^2 \tag{8.16}
```

여기서 $\theta_{push} = -10°$ (추진, Push-off).

#### 8.3.3 이중 지지 코사인 블렌딩 (Double Support Cosine Blending)

스윙→지지 전환 시 각도의 불연속을 방지하기 위해 코사인 블렌딩을 사용한다:

```math
\theta = (1 - \beta) \theta_{end} + \beta \theta_{start}' \tag{8.17}
```

```math
\beta(s_{DS}) = \frac{1 - \cos(\pi s_{DS})}{2}, \quad s_{DS} \in [0, 1] \tag{8.18}
```

여기서 $\theta_{end}$ 는 이전 위상 끝, $\theta_{start}'$ 는 다음 위상 시작 각도다.

```python
# thor/control/walking_controller.py
blend = 0.5 - 0.5 * math.cos(math.pi * s_ds)   # 식 (8.18)
l_hip = swing_hip_end * (1 - blend) + stance_hip_start * blend  # 식 (8.17)
```

### 8.4 중심 LQR과 전신 QP (Centroidal LQR + Whole-Body QP)

계층적 제어 구조의 요약:

**계층 1 (중심 LQR)**: LIPM 동역학 (식 6.7)에서 Riccati 방정식:

```math
A^T P + P A - P B R^{-1} B^T P + Q = 0 \tag{8.19}
```

이득: $K = R^{-1} B^T P$

제어: $u = -K \begin{bmatrix} c - c_{des} \\ \dot{c} - \dot{c}_{des} \end{bmatrix}$

**계층 2 (전신 QP)**: CMM 기반 운동량 추적:

```math
\min_{\tau, f} \|A_G(q) v - h_{G,ref}\|^2 + \lambda \|\tau\|^2 \tag{8.20}
```

```math
\text{s.t.} \quad M \dot{v} + h = S^T \tau + J_c^T f_c, \quad \text{마찰 원뿔 제약}
```

### 8.5 마찰 원뿔 제약 (Friction Cone Constraint)

접촉력 $f_c = [f_t^T, f_n]^T$ 는 마찰 원뿔(Friction Cone) 내에 있어야 한다:

```math
\|f_t\| \leq \mu f_n, \quad f_n \geq 0 \tag{8.21}
```

이를 선형화(Linearized Friction Cone)하면 다각형으로 근사한다 (`FRICTION_CONE_FACES = 8` 면):

각도 $\theta_k = 2\pi k / 8$, $k = 0, \ldots, 7$ 에 대해:

```math
\cos\theta_k f_x + \sin\theta_k f_y \leq \mu f_n \tag{8.22}
```

8개의 선형 부등식으로 원뿔을 내접 다각형으로 근사한다.

### 8.6 중력 보상의 중요성

중력 보상 없이 PD 제어만 사용하면:

```math
\tau_{PD} = K_p(q_{ref} - q) + K_d(\dot{q}_{ref} - \dot{q}) \tag{8.23}
```

정적 오차: $\Delta q_{static} = K_p^{-1} g_j(q)$. THOR 다리에서 $\|g_j\| \approx 200$ N·m, $K_p = 600$ N·m/rad 이면 $\Delta q_{static} \approx 0.33$ rad $\approx 19°$. 이는 허용 불가능한 오차다.

계산 토크 제어에서는 $h_j$ 에 중력이 포함되어 자동으로 보상된다.

---

## 9. 수치 적분

**구현 파일**: `thor/dynamics/integrators.py`, `thor/dynamics/contact_implicit.py`

### 9.1 반암시적 오일러 (Semi-Implicit / Symplectic Euler)

**속도 업데이트** (전진 오일러):

```math
v_{k+1} = v_k + h \cdot \dot{v}_k \tag{9.1}
```

**설정 업데이트** (후진 오일러 — 새 속도 사용):

```math
q_{k+1} = q_k + h \cdot v_{k+1} \tag{9.2}
```

표준 전진 오일러($q_{k+1} = q_k + h v_k$)와 달리, 반암시적 오일러는 설정 업데이트에 **새 속도** $v_{k+1}$ 을 사용한다.

**심플렉틱 구조(Symplectic Structure)**: 위상 공간 부피가 보존된다. 즉, 에너지는 발산하지 않고 유계 오차 내에서 진동한다. 반면 전진 오일러는 에너지가 단조 증가하여 발산할 수 있다.

```python
# thor/dynamics/integrators.py (SemiImplicitEuler)
def integrate_velocity(self, v, ddq, dt):
    return v + dt * ddq           # 식 (9.1)

def integrate_config(self, q, v, dt):   # v = v_{k+1}
    q_new[:3] += dt * v[3:6]     # 위치 (식 9.2)
    q_new[3:7] = quat_integrate(q[3:7], v[0:3], dt)  # 쿼터니언 (식 9.4)
    q_new[7:]  += dt * v[6:]     # 관절 (식 9.2)
```

### 9.2 쿼터니언 적분 (Quaternion Integration)

쿼터니언의 시간 미분:

```math
\frac{d\mathbf{q}}{dt} = \frac{1}{2} \begin{bmatrix} 0 \\ \omega \end{bmatrix} \otimes \mathbf{q} \tag{9.3}
```

여기서 $\otimes$ 는 쿼터니언 곱이다.

**유도**: 회전 행렬의 시간 미분 $\dot{R} = [\omega]_\times R$ 에서 출발하여 쿼터니언 표현으로 변환하면 식 (9.3)을 얻는다.

식 (9.3)을 전개하면 ($\mathbf{q} = [w, x, y, z]^T$):

```math
\frac{d}{dt}\begin{bmatrix} w \\ x \\ y \\ z \end{bmatrix} = \frac{1}{2} \begin{bmatrix} -\omega_x x - \omega_y y - \omega_z z \\ \omega_x w + \omega_z y - \omega_y z \\ \omega_y w - \omega_z x + \omega_x z \\ \omega_z w + \omega_x y - \omega_y x \end{bmatrix} \tag{9.3'}
```

**1차 근사 + 정규화**:

```math
\mathbf{q}_{k+1} = \mathbf{q}_k + h \cdot \frac{d\mathbf{q}_k}{dt} \tag{9.4a}
```

```math
\mathbf{q}_{k+1} \leftarrow \frac{\mathbf{q}_{k+1}}{\|\mathbf{q}_{k+1}\|} \tag{9.4b}
```

정규화 (9.4b)는 적분 오차로 인한 단위 쿼터니언 제약 위반을 보정한다.

```python
# thor/model/quaternion.py
def quat_integrate(quat, omega, dt):
    w, x, y, z = quat
    dquat = 0.5 * dt * np.array([
        -omega[0]*x - omega[1]*y - omega[2]*z,  # dw/dt
         omega[0]*w + omega[2]*y - omega[1]*z,  # dx/dt
         omega[1]*w - omega[2]*x + omega[0]*z,  # dy/dt
         omega[2]*w + omega[1]*x - omega[0]*y,  # dz/dt
    ])                                            # 식 (9.3')
    q_new = quat + dquat
    norm = np.linalg.norm(q_new)
    q_new /= norm                                 # 식 (9.4b)
    return q_new
```

### 9.3 에너지 드리프트 분석 (Energy Drift Analysis)

반암시적 오일러의 수치 에너지 오차를 분석한다.

**국소 절단 오차 (Local Truncation Error)**:

속도에 대해: $v_{k+1} = v_k + h\dot{v}_k + O(h^2)$ → 1차 정확도

설정에 대해: $q_{k+1} = q_k + h v_{k+1} = q_k + h v_k + h^2 \dot{v}_k + O(h^3)$

실제 궤적과 비교: $q(t_{k+1}) = q_k + h v_k + \frac{h^2}{2}\dot{v}_k + O(h^3)$

오차: $q_{k+1} - q(t_{k+1}) = \frac{h^2}{2}\dot{v}_k + O(h^3)$ → 스텝당 $O(h^2)$ 오차

**에너지 드리프트**: 심플렉틱 적분기는 수정 해밀토니안(Modified Hamiltonian)을 보존한다. 에너지 드리프트는 $O(h)$ 크기이지만 유계(Bounded)이므로 장기 시뮬레이션에서도 안정적이다:

```math
|H(q_{k}, v_{k}) - H(q_0, v_0)| = O(h) \quad \text{(유계, 선형 증가 없음)} \tag{9.5}
```

반면 전진 오일러(Explicit Euler)는:

```math
|H(q_{k}, v_{k}) - H(q_0, v_0)| \sim O(e^{h \Lambda_{max} t}) \tag{9.6}
```

지수적으로 발산할 수 있다.

**THOR 시뮬레이션 파라미터**:
- 시뮬레이션 스텝: $h = 0.001$ s (1 kHz)
- MPC 스텝: $h_{MPC} = 0.02$ s (50 Hz)

```python
# thor/core/constants.py
DEFAULT_DT: float = 0.001   # 1 kHz dynamics
MPC_DT: float = 0.02        # 50 Hz MPC
```

### 9.4 접촉 시뮬레이션의 설정 적분

`contact_implicit_step` 함수 내 설정 적분:

```python
# thor/dynamics/contact_implicit.py (_integrate_config 함수)
q_new[:3] += h * v[3:6]         # 위치 += h * 선속도 (v[3:6])
q_new[3:7] = quat_integrate(q[3:7], v[0:3], h)  # 쿼터니언 (v[0:3] = ω)
q_new[7:]  += h * v[6:]         # 관절 += h * 관절속도
```

공간 속도 규약 확인: `v[0:3]` = 각속도 $\omega$, `v[3:6]` = 선속도 $v_{lin}$ (omega-first, 식 2.1)

---

## 10. 참고 문헌

1. **Featherstone, R. (2008)**. *Rigid Body Dynamics Algorithms*. Springer.
   - Ch. 2: 공간 벡터 대수 (Section 2 전체)
   - Ch. 4: 순기구학 (Section 3)
   - Ch. 5: RNEA (Section 5.1)
   - Ch. 6: CRBA (Section 5.2)
   - Ch. 7: ABA (Section 5.3)
   - Ch. 9: 접촉 역학 (Section 7)

2. **Winter, D.A. (1991)**. *Biomechanics and Motor Control of Human Movement* (2nd ed.). Wiley.
   - 보행 관절 각도 정규 데이터 (Section 8.3)

3. **Stewart, D.E. & Trinkle, J.C. (1996)**. "An Implicit Time-Stepping Scheme for Rigid Body Dynamics with Inelastic Collisions and Coulomb Friction." *International Journal for Numerical Methods in Engineering*, 39(15), 2673–2691.
   - LCP 접촉-내재적 시간 스테핑 (Section 7.5)

4. **Le Cleac'h, S., Howell, T., Schwager, M. & Manchester, Z. (2024)**. "Fast Contact-Implicit Model Predictive Control." *IEEE Transactions on Robotics*, 40, 1617–1634.
   - CI-MPC 프레임워크 (Section 8.1)

5. **Hopkins, M.A. & Leonessa, A. (2015)**. "Optimization-Based Whole-Body Control of a Series Elastic Humanoid Robot." *International Journal of Humanoid Robotics*, 12(3).
   - THOR 로봇 물리 파라미터 및 제어 구조

6. **Orin, D.E., Goswami, A. & Lee, S.-H. (2013)**. "Centroidal Dynamics of a Humanoid Robot." *Autonomous Robots*, 35(2-3), 161–176.
   - 중심 운동량 행렬 이론 (Section 6)

7. **Kajita, S., Kanehiro, F., Kaneko, K., Fujiwara, K., Harada, K., Yokoi, K. & Hirukawa, H. (2003)**. "Biped Walking Pattern Generation by using Preview Control of Zero-Moment Point." *IEEE ICRA*.
   - LIPM과 ZMP 개념 (Section 6.4)

8. **Fischer, A. (1992)**. "A Special Newton-Type Optimization Method." *Optimization*, 24(3-4), 269–284.
   - Fischer-Burmeister NCP 함수 (Section 7.3)

9. **Marhefka, D.W. & Orin, D.E. (1999)**. "A Compliant Contact Model with Nonlinear Damping for Simulation of Robotic Systems." *IEEE Transactions on Systems, Man, and Cybernetics*, 29(6), 566–572.
   - 켈빈-포이그트 접촉 모델 (Section 7.1)

10. **Diebel, J. (2006)**. "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors." *Stanford Technical Report*.
    - 쿼터니언 규약 및 미분 방정식 (Section 1.3, 9.2)

11. **Spong, M.W., Hutchinson, S. & Vidyasagar, M. (2005)**. *Robot Modeling and Control*. Wiley.
    - 계산 토크 제어 이론 (Section 8.2)

12. **Wieber, P.-B. (2006)**. "Trajectory Free Linear Model Predictive Control for Stable Walking in the Presence of Strong Perturbations." *IEEE-RAS International Conference on Humanoid Robots*.
    - CoM 궤적 MPC (Section 8.1)

---

## 부록 A. 기호 일람 (Notation Summary)

| 기호 | 차원 | 의미 |
|:---:|:---:|:---|
| $q$ | $\mathbb{R}^{41}$ | 일반화 좌표 $[p_{base}(3),\, \mathbf{q}_{quat}(4),\, q_{joints}(34)]$ |
| $v$ | $\mathbb{R}^{40}$ | 일반화 속도 $[\omega_{base}(3),\, v_{lin,base}(3),\, \dot{q}_{joints}(34)]$ |
| $\dot{v}$ | $\mathbb{R}^{40}$ | 일반화 가속도 |
| $M(q)$ | $\mathbb{R}^{40 \times 40}$ | 질량 행렬 (대칭 양정치) |
| $h(q,v)$ | $\mathbb{R}^{40}$ | 바이어스 힘 (코리올리+원심+중력) |
| $\tau$ | $\mathbb{R}^{34}$ | 관절 토크 벡터 |
| $S^T$ | $\mathbb{R}^{40 \times 34}$ | 선택 행렬 |
| $J_c$ | $\mathbb{R}^{3n_c \times 40}$ | 접촉 자코비안 |
| $f_c$ | $\mathbb{R}^{3n_c}$ | 접촉력 벡터 |
| $\mathbf{v}$ | $\mathbb{R}^6$ | 공간 속도 (Twist) $[\omega;\, v_{lin}]$ |
| $\mathbf{f}$ | $\mathbb{R}^6$ | 공간 힘 (Wrench) $[\tau;\, f]$ |
| $X$ | $\mathbb{R}^{6 \times 6}$ | 공간 변환 행렬 (Plücker) |
| $\mathcal{I}$ | $\mathbb{R}^{6 \times 6}$ | 공간 관성 행렬 |
| $S_i$ | $\mathbb{R}^6$ | 관절 $i$ 의 운동 부분공간 |
| $X_{up,i}$ | $\mathbb{R}^{6 \times 6}$ | 몸체 $i$ 에서 부모로의 공간 변환 |
| $\mathcal{I}_{c}^{(i)}$ | $\mathbb{R}^{6 \times 6}$ | 몸체 $i$ 의 합성 관성 (CRBA) |
| $IA_i$ | $\mathbb{R}^{6 \times 6}$ | 몸체 $i$ 의 관절체 관성 (ABA) |
| $pA_i$ | $\mathbb{R}^6$ | 몸체 $i$ 의 관절체 바이어스 힘 (ABA) |
| $A_G$ | $\mathbb{R}^{6 \times 40}$ | 중심 운동량 행렬 (CMM) |
| $h_G$ | $\mathbb{R}^6$ | 중심 운동량 $[k_G;\, l_G]$ |
| $c$ | $\mathbb{R}^3$ | 전체 시스템 질량 중심 위치 |
| $R$ | $\mathbb{R}^{3 \times 3}$ | 회전 행렬 ($\in SO(3)$) |
| $[p]_\times$ | $\mathbb{R}^{3 \times 3}$ | 벡터 $p$ 의 반대칭 행렬 (Skew-Symmetric) |
| $\phi_{FB}$ | $\mathbb{R}$ | Fischer-Burmeister NCP 함수 |
| $\lambda$ | $\mathbb{R}^{n_c}$ | LCP 해 (접촉 법선력) |
| $h_{step}$ | $\mathbb{R}$ | 시간 스텝 크기 |
| $N$ | 정수 | 몸체 수 ($=35$, THOR) |
| $n_{dof}$ | 정수 | DOF 수 ($=40$, THOR) |
| $\mathbf{q}$ | $\mathbb{R}^4$ | 쿼터니언 $[w, x, y, z]^T$ |
| $\omega$ | $\mathbb{R}^3$ | 각속도 벡터 |
| $g$ | $\mathbb{R}$ | 중력 가속도 크기 ($9.81$ m/s²) |

---

## 부록 B. 스큐 대칭 행렬 (Skew-Symmetric Matrix) 성질

### B.1 정의와 기본 성질

벡터 $p = [p_x, p_y, p_z]^T$ 에 대한 스큐 대칭 행렬:

```math
[p]_\times = \begin{bmatrix} 0 & -p_z & p_y \\ p_z & 0 & -p_x \\ -p_y & p_x & 0 \end{bmatrix} \tag{B.1}
```

**성질 1** (반대칭): $[p]_\times^T = -[p]_\times$

**증명**: $(i,j)$ 원소와 $(j,i)$ 원소의 부호가 반대이므로 직접 확인 가능. $\quad \checkmark$

**성질 2** (크로스곱 표현): $p \times q = [p]_\times q$

**증명**: 

```math
[p]_\times q = \begin{bmatrix} -p_z q_y + p_y q_z \\ p_z q_x - p_x q_z \\ -p_y q_x + p_x q_y \end{bmatrix} = p \times q \quad \checkmark \tag{B.2}
```

**성질 3**: $[p]_\times^2 = p p^T - \|p\|^2 I_3$

**증명**: 

```math
(p \times q) \times r = (q r^T - r q^T)p
```

에서 행렬 형태로 변환하면 확인 가능. $\quad \checkmark$

**성질 4**: $[p]_\times [p]_\times^T = \|p\|^2 I_3 - p p^T$ (음의 준정치)

이로부터 공간 관성 식 (2.14)에서 $I_{cm} + m[c]_\times [c]_\times^T$ 가 양정치임을 알 수 있다.

### B.2 회전 변환 하에서의 스큐 행렬

```math
[Rp]_\times = R[p]_\times R^T \tag{B.3}
```

**증명**: 임의의 $q$ 에 대해 $(Rp) \times q = R(p \times R^T q)$ 에서:

```math
[Rp]_\times q = R[p]_\times R^T q \quad \forall q \implies [Rp]_\times = R[p]_\times R^T \quad \checkmark
```

이 성질은 공간 관성의 좌표 변환 (식 2.15)에서 핵심적으로 사용된다.

---

## 부록 C. 회전 행렬의 미분 (Rotation Matrix Derivatives)

### C.1 SO(3) 위의 미분

$R(t) \in SO(3)$ 의 시간 미분:

```math
\dot{R} = [\omega]_\times R \tag{C.1}
```

**유도**: $R R^T = I$ 를 시간 미분하면:

```math
\dot{R} R^T + R \dot{R}^T = 0
```

```math
\dot{R} R^T = -(\dot{R} R^T)^T
```

따라서 $\dot{R} R^T$ 는 반대칭이므로 어떤 벡터 $\omega$ 에 대해 $\dot{R} R^T = [\omega]_\times$ 가 성립한다. 이로부터:

```math
\dot{R} = [\omega]_\times R \quad \checkmark \tag{C.2}
```

### C.2 쿼터니언 미분 유도 (상세)

식 (9.3)의 완전 유도. $R(\mathbf{q}(t))$ 를 시간 미분하면 식 (C.1)이 성립해야 하므로:

```math
\frac{dR}{dt} = \frac{\partial R}{\partial \mathbf{q}} \dot{\mathbf{q}} = [\omega]_\times R \tag{C.3}
```

식 (1.7)의 $R$ 을 $\mathbf{q} = [w, x, y, z]^T$ 에 대해 편미분하고, $[\omega]_\times R$ 와 등치 조건을 비교하면:

```math
\dot{w} = \frac{1}{2}(-\omega_x x - \omega_y y - \omega_z z) \tag{C.4a}
```

```math
\dot{x} = \frac{1}{2}(\omega_x w + \omega_z y - \omega_y z) \tag{C.4b}
```

```math
\dot{y} = \frac{1}{2}(\omega_y w - \omega_z x + \omega_x z) \tag{C.4c}
```

```math
\dot{z} = \frac{1}{2}(\omega_z w + \omega_x y - \omega_y x) \tag{C.4d}
```

이를 행렬로 표현하면:

```math
\dot{\mathbf{q}} = \frac{1}{2} \Omega(\omega) \mathbf{q}, \quad \Omega(\omega) = \begin{bmatrix} 0 & -\omega_x & -\omega_y & -\omega_z \\ \omega_x & 0 & \omega_z & -\omega_y \\ \omega_y & -\omega_z & 0 & \omega_x \\ \omega_z & \omega_y & -\omega_x & 0 \end{bmatrix} \tag{C.5}
```

이것이 식 (9.3)의 행렬 형태다. $\quad \blacksquare$

---

## 부록 D. 공간 벡터의 좌표 변환 심층 분석

### D.1 프레임 A에서 프레임 B로의 변환

프레임 B가 프레임 A에 대해 회전 $R_{AB}$ 와 이동 $p_{AB}$ 로 정의될 때, 프레임 A의 공간 속도 $\mathbf{v}_A$ 를 프레임 B에서 표현하면:

```math
\mathbf{v}_B = X_{AB} \mathbf{v}_A \tag{D.1}
```

```math
X_{AB} = \begin{bmatrix} R_{AB} & 0 \\ -R_{AB}[p_{AB}]_\times & R_{AB} \end{bmatrix} \tag{D.2}
```

### D.2 변환 합성 (Transform Composition)

프레임 A → B → C 의 연속 변환:

```math
X_{AC} = X_{BC} \cdot X_{AB} \tag{D.3}
```

**증명**: $\mathbf{v}_C = X_{BC} \mathbf{v}_B = X_{BC}(X_{AB} \mathbf{v}_A) = (X_{BC} X_{AB}) \mathbf{v}_A$ $\quad \checkmark$

이는 순기구학 식 (3.4)에서 `X_world[i] = X_parent[i] @ X_world[parent]` 로 구현된다.

### D.3 힘 변환과 운동 변환의 쌍대성

운동 변환 $X$ 에 대해 힘 변환은 $X^{-T}$ 이다. 두 변환 행렬의 관계:

```math
X^{-T} = \begin{bmatrix} R & [p]_\times R \\ 0 & R \end{bmatrix} \tag{D.4}
```

**유도**: 식 (2.9)에서 $X^{-1}$ 을 구한 후 전치:

```math
X^{-1} = \begin{bmatrix} R^T & 0 \\ [p]_\times R^T & R^T \end{bmatrix}
```

전치하면:

```math
X^{-T} = (X^{-1})^T = \begin{bmatrix} R & R[p]_\times^T \\ 0 & R \end{bmatrix} = \begin{bmatrix} R & -R[p]_\times \\ 0 & R \end{bmatrix} \tag{D.5}
```

$[p]_\times^T = -[p]_\times$ 를 사용하면 식 (D.4)를 얻는다.

RNEA의 역전파에서 `X_up[i].T @ f[i]` 는 정확히 $X_{up,i}^T = X_{up,i}^{-T}$ (힘 변환의 역)를 사용함을 주목하라 — 즉 `X_up[i]`가 운동 변환이면 그 전치가 힘을 "역방향"으로 변환한다.

---

## 부록 E. 수치 구현 세부 사항

### E.1 Cholesky 분해와 선형 시스템 풀기

$M_{jj}$ 는 대칭 양정치 행렬이므로 Cholesky 분해를 사용한다:

```math
M_{jj} = L L^T \tag{E.1}
```

여기서 $L$ 은 하삼각 행렬(Lower Triangular Matrix)이다.

시스템 $M_{jj} x = b$ 풀기:
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

질량 행렬 $M_{jj}$ 는 설정 $q$ 에 의존하지만, 느리게 변화한다. 매 스텝마다 재계산하는 비용을 줄이기 위해 조건부 캐시를 사용한다:

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
- 질량 비 최대: $m_{pelvis}/m_{finger} \approx 10.6/0.3 \approx 35$
- 링크 길이 비: 상체(0.3m) vs 허벅지(0.35m)

정규화 항 추가로 완화: `M_jj += 1e-10 * np.eye(n_j)`

### F.2 LCP 수렴 조건

Fischer-Burmeister Newton 방법은 M이 **P-행렬(P-matrix)**일 때 전역 수렴이 보장된다.

**P-행렬 정의**: 모든 주부분행렬(Principal Submatrix)의 행렬식이 양수.

Delassus 행렬 $M_{LCP} = J_n M^{-1} J_n^T$ 의 경우:
- $M \succ 0$ 이고 $J_n$ 이 행 독립 → $M_{LCP} \succ 0$ → P-행렬 조건 만족
- 단, 접촉점이 기하학적으로 종속이면 $J_n$ 이 행 종속 → 정규화 필요

### F.3 보행 제어의 안정성

계산 토크 제어 (8.2절)에서 오차 동역학 (식 8.7)의 안정성:

특성 다항식 $s^2 + K_d s + K_p = 0$ 의 근:

```math
s_{1,2} = \frac{-K_d \pm \sqrt{K_d^2 - 4K_p}}{2} \tag{F.1}
```

THOR 다리 관절 ($K_p=600$, $K_d=60$):

```math
K_d^2 - 4K_p = 3600 - 2400 = 1200 > 0 \tag{F.2}
```

두 근 모두 실수이고 음수이므로 **과감쇠(Overdamped)** 수렴. 진동 없이 지수적 감소.

```math
s_{1,2} = \frac{-60 \pm \sqrt{1200}}{2} \approx \frac{-60 \pm 34.6}{2} = \{-12.7,\; -47.3\} \tag{F.3}
```

시정수: $\tau_1 = 1/12.7 \approx 79$ ms, $\tau_2 = 1/47.3 \approx 21$ ms.

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
| 스윙 지속 시간 $T_{swing}$ | 0.55 | s |
| 이중 지지 시간 $T_{DS}$ | 0.25 | s |
| 보행 사이클 $T_{cycle}$ | 0.80 | s |
| 엉덩이 신전각 $\theta_{ext}$ | -5 | ° |
| 엉덩이 굴곡각 $\theta_{flex}$ | 20 | ° |
| 무릎 스윙각 $\theta_{knee,swing}$ | 45 | ° |
| 발목 스윙각 $\theta_{ankle,swing}$ | 5 | ° |
| 다리 관절 $K_p$ | 600 | N·m/rad |
| 다리 관절 $K_d$ | 60 | N·m·s/rad |

---

> **수식 번호 색인**: 절 번호.수식 번호 형식 (예: (2.8) = 절 2의 8번째 수식). 크로스 레퍼런스 시 이 번호를 사용할 것.
>
> **구현 파일 대응**:
> - `thor/core/spatial/` → 절 2
> - `thor/model/kinematics.py`, `thor/model/quaternion.py` → 절 3, 절 1.3
> - `thor/dynamics/rnea.py` → 절 4, 5.1
> - `thor/dynamics/crba.py` → 절 5.2
> - `thor/dynamics/aba.py` → 절 5.3
> - `thor/dynamics/centroidal.py` → 절 6
> - `thor/dynamics/contact.py` → 절 7.1
> - `thor/optimization/lcp_solver.py` → 절 7.2–7.4
> - `thor/dynamics/contact_implicit.py` → 절 7.5
> - `thor/control/walking_controller.py` → 절 8.2–8.3
> - `thor/control/centroidal_lqr.py` → 절 6.4, 8.4
> - `thor/control/contact_implicit_mpc.py` → 절 8.1
> - `thor/dynamics/integrators.py`, `thor/model/quaternion.py` → 절 9
