"""
Модель гексапода из hexapod_wave.py, оформленная как модуль с параметром DUTY,
чтобы её можно было импортировать в исследовательские скрипты (заданиям 3 и 4).
Формулы foot_trajectory и legs_state — те же, что в исходном листинге.
Дополнительно: координата y стопы (вид сверху) и флаг «нога в опоре».
"""
import numpy as np

N_LEGS   = 6
STEP_LEN = 0.6
SWING_H  = 0.25
T_STEP   = 1.0
HIP_X    = np.array([0.6, 0.0, -0.6, 0.6, 0.0, -0.6])
HIP_SIDE = np.array([+1, +1, +1, -1, -1, -1])   # ноги 0,1,2 — левые; 3,4,5 — правые
FOOT_Y   = 0.6          # боковой вынос стопы от продольной оси (для вида сверху)
BODY_W   = 0.25         # полуширина корпуса (для рисунка)

PHASE_ORDER = np.array([0, 2, 4, 1, 3, 5])     # как в исходном коде


def phase_offsets(duty, scheme='original'):
    """Фазовые сдвиги ног.
    'original' — формула из методички: PHASE_OFFSET = k·DUTY/(N_LEGS-1),
                 k — позиция ноги в PHASE_ORDER;
    'tripod'   — классический трипод: ноги {0, 2, 4} и {1, 3, 5} в противофазе
                 (левая передняя, левая задняя, правая средняя — одна группа)."""
    if scheme == 'original':
        by_rank = np.arange(N_LEGS) * duty / (N_LEGS - 1)
        return np.array([by_rank[np.where(PHASE_ORDER == i)[0][0]]
                         for i in range(N_LEGS)])
    if scheme == 'tripod':
        return np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
    raise ValueError(scheme)


def foot_trajectory(phase, duty):
    """(x, z) стопы в системе координат ноги — как в исходном листинге."""
    if phase < duty:                     # опора: стопа идёт назад
        s = phase / duty
        return STEP_LEN * (0.5 - s), 0.0
    s = (phase - duty) / (1 - duty)      # перенос: стопа идёт вперёд по дуге
    return STEP_LEN * (-0.5 + s), SWING_H * np.sin(np.pi * s)


def legs_state(t, duty, scheme='original'):
    """Позиции всех стоп: xs, ys (вид сверху), zs (высота) и маска опоры."""
    off = phase_offsets(duty, scheme)
    xs, ys, zs = np.zeros(N_LEGS), FOOT_Y * HIP_SIDE.astype(float), np.zeros(N_LEGS)
    stance = np.zeros(N_LEGS, dtype=bool)
    for i in range(N_LEGS):
        phase = (t / T_STEP + off[i]) % 1.0
        fx, fz = foot_trajectory(phase, duty)
        xs[i] = HIP_X[i] + fx
        zs[i] = fz
        stance[i] = phase < duty
    return xs, ys, zs, stance


def n_swing(t, duty, scheme='original'):
    """Число ног в воздухе (фаза переноса) в момент t."""
    return int(np.sum(~legs_state(t, duty, scheme)[3]))
