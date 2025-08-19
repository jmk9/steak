from steak_env import SteakEnv, PAN_SET_MIN, PAN_SET_MAX
import numpy as np

env = SteakEnv(doneness="mr", accel=60.0)  # 학습과 동일 60×
obs,_ = env.reset()

def step_fixed(Tset_C, flip=False, steps=600):  # 600스텝 = 현실 60초
    x = (Tset_C - PAN_SET_MIN)/(PAN_SET_MAX - PAN_SET_MIN)*2 - 1
    a = np.array([x, 1.0 if flip else 0.0], dtype=np.float32)
    for _ in range(steps):
        o,r,d,t,_ = env.step(a)
    return o

# 1) 바닥 접촉 상태로 60초 가열(예열 강화)
o = step_fixed(240, flip=False, steps=900)   # 현실 90초
print("phase1  Ttop=%.1f  Tbot=%.1f  c_top=%.6f  c_bot=%.6f" % (o[1], o[2], env.c_top, env.c_bot))

# 2) 뒤집고 다시 60초
o = step_fixed(240, flip=True, steps=1)
o = step_fixed(240, flip=False, steps=600)
print("phase2  Ttop=%.1f  Tbot=%.1f  c_top=%.6f  c_bot=%.6f" % (o[1], o[2], env.c_top, env.c_bot))
