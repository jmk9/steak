import os, time, numpy as np
from stable_baselines3 import SAC
import mujoco as mj
from mujoco import viewer

XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <!-- 바닥 -->
    <geom type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>

    <!-- 팬 -->
    <body name="pan" pos="0 0 0">
      <geom type="box" size="0.15 0.15 0.005" rgba="0.2 0.2 0.2 1"/>
    </body>

    <!-- 스테이크 본체 (y축 힌지로 뒤집기) -->
    <body name="steak" pos="0 0 0.055">
      <joint name="steak_hinge" type="hinge" axis="0 1 0" limited="true" range="0 3.14159"/>
      <!-- 두께 0.01 → 중심 본체는 살짝 얇게, 상/하 표면을 아주 얇은 레이어로 분리 -->
      <geom name="steak_core"   type="box" size="0.06 0.04 0.008" rgba="0.90 0.40 0.45 1"/>
      <geom name="steak_top"    type="box" pos="0 0 0.0092" size="0.0601 0.0401 0.0009" rgba="0.90 0.40 0.45 1"/>
      <geom name="steak_bottom" type="box" pos="0 0 -0.0092" size="0.0601 0.0401 0.0009" rgba="0.90 0.40 0.45 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 색상 매핑: raw(분홍/붉은) → brown(갈색)로 선형 보간
RAW_COLOR   = np.array([0.90, 0.40, 0.45, 1.0])  # 생고기 느낌
BROWN_COLOR = np.array([0.40, 0.18, 0.08, 1.0])  # 잘 구운 갈색

def lerp_color(raw_rgba, brown_rgba, progress):
    x = float(np.clip(progress, 0.0, 1.0))
    return (1.0 - x) * raw_rgba + x * brown_rgba

class VizEnv:
    """
    시각화 전용 환경:
    - 상/하 표면 온도(Tsurf_top/bot), 갈변(cvar_top/bot), 내부(core) 분리
    - flip 신호로 y축 회전 (0 <-> pi), 상/하 표면 역할이 뒤바뀌도록 내부 상태 스왑
    - 상/하 표면의 갈변 정도에 따라 각각 다른 색으로 표시
    - HUD(윈도우 타이틀)에 T_core, T_surf_top, T_surf_bot 표시
    """
    def __init__(self):
        self.model = mj.MjModel.from_xml_string(XML)
        self.data = mj.MjData(self.model)

        # 인덱스 캐시
        self.jid     = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "steak_hinge")
        self.qadr    = self.model.jnt_qposadr[self.jid]
        self.geom_top= mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "steak_top")
        self.geom_bot= mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "steak_bottom")
        self.geom_core= mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "steak_core")

        # 내부 상태
        self.reset()

        # 뒤집기 토글 관련
        self.flip_target = 0.0
        self.last_flip = False

    def reset(self):
        # 초기: 내부/표면 모두 5°C, 갈변 0.0(완전 생고기색)
        self.T_core = 5.0
        self.T_top  = 5.0
        self.T_bot  = 5.0
        self.cvar_top = 0.0
        self.cvar_bot = 0.0
        self.t = 0.0

        # 초기 포즈/색
        self.data.qpos[self.qadr] = 0.0
        self.model.geom_rgba[self.geom_core] = RAW_COLOR
        self.model.geom_rgba[self.geom_top ] = RAW_COLOR
        self.model.geom_rgba[self.geom_bot ] = RAW_COLOR
        mj.mj_forward(self.model, self.data)
        return self.obs()

    def obs(self):
        # 관측 벡터(참고): [T_core, T_surf_mean, cvar_mean, t] 대신 본 시각화에서는 직접 항목별로 HUD 표시
        T_surf_mean = 0.5*(self.T_top + self.T_bot)
        cvar_mean   = 0.5*(self.cvar_top + self.cvar_bot)
        return np.array([self.T_core, T_surf_mean, cvar_mean, self.t], dtype=np.float32)

    def step(self, action):
        dt = 0.02
        # 행동 해석
        temp_set = 120 + (np.clip(action[0], -1, 1) + 1) * 0.5 * (260 - 120)  # 120~260°C
        flip = bool(action[1] > 0.5)

        # 열전달(간이): 팬은 하부(현재 "아래" 면)에 더 강하게 영향
        k_top, k_bot = 0.010, 0.020   # 상/하 가열 민감도 (팬 접촉: 하단↑)
        self.T_top += k_top * (temp_set - self.T_top) * dt
        self.T_bot += k_bot * (temp_set - self.T_bot) * dt
        # 내부 코어는 표면 평균 쪽으로 천천히 수렴
        self.T_core += 0.015 * ((self.T_top + self.T_bot)/2.0 - self.T_core) * dt

        # 갈변 진행: 표면 온도가 높을수록 증가 (150°C 이후 가속), 뒤집으면 해당 면 잠시 완화
        def update_brown(cvar, Tsurf, flipped):
            base = max(0.0, Tsurf - 150.0) * 1e-3 * dt
            if flipped:   # 뒤집기 시 열 분포 변화로 살짝 완화 효과
                cvar *= 0.95
            return float(np.clip(cvar + base, 0.0, 1.0))
        self.cvar_top = update_brown(self.cvar_top, self.T_top,  flip and not self.last_flip)
        self.cvar_bot = update_brown(self.cvar_bot, self.T_bot,  flip and not self.last_flip)

        # 뒤집기 토글: 상/하 역할 스왑 + 관측의 상/하 표면도 자연히 바뀌도록 상태 스왑
        if flip and not self.last_flip:
            self.flip_target = 0.0 if np.isclose(self.flip_target, np.pi, atol=1e-3) else np.pi
            # 상태 스왑(상/하 교체)
            self.T_top, self.T_bot = self.T_bot, self.T_top
            self.cvar_top, self.cvar_bot = self.cvar_bot, self.cvar_top
        self.last_flip = flip

        # 힌지 조인트 각도 보간(부드럽게 뒤집힘)
        curr = float(self.data.qpos[self.qadr])
        speed = 6.0  # rad/s
        self.data.qpos[self.qadr] = curr + np.clip(self.flip_target - curr, -speed*dt, speed*dt)

        # 색 업데이트: 상/하 표면 각각 raw→brown 보간, 코어도 평균 갈변으로 살짝 변하도록
        rgba_top = lerp_color(RAW_COLOR, BROWN_COLOR, self.cvar_top)
        rgba_bot = lerp_color(RAW_COLOR, BROWN_COLOR, self.cvar_bot)
        rgba_core= lerp_color(RAW_COLOR, BROWN_COLOR, 0.5*(self.cvar_top+self.cvar_bot)*0.6)  # 코어는 덜 변함
        self.model.geom_rgba[self.geom_top]  = rgba_top
        self.model.geom_rgba[self.geom_bot]  = rgba_bot
        self.model.geom_rgba[self.geom_core] = rgba_core

        # 물리계 갱신
        mj.mj_forward(self.model, self.data)

        # 시간 진행
        self.t += dt

        # HUD 텍스트용 데이터 반환
        return {
            "T_core": self.T_core,
            "T_top":  self.T_top,
            "T_bot":  self.T_bot,
            "c_top":  self.cvar_top,
            "c_bot":  self.cvar_bot,
            "temp_set": temp_set,
            "flip": int(flip),
            "t": self.t,
        }

if __name__ == "__main__":
    assert os.getenv("MUJOCO_GL","egl")=="glfw", "Set MUJOCO_GL=glfw and DISPLAY first."
    model = SAC.load("sac_steak.zip", device="cpu")

    env = VizEnv()
    env.reset()

    with viewer.launch_passive(env.model, env.data) as v:
        for k in range(2000):  # ~20초 재생
            # 관측은 내부에서 관리하므로 더미 벡터로 대체
            info = env.step(model.predict(np.zeros(4, dtype=np.float32), deterministic=True)[0])
            # HUD(윈도 타이틀) 업데이트: 내부 온도와 상/하 표면 온도를 명시적으로 분리 표시
            v._viewport_title = (
                f"t={info['t']:.1f}s | "
                f"Tcore={info['T_core']:.1f}°C, "
                f"Tsurf_top={info['T_top']:.1f}°C, "
                f"Tsurf_bot={info['T_bot']:.1f}°C | "
                f"flip={info['flip']} | Tset={info['temp_set']:.0f}°C"
            )
            v.sync()
            time.sleep(0.01)

    print("Playback done.")
