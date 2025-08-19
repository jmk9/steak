import os, time, argparse, numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

from mujoco import viewer
from stable_baselines3 import SAC
from steak_env import SteakEnv, lerp_color, PAN_SET_MIN, PAN_SET_MAX

XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <geom type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>
    <body name="pan" pos="0 0 0">
      <geom type="box" size="0.15 0.15 0.005" rgba="0.2 0.2 0.2 1"/>
    </body>

    <body name="steak" pos="0 0 0.055">
      <joint name="steak_hinge" type="hinge" axis="0 1 0" limited="true" range="0 3.14159"/>
      <geom name="steak_core"   type="box" size="0.06 0.04 0.008" rgba="0.90 0.40 0.45 1"/>
      <geom name="steak_top"    type="box" pos="0 0 0.0092"  size="0.0601 0.0401 0.0009" rgba="0.90 0.40 0.45 1"/>
      <geom name="steak_bot"    type="box" pos="0 0 -0.0092" size="0.0601 0.0401 0.0009" rgba="0.90 0.40 0.45 1"/>

      <!-- 라벨(①/II) 지오메트리 -->
      <geom name="label_top_bg" type="box" pos="0.05 0.028 0.0103" size="0.010 0.007 0.0003" rgba="1 1 1 0.85"/>
      <geom name="label_bot_bg" type="box" pos="0.05 -0.028 -0.0103" size="0.010 0.007 0.0003" rgba="1 1 1 0.85"/>

      <!-- '1' (윗면) -->
      <geom name="label_top_1"  type="box" pos="0.050  0.028  0.0106" size="0.0012 0.0048 0.0002" rgba="0 0 0 1"/>

      <!-- 'II' (아랫면) : 막대 2개 -->
      <geom name="label_bot_2L" type="box" pos="0.0478 -0.028 -0.0106" size="0.0012 0.0048 0.0002" rgba="0 0 0 1"/>
      <geom name="label_bot_2R" type="box" pos="0.0522 -0.028 -0.0106" size="0.0012 0.0048 0.0002" rgba="0 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

def init_hud():
    plt.ion()
    fig, ax = plt.subplots(figsize=(5,3))
    ax.axis('off')
    t_top = ax.text(0.02, 0.80, "", fontsize=18)
    t_bot = ax.text(0.02, 0.60, "", fontsize=18)
    t_core= ax.text(0.02, 0.40, "", fontsize=18)
    t_pan = ax.text(0.02, 0.20, "", fontsize=18)
    t_misc= ax.text(0.02, 0.05, "", fontsize=12)
    fig.canvas.manager.set_window_title("Steak HUD")
    fig.tight_layout()
    return fig, (t_top, t_bot, t_core, t_pan, t_misc)

def update_hud(fig, texts, T_top, T_bot, T_core, T_pan, flips, t_real, c_top, c_bot):
    t_top, t_bot, t_core, t_pan, t_misc = texts
    t_top.set_text(f"① TOP surface : {T_top:6.1f} °C   c_top={c_top:.3f}")
    t_bot.set_text(f"II BOTTOM surf: {T_bot:6.1f} °C   c_bot={c_bot:.3f}")
    t_core.set_text(f"CORE (center) : {T_core:6.1f} °C")
    t_pan.set_text(f"PAN           : {T_pan:6.1f} °C")
    t_misc.set_text(f"time(real) {t_real:6.1f}s   flips {flips:d}")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()




def main():
    assert os.getenv("MUJOCO_GL","egl")=="glfw", "Set MUJOCO_GL=glfw and DISPLAY first."

    parser = argparse.ArgumentParser()
    parser.add_argument("--doneness", default="mr", choices=["rare","mr","medium","mw","well"])
    parser.add_argument("--scripted", action="store_true", help="Ignore policy; run a fixed heating+flip demo")
    parser.add_argument("--model", default="sac_steak_pref.zip")
    parser.add_argument("--realtime", action="store_true", help="1× (real time) visualization")
    parser.add_argument("--force-run", action="store_true", help="Ignore done/trunc for 10s to keep window open")
    args = parser.parse_args()

    acc = 1.0 if args.realtime else 60.0
    env = SteakEnv(doneness=args.doneness, accel=acc)
    obs,_ = env.reset()
    fig, hud_texts = init_hud()

    use_policy = not args.scripted
    model = None
    if use_policy:
        try:
            model = SAC.load(args.model, device="cpu")
        except Exception as e:
            print("Model load failed, fallback to scripted:", e)
            use_policy = False

    m = mj.MjModel.from_xml_string(XML)
    d = mj.MjData(m)
    j_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "steak_hinge")
    qadr = m.jnt_qposadr[j_id]
    g_top = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "steak_top")
    g_bot = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "steak_bot")
    g_core= mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "steak_core")

    # 초기 색
    m.geom_rgba[g_top]  = np.array([0.90, 0.40, 0.45, 1.0])
    m.geom_rgba[g_bot]  = np.array([0.90, 0.40, 0.45, 1.0])
    m.geom_rgba[g_core] = np.array([0.90, 0.40, 0.45, 1.0])

    start_wall = time.time()
    
    with viewer.launch_passive(m, d) as v:
        # 1×(--realtime) = 프레임당 1스텝, 60×(기본) = 프레임당 60스텝
        steps_per_frame = 1 if args.realtime else 10

        flip_target = 0.0
        last_seen_flips = int(obs[11])  # 환경의 누적 flip 카운트 기준
        scripted_timer = 0.0
        scripted_flip_period_real = 2.0  # [s] 현실 기준

        for _ in range(100000):  # 충분히 크게
            done = False
            trunc = False

            # ---- 시뮬레이션 여러 스텝(배속) ----
            for _inner in range(steps_per_frame):
                if use_policy and model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # 스크립트 모드: 간단 고정 Tset + 주기적 뒤집기
                    t_real = obs[10]
                    Tset_norm = (240.0 - PAN_SET_MIN) / (PAN_SET_MAX - PAN_SET_MIN) * 2.0 - 1.0
                    do_flip = 1.0 if (t_real - scripted_timer) >= scripted_flip_period_real else 0.0
                    if do_flip > 0.5:
                        scripted_timer = t_real
                    action = np.array([Tset_norm, do_flip], dtype=np.float32)

                obs, r, done, trunc, _ = env.step(action)
                if done or trunc:
                    break

            # ---- 한 번만 렌더/표시 ----
            # 색상 (누적 갈변도 기반)
            m.geom_rgba[g_top]  = lerp_color(env.c_top)
            m.geom_rgba[g_bot]  = lerp_color(env.c_bot)
            m.geom_rgba[g_core] = 0.5*(m.geom_rgba[g_top] + m.geom_rgba[g_bot])

            # flip 애니메이션: 환경 flips 증가 순간에만 0↔π 토글 + 스냅
            flips = int(obs[11])
            if flips > last_seen_flips:
                flip_target = 0.0 if np.isclose(flip_target, np.pi, atol=1e-3) else np.pi
                last_seen_flips = flips

            curr = float(d.qpos[qadr])
            speed = 10.0  # rad/s
            step = np.clip(flip_target - curr, -speed*0.01, speed*0.01)
            d.qpos[qadr] = curr + step
            if abs(d.qpos[qadr] - flip_target) < 0.02:  # ≈1.1°
                d.qpos[qadr] = flip_target

            mj.mj_forward(m, d)

            # 외부 HUD(항상 표시)
            T_core, T_top, T_bot = float(obs[0]), float(obs[1]), float(obs[2])
            T_pan, t_real        = float(obs[9]), float(obs[10])
            update_hud(fig, hud_texts, T_top, T_bot, T_core, T_pan, flips, t_real, env.c_top, env.c_bot)

            v.sync()

            # 1×일 때만 약간 슬립; 60×는 슬립 없음(빠르게 재생)
            if args.realtime:
                time.sleep(0.01)

            # 렌더 한 번은 하고 나서 종료
            if done or trunc:
                break

    print("Playback done.")

if __name__ == "__main__":
    # 필수 환경변수 재확인 (실행 중 덮어쓰기 방지)
    if os.getenv("MUJOCO_GL") != "glfw":
        print("Hint: export MUJOCO_GL=glfw  and DISPLAY=host.docker.internal:0")
    main()
