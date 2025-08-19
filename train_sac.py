# train_sac.py (벡터 환경 버전)
import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from steak_env import SteakEnv

if __name__ == "__main__":
    N_ENVS = 8  # 코어 여유에 맞춰 조절 (예: 4~8)
    # doneness 인자를 각 프로세스에 전달
    venv = make_vec_env(
    SteakEnv,
    n_envs=8,
    env_kwargs={"doneness": "mr"},  # 기호모드: 미디움레어
)

# 관측·보상 정규화
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
)
    ckpt = CheckpointCallback(save_freq=50_000, save_path="./ckpts", name_prefix="sac_steak")
    model = SAC("MlpPolicy", venv, verbose=1, device="cpu",
                learning_rate=3e-4, batch_size=256, gamma=0.99,
                ent_coef='auto_0.5',   # ← 탐색 강하게
                tensorboard_log="./tb")


    # 주의: SB3의 total_timesteps는 "환경 스텝 총합"입니다.
    # n_envs=8이면 벽시계상 훨씬 빨리 끝납니다(한 스텝에 8 transition 수집).
    model.learn(total_timesteps=300_000, callback=ckpt)
    model.save("sac_steak_pref.zip")
    venv.close()
    print("Training done. Saved sac_steak_pref.zip")
